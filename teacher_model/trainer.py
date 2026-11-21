"""Mini-batch training loop: AdamW/SGD, cosine LR schedule, AMP (CUDA), early stopping, resume."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .checkpoint import check_resume_compatible, read_checkpoint, save_checkpoint
from .config import TeacherConfig

LOG = logging.getLogger(__name__)


def build_optimizer(model: nn.Module, cfg: TeacherConfig) -> torch.optim.Optimizer:
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    return torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9,
                           weight_decay=cfg.weight_decay, nesterov=True)


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, *,
              optimizer: torch.optim.Optimizer | None = None, scaler: Any = None,
              use_amp: bool = False, max_batches: int | None = None,
              log_every: int = 50, tag: str = "train") -> tuple[float, float]:
    """One pass over ``loader``. Trains if ``optimizer`` is given, otherwise evaluates. -> (loss, acc)."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for step, (images, targets) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            n = targets.size(0)
            total_loss += loss.item() * n
            correct += (logits.argmax(dim=1) == targets).sum().item()
            seen += n
            if log_every and step % log_every == 0:
                LOG.info("  [%s] batch %d/%d loss=%.4f acc=%.4f", tag, step, len(loader),
                         total_loss / seen, correct / seen)
            if max_batches is not None and step >= max_batches:
                break
    if seen == 0:
        raise RuntimeError(f"No samples processed in the {tag} loop.")
    return total_loss / seen, correct / seen


class EarlyStopping:
    """Tracks the monitored metric; ``improved`` is True when it beats the best by ``min_delta``."""

    def __init__(self, mode: str, patience: int, min_delta: float) -> None:
        self.mode, self.patience, self.min_delta = mode, patience, min_delta
        self.best: float | None = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        better = (self.best is None or
                  (value < self.best - self.min_delta if self.mode == "min"
                   else value > self.best + self.min_delta))
        if better:
            self.best, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        return better

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


def last_checkpoint_path(best_path: Path) -> Path:
    return best_path.with_name(best_path.stem.replace("_best", "") + "_last" + best_path.suffix)


def fit(model: nn.Module, loaders: dict[str, DataLoader], cfg: TeacherConfig, device: torch.device,
        class_names: list[str], class_to_idx: dict[str, int]) -> dict[str, Any]:
    """Train with validation-based model selection. The test loader is NOT touched here."""
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)
    use_amp = bool(cfg.mixed_precision and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    mode = "min" if cfg.monitor == "val_loss" else "max"
    stopper = EarlyStopping(mode, cfg.early_stopping_patience, cfg.early_stopping_min_delta)
    history: list[dict[str, Any]] = []
    start_epoch, best_epoch = 1, 0
    config_dict = cfg.to_dict()
    best_path, last_path = cfg.checkpoint_path, last_checkpoint_path(cfg.checkpoint_path)

    if cfg.resume:
        ckpt = read_checkpoint(cfg.resume, map_location="cpu")
        check_resume_compatible(ckpt, class_names, cfg.image_size)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        history = list(ckpt.get("history", []))
        start_epoch = int(ckpt["epoch"]) + 1
        best_epoch = int(ckpt.get("extra", {}).get("best_epoch", ckpt["epoch"]))
        stopper.best = float(ckpt.get("extra", {}).get("best_monitor_value", ckpt["best_val_metric"]))
        LOG.info("Resumed from %s at epoch %d", cfg.resume, start_epoch)

    LOG.info("Training on %s | AMP=%s | optimizer=%s | monitor=%s (%s)",
             device, use_amp, cfg.optimizer, cfg.monitor, mode)
    stopped_early = False
    started = time.time()
    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, device, optimizer=optimizer,
                                          scaler=scaler, use_amp=use_amp,
                                          max_batches=cfg.max_train_batches, tag="train")
        val_loss, val_acc = run_epoch(model, loaders["validation"], criterion, device,
                                      use_amp=use_amp, log_every=0, tag="val")
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        record = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                  "val_loss": val_loss, "val_accuracy": val_acc, "lr": lr,
                  "seconds": round(time.time() - t0, 1)}
        history.append(record)
        monitored = val_loss if cfg.monitor == "val_loss" else val_acc
        improved = stopper.step(monitored)
        LOG.info("Epoch %d/%d | train loss %.4f acc %.4f | val loss %.4f acc %.4f | lr %.2e | %s",
                 epoch, cfg.epochs, train_loss, train_acc, val_loss, val_acc, lr,
                 "improved" if improved else f"no improvement ({stopper.bad_epochs}/{stopper.patience})")
        if improved:
            best_epoch = epoch
        common = dict(model=model, epoch=epoch, class_names=class_names, class_to_idx=class_to_idx,
                      config=config_dict, history=history, best_metric_name=cfg.monitor)
        extra = {"best_epoch": best_epoch, "best_monitor_value": stopper.best,
                 "val_loss": val_loss, "val_accuracy": val_acc}
        if improved:
            save_checkpoint(best_path, best_val_metric=monitored, optimizer=optimizer,
                            scheduler=scheduler, scaler=scaler, extra=extra, **common)
        save_checkpoint(last_path, best_val_metric=stopper.best, optimizer=optimizer,
                        scheduler=scheduler, scaler=scaler, extra=extra, **common)
        if stopper.should_stop:
            LOG.info("Early stopping: no improvement for %d epochs.", stopper.bad_epochs)
            stopped_early = True
            break

    return {"history": history, "best_epoch": best_epoch, "best_monitor_value": stopper.best,
            "monitor": cfg.monitor, "stopped_early": stopped_early,
            "epochs_run": len(history), "train_seconds": round(time.time() - started, 1),
            "best_checkpoint": str(best_path), "last_checkpoint": str(last_path)}
