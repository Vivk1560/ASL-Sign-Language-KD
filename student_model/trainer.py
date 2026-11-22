"""Distillation training loop: frozen ResNet50 teacher -> MobileNetV2 student.

Optimizer construction, the validation pass (``run_epoch``), early stopping and last-checkpoint naming are
reused from ``teacher_model.trainer`` so student and teacher follow identical conventions. Only the
training step differs: both models see the same augmented batch, the teacher runs without gradients and
the student minimises ``DistillationLoss``. Validation uses plain cross-entropy so ``val_loss`` is directly
comparable with the teacher's. The test loader is never touched here.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from teacher_model.checkpoint import check_resume_compatible
from teacher_model.trainer import EarlyStopping, build_optimizer, last_checkpoint_path, run_epoch

from .checkpoint import read_student_checkpoint, save_student_checkpoint
from .config import StudentConfig
from .distillation import DistillationLoss
from .teacher import freeze_teacher, teacher_logits

LOG = logging.getLogger(__name__)


def train_epoch(student: nn.Module, teacher: nn.Module, loader: DataLoader, criterion: DistillationLoss,
                optimizer: torch.optim.Optimizer, device: torch.device, *, scaler: Any = None,
                use_amp: bool = False, max_batches: int | None = None, log_every: int = 50
                ) -> dict[str, float]:
    """One distillation epoch. Only student parameters are updated."""
    student.train()
    teacher.eval()
    totals = {"loss": 0.0, "ce": 0.0, "kd": 0.0}
    correct = teacher_correct = seen = 0
    for step, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        t_logits = teacher_logits(teacher, images, use_amp)          # no grad, eval mode
        with torch.autocast(device_type=device.type, enabled=use_amp):
            s_logits = student(images)
        out = criterion(s_logits, t_logits, targets)                  # float32 inside
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(out.total).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out.total.backward()
            optimizer.step()
        n = targets.size(0)
        totals["loss"] += out.total.item() * n
        totals["ce"] += out.ce.item() * n
        totals["kd"] += out.kd.item() * n
        correct += (s_logits.argmax(dim=1) == targets).sum().item()
        teacher_correct += (t_logits.argmax(dim=1) == targets).sum().item()
        seen += n
        if log_every and step % log_every == 0:
            LOG.info("  [train] batch %d/%d loss=%.4f ce=%.4f kd=%.4f acc=%.4f", step, len(loader),
                     totals["loss"] / seen, totals["ce"] / seen, totals["kd"] / seen, correct / seen)
        if max_batches is not None and step >= max_batches:
            break
    if seen == 0:
        raise RuntimeError("No samples processed in the train loop.")
    return {"loss": totals["loss"] / seen, "ce": totals["ce"] / seen, "kd": totals["kd"] / seen,
            "accuracy": correct / seen, "teacher_accuracy": teacher_correct / seen}


def fit(student: nn.Module, teacher: nn.Module, loaders: dict[str, DataLoader], cfg: StudentConfig,
        device: torch.device, class_names: list[str], class_to_idx: dict[str, int],
        teacher_info: dict[str, Any]) -> dict[str, Any]:
    """Train with validation-based model selection; writes best and last student checkpoints."""
    student.to(device)
    teacher.to(device)
    freeze_teacher(teacher)
    criterion = DistillationLoss(cfg.temperature, cfg.alpha, cfg.label_smoothing)
    val_criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(student, cfg)               # duck-typed: needs optimizer/learning_rate/weight_decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)
    use_amp = bool(cfg.mixed_precision and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    mode = "min" if cfg.monitor == "val_loss" else "max"
    stopper = EarlyStopping(mode, cfg.early_stopping_patience, cfg.early_stopping_min_delta)
    history: list[dict[str, Any]] = []
    start_epoch, best_epoch = 1, 0
    config_dict = cfg.to_dict()
    distillation = {"temperature": cfg.temperature, "alpha": cfg.alpha, "label_smoothing": cfg.label_smoothing}
    best_path, last_path = cfg.checkpoint_path, last_checkpoint_path(cfg.checkpoint_path)

    if cfg.resume:
        ckpt = read_student_checkpoint(cfg.resume, map_location="cpu")
        check_resume_compatible(ckpt, class_names, cfg.image_size)
        if ckpt.get("teacher", {}).get("sha256") != teacher_info["sha256"]:
            raise ValueError("Cannot resume: the checkpoint was distilled from a different teacher "
                             "(SHA-256 mismatch). Use the original teacher checkpoint.")
        old = ckpt.get("distillation", {})
        if (old.get("temperature"), old.get("alpha")) != (cfg.temperature, cfg.alpha):
            LOG.warning("Resuming with different KD settings (checkpoint: %s, now: T=%s alpha=%s).",
                        old, cfg.temperature, cfg.alpha)
        student.load_state_dict(ckpt["model_state_dict"])
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

    LOG.info("Distilling on %s | AMP=%s | T=%s alpha=%s | optimizer=%s | monitor=%s (%s)", device, use_amp,
             cfg.temperature, cfg.alpha, cfg.optimizer, cfg.monitor, mode)
    stopped_early = False
    started = time.time()
    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        tr = train_epoch(student, teacher, loaders["train"], criterion, optimizer, device, scaler=scaler,
                         use_amp=use_amp, max_batches=cfg.max_train_batches)
        val_loss, val_acc = run_epoch(student, loaders["validation"], val_criterion, device,
                                      use_amp=use_amp, log_every=0, tag="val")
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        record = {"epoch": epoch, "train_loss": tr["loss"], "train_ce": tr["ce"], "train_kd": tr["kd"],
                  "train_accuracy": tr["accuracy"], "teacher_train_accuracy": tr["teacher_accuracy"],
                  "val_loss": val_loss, "val_accuracy": val_acc, "lr": lr,
                  "seconds": round(time.time() - t0, 1)}
        history.append(record)
        monitored = val_loss if cfg.monitor == "val_loss" else val_acc
        improved = stopper.step(monitored)
        LOG.info("Epoch %d/%d | train loss %.4f (ce %.4f kd %.4f) acc %.4f | val loss %.4f acc %.4f | "
                 "lr %.2e | %s", epoch, cfg.epochs, tr["loss"], tr["ce"], tr["kd"], tr["accuracy"], val_loss,
                 val_acc, lr, "improved" if improved else
                 f"no improvement ({stopper.bad_epochs}/{stopper.patience})")
        if improved:
            best_epoch = epoch
        common = dict(model=student, epoch=epoch, class_names=class_names, class_to_idx=class_to_idx,
                      config=config_dict, history=history, best_metric_name=cfg.monitor,
                      distillation=distillation, teacher=teacher_info)
        extra = {"best_epoch": best_epoch, "best_monitor_value": stopper.best,
                 "val_loss": val_loss, "val_accuracy": val_acc}
        if improved:
            save_student_checkpoint(best_path, best_val_metric=monitored, optimizer=optimizer,
                                    scheduler=scheduler, scaler=scaler, extra=extra, **common)
        save_student_checkpoint(last_path, best_val_metric=stopper.best, optimizer=optimizer,
                                scheduler=scheduler, scaler=scaler, extra=extra, **common)
        if stopper.should_stop:
            LOG.info("Early stopping: no improvement for %d epochs.", stopper.bad_epochs)
            stopped_early = True
            break

    return {"history": history, "best_epoch": best_epoch, "best_monitor_value": stopper.best,
            "monitor": cfg.monitor, "stopped_early": stopped_early, "epochs_run": len(history),
            "train_seconds": round(time.time() - started, 1),
            "best_checkpoint": str(best_path), "last_checkpoint": str(last_path)}
