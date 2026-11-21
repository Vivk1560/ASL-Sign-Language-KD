"""Checkpoint save / load (state_dict based, `.pth`). Compatible with later student / KD code."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import ARCHITECTURE_ID, IMAGENET_MEAN, IMAGENET_STD
from .model import build_teacher

LOG = logging.getLogger(__name__)
CHECKPOINT_FORMAT_VERSION = 1


def save_checkpoint(path: Path, model: nn.Module, *, epoch: int, best_val_metric: float,
                    best_metric_name: str, class_names: list[str], class_to_idx: dict[str, int],
                    config: dict[str, Any], optimizer: torch.optim.Optimizer | None = None,
                    scheduler: Any = None, scaler: Any = None,
                    history: list[dict[str, Any]] | None = None,
                    extra: dict[str, Any] | None = None) -> Path:
    """Atomically write a checkpoint holding the weights plus everything needed to rebuild the model."""
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": ARCHITECTURE_ID,
        "num_classes": len(class_names),
        "class_names": list(class_names),
        "class_to_idx": dict(class_to_idx),
        "input_size": int(config.get("image_size", 224)),
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "epoch": int(epoch),
        "best_val_metric": float(best_val_metric),
        "best_metric_name": best_metric_name,
        "config": config,
        "history": history or [],
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
        payload["scaler_state_dict"] = scaler.state_dict()
    if extra:
        payload["extra"] = extra
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def read_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint dict. Uses ``weights_only=True`` where supported (plain tensors/primitives only)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # very old torch without weights_only
        ckpt = torch.load(path, map_location=map_location)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"{path} is not a teacher_model checkpoint (no 'model_state_dict').")
    if ckpt.get("architecture") != ARCHITECTURE_ID:
        raise ValueError(f"Checkpoint architecture {ckpt.get('architecture')!r} != {ARCHITECTURE_ID!r}")
    return ckpt


def load_teacher(path: Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild ResNet50 (no ImageNet download needed), load weights, return ``(model.eval(), checkpoint)``."""
    ckpt = read_checkpoint(path, map_location="cpu")
    model = build_teacher(num_classes=ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt


def check_resume_compatible(ckpt: dict[str, Any], class_names: list[str], image_size: int) -> None:
    if list(ckpt["class_names"]) != list(class_names):
        raise ValueError("Cannot resume: checkpoint class list differs from the current labels.json.")
    if int(ckpt.get("input_size", image_size)) != image_size:
        raise ValueError(f"Cannot resume: checkpoint image size {ckpt.get('input_size')} != {image_size}.")
