"""Student checkpoint save / load.

Same layout as the teacher checkpoint (``teacher_model.checkpoint``) so tooling stays uniform, but with
``architecture == "mobilenet_v2"`` and extra ``distillation`` / ``teacher`` metadata. A separate module is
needed because the teacher's ``save_checkpoint`` / ``read_checkpoint`` hard-code the ResNet50 architecture.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from teacher_model.config import IMAGENET_MEAN, IMAGENET_STD

from .config import ARCHITECTURE_ID
from .model import build_student

CHECKPOINT_FORMAT_VERSION = 1


def save_student_checkpoint(path: Path, model: nn.Module, *, epoch: int, best_val_metric: float,
                            best_metric_name: str, class_names: list[str], class_to_idx: dict[str, int],
                            config: dict[str, Any], distillation: dict[str, Any],
                            teacher: dict[str, Any], optimizer: torch.optim.Optimizer | None = None,
                            scheduler: Any = None, scaler: Any = None,
                            history: list[dict[str, Any]] | None = None,
                            extra: dict[str, Any] | None = None) -> Path:
    """Atomically write a checkpoint with the weights plus everything needed to rebuild / verify the model."""
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
        "distillation": distillation,   # temperature, alpha, label_smoothing
        "teacher": teacher,             # file name, sha256, architecture, epoch, metric of the teacher used
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


def read_student_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a student checkpoint dict (``weights_only=True``: plain tensors / primitives only)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"{path} is not a student_model checkpoint (no 'model_state_dict').")
    if ckpt.get("architecture") != ARCHITECTURE_ID:
        raise ValueError(f"Checkpoint architecture {ckpt.get('architecture')!r} != {ARCHITECTURE_ID!r}; "
                         "teacher checkpoints must be loaded with teacher_model.checkpoint.")
    if ckpt.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported student checkpoint format_version {ckpt.get('format_version')!r}.")
    return ckpt


def load_student(path: Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild MobileNetV2 (no ImageNet download), load weights, return ``(model.eval(), checkpoint)``.
    This is the entry point for inference / later integration."""
    ckpt = read_student_checkpoint(path, map_location="cpu")
    dropout = float(ckpt.get("config", {}).get("dropout", 0.2))
    model = build_student(num_classes=ckpt["num_classes"], pretrained=False, dropout=dropout)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt
