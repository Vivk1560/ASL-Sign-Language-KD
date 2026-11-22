"""Loading, verifying and freezing the ResNet50 teacher for distillation.

The teacher implementation and checkpoint format are NOT duplicated: loading is delegated to
``teacher_model.checkpoint.load_teacher``. This module only adds the distillation-specific guarantees
(frozen, eval mode, no gradients) and the teacher/student compatibility checks.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from teacher_model.checkpoint import load_teacher
from teacher_model.config import IMAGENET_MEAN, IMAGENET_STD

LOG = logging.getLogger(__name__)


class TeacherCompatibilityError(ValueError):
    """The teacher checkpoint cannot be used to distil into this student / dataset."""


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_teacher(teacher: nn.Module) -> nn.Module:
    """Put the teacher in eval mode and disable gradients for every parameter."""
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher.eval()


def verify_teacher_compatibility(teacher: nn.Module, ckpt: dict[str, Any], class_names: list[str],
                                 class_to_idx: dict[str, int], image_size: int) -> None:
    """Fail loudly (no silent fallback) unless the teacher matches the dataset labels and the student input."""
    if list(ckpt.get("class_names", [])) != list(class_names):
        raise TeacherCompatibilityError(
            "Teacher class list differs from labels.json (names or order); the student would learn a "
            "different class-index mapping than the teacher.")
    if dict(ckpt.get("class_to_idx", {})) != dict(class_to_idx):
        raise TeacherCompatibilityError("Teacher class_to_idx differs from labels.json.")
    out_features = getattr(getattr(teacher, "fc", None), "out_features", None)
    if out_features != len(class_names):
        raise TeacherCompatibilityError(
            f"Teacher outputs {out_features} classes but labels.json defines {len(class_names)}.")
    if int(ckpt.get("input_size", image_size)) != int(image_size):
        raise TeacherCompatibilityError(
            f"Teacher was trained at image size {ckpt.get('input_size')} but the student is configured for "
            f"{image_size}. Both must see identical inputs; set --image-size {ckpt.get('input_size')}.")
    norm = ckpt.get("normalization")
    if norm is not None and (tuple(norm.get("mean", ())) != IMAGENET_MEAN
                             or tuple(norm.get("std", ())) != IMAGENET_STD):
        raise TeacherCompatibilityError(
            "Teacher checkpoint normalisation differs from the project's ImageNet mean/std, which the shared "
            "data pipeline applies to both models.")


def teacher_summary(path: Path, ckpt: dict[str, Any]) -> dict[str, Any]:
    """Verifiable reference to the teacher used (file name + content hash, not a machine-specific path)."""
    return {"file_name": Path(path).name, "sha256": file_sha256(path),
            "architecture": ckpt.get("architecture"), "epoch": ckpt.get("epoch"),
            "best_val_metric": ckpt.get("best_val_metric"), "best_metric_name": ckpt.get("best_metric_name")}


def load_frozen_teacher(path: Path, device: str | torch.device, class_names: list[str],
                        class_to_idx: dict[str, int], image_size: int
                        ) -> tuple[nn.Module, dict[str, Any]]:
    """Load the teacher checkpoint, verify compatibility, freeze it. Returns ``(teacher, teacher_summary)``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {path}. Train the teacher first (python -m teacher_model.train) "
            "or pass --teacher-checkpoint <path>.")
    teacher, ckpt = load_teacher(path, device)     # validates architecture; never mutates the file
    verify_teacher_compatibility(teacher, ckpt, class_names, class_to_idx, image_size)
    freeze_teacher(teacher)
    return teacher, teacher_summary(path, ckpt)


@torch.no_grad()
def teacher_logits(teacher: nn.Module, images: torch.Tensor, use_amp: bool = False) -> torch.Tensor:
    """Gradient-free teacher forward pass. Re-asserts eval mode on every call so a stray ``.train()`` on a
    parent module can never switch BatchNorm to batch statistics or update running stats."""
    teacher.eval()
    with torch.autocast(device_type=images.device.type, enabled=use_amp):
        return teacher(images).detach()
