"""Checkpoint-backed image classifier shared by the CLI, comparison, benchmark and webcam code.

Reuses ``student_model.checkpoint.load_student`` / ``teacher_model.checkpoint.load_teacher`` and
``teacher_model.augmentation.EvalTransform`` (the exact transform used for validation/test), so inference
preprocessing cannot drift from evaluation preprocessing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn

from preprocessing.config import PROJECT_ROOT
from teacher_model.augmentation import EvalTransform
from teacher_model.config import IMAGENET_MEAN, IMAGENET_STD
from teacher_model.utils import resolve_device

# Checkpoint locations are configurable: CLI flag > environment variable > these defaults.
# The recovered checkpoints are local artifacts and are NOT part of the Git repository.
ENV_STUDENT_CHECKPOINT = "ASL_STUDENT_CHECKPOINT"
ENV_TEACHER_CHECKPOINT = "ASL_TEACHER_CHECKPOINT"
DEFAULT_STUDENT_CHECKPOINT = PROJECT_ROOT / "student_output" / "student_mobilenetv2_kd_best.pth"
DEFAULT_TEACHER_CHECKPOINT = PROJECT_ROOT / "model_weights" / "teacher_resnet50_best.pth"

# Pre-Partition-2/3 checkpoints. They are raw state_dicts of a different (26-class) wrapper model.
LEGACY_CHECKPOINT_NAMES = frozenset({"teacher_best_aug.pth", "student_best_aug.pth"})
MODEL_KINDS = ("student", "teacher")


class DeploymentError(Exception):
    """Base class for errors that the CLIs report as a one-line message (no traceback)."""


class CheckpointNotFoundError(DeploymentError):
    pass


class CheckpointError(DeploymentError):
    pass


class InvalidImageError(DeploymentError):
    pass


def default_checkpoint(kind: str) -> Path:
    """Configured default checkpoint path: environment variable if set, else the documented default."""
    if kind == "student":
        return Path(os.environ.get(ENV_STUDENT_CHECKPOINT) or DEFAULT_STUDENT_CHECKPOINT)
    if kind == "teacher":
        return Path(os.environ.get(ENV_TEACHER_CHECKPOINT) or DEFAULT_TEACHER_CHECKPOINT)
    raise ValueError(f"kind must be one of {MODEL_KINDS}, got {kind!r}")


@dataclass(frozen=True)
class Prediction:
    label: str
    index: int
    confidence: float                       # softmax probability of ``label``
    top_k: tuple[tuple[str, float], ...] = field(default_factory=tuple)
    probabilities: tuple[float, ...] = field(default_factory=tuple, repr=False)


def check_checkpoint_path(path: Path, kind: str) -> Path:
    path = Path(path)
    if path.name in LEGACY_CHECKPOINT_NAMES:
        raise CheckpointError(
            f"{path.name} is a legacy pre-Partition-2/3 checkpoint and is not supported by the deployment "
            f"code. Pass the {kind} checkpoint produced by the Partition 2/3 training pipeline.")
    if not path.is_file():
        raise CheckpointNotFoundError(
            f"{kind.capitalize()} checkpoint not found: {path}\n"
            f"  Pass the checkpoint path on the command line (--checkpoint, or --{kind}-checkpoint for the "
            f"compare/benchmark tools) or set the {ENV_STUDENT_CHECKPOINT if kind == 'student' else ENV_TEACHER_CHECKPOINT} "
            f"environment variable. Checkpoints are local artifacts and are not stored in Git.")
    return path


def load_model(path: Path, kind: str, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Load a student/teacher checkpoint with the Partition 2/3 loaders. Returns ``(model.eval(), ckpt)``."""
    if kind not in MODEL_KINDS:
        raise ValueError(f"kind must be one of {MODEL_KINDS}, got {kind!r}")
    path = check_checkpoint_path(path, kind)
    if kind == "student":
        from student_model.checkpoint import load_student as loader
    else:
        from teacher_model.checkpoint import load_teacher as loader
    try:
        return loader(path, device)
    except DeploymentError:
        raise
    except (ValueError, KeyError, RuntimeError, OSError, EOFError) as exc:
        raise CheckpointError(f"Could not load {kind} checkpoint {path}: {exc}") from exc
    except Exception as exc:  # corrupt pickle/zip archives raise assorted types
        raise CheckpointError(f"Could not load {kind} checkpoint {path}: {exc!r}") from exc


def checkpoint_summary(ckpt: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    """Metadata actually stored in the checkpoint (nothing is inferred or invented)."""
    cfg = ckpt.get("config", {}) or {}
    summary = {
        "architecture": ckpt.get("architecture"),
        "num_classes": ckpt.get("num_classes"),
        "input_size": ckpt.get("input_size"),
        "epoch": ckpt.get("epoch"),
        "best_metric_name": ckpt.get("best_metric_name"),
        "best_val_metric": ckpt.get("best_val_metric"),
        "pretrained_flag_in_config": cfg.get("pretrained"),
        "distillation": ckpt.get("distillation"),
    }
    if path is not None:
        summary["file"] = str(path)
        summary["file_megabytes"] = round(Path(path).stat().st_size / 1e6, 3)
    return summary


class Predictor:
    """Classifies PIL images / RGB arrays with one eval-mode model.

    The constructor takes an already-built model so tests can inject lightweight fakes;
    use :meth:`from_checkpoint` for real checkpoints.
    """

    def __init__(self, model: nn.Module, class_names: Sequence[str], input_size: int = 224,
                 device: str | torch.device = "cpu", checkpoint: dict[str, Any] | None = None,
                 checkpoint_path: Path | None = None) -> None:
        if not class_names:
            raise ValueError("class_names must not be empty")
        self.model = model.to(device).eval()
        self.class_names = list(class_names)
        self.input_size = int(input_size)
        self.device = torch.device(device)
        self.checkpoint = checkpoint
        self.checkpoint_path = checkpoint_path
        self.transform = EvalTransform(self.input_size)   # identical to validation/test preprocessing

    @classmethod
    def from_checkpoint(cls, path: Path | str, kind: str = "student", device: str = "auto") -> "Predictor":
        dev = resolve_device(device)
        model, ckpt = load_model(Path(path), kind, dev)
        class_names = list(ckpt["class_names"])
        norm = ckpt.get("normalization") or {}
        if (tuple(norm.get("mean", IMAGENET_MEAN)) != tuple(IMAGENET_MEAN)
                or tuple(norm.get("std", IMAGENET_STD)) != tuple(IMAGENET_STD)):
            raise CheckpointError("Checkpoint normalization differs from EvalTransform's ImageNet statistics; "
                                  "refusing to run with mismatched preprocessing.")
        return cls(model, class_names, int(ckpt.get("input_size", 224)), dev, ckpt, Path(path))

    # ------------------------------------------------------------------ inference
    def _to_batch(self, images: Sequence[Image.Image]) -> torch.Tensor:
        return torch.stack([self.transform(img) for img in images]).to(self.device)

    @torch.inference_mode()
    def predict_images(self, images: Sequence[Image.Image], top_k: int = 3) -> list[Prediction]:
        if not images:
            return []
        logits = self.model(self._to_batch(images))
        if logits.ndim != 2 or logits.shape[1] != len(self.class_names):
            raise CheckpointError(f"Model output shape {tuple(logits.shape)} does not match "
                                  f"{len(self.class_names)} classes.")
        probs = torch.softmax(logits.float(), dim=1).cpu()
        k = max(1, min(int(top_k), len(self.class_names)))
        out = []
        for row in probs:
            values, indices = row.topk(k)
            idx = int(indices[0])
            out.append(Prediction(
                label=self.class_names[idx], index=idx, confidence=float(values[0]),
                top_k=tuple((self.class_names[int(i)], float(v)) for v, i in zip(values, indices)),
                probabilities=tuple(float(p) for p in row)))
        return out

    def predict_image(self, image: Image.Image, top_k: int = 3) -> Prediction:
        return self.predict_images([image], top_k)[0]

    def predict_rgb_array(self, rgb: np.ndarray, top_k: int = 3) -> Prediction:
        """``rgb``: uint8 HxWx3 array in RGB channel order (convert OpenCV BGR frames first)."""
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] < 1 or rgb.shape[1] < 1:
            raise InvalidImageError(f"Expected an HxWx3 RGB array, got shape {rgb.shape}.")
        return self.predict_image(Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)), top_k)

    def predict_path(self, path: Path | str, top_k: int = 3) -> Prediction:
        return self.predict_image(open_image(path), top_k)


def open_image(path: Path | str) -> Image.Image:
    """Open an image file as RGB, converting every failure into :class:`InvalidImageError`."""
    path = Path(path)
    if not path.is_file():
        raise InvalidImageError(f"Image file not found: {path}")
    try:
        with Image.open(path) as img:
            img.load()
            if img.width < 1 or img.height < 1:
                raise InvalidImageError(f"Image has no pixels: {path}")
            return img.convert("RGB")
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(f"Not a readable image: {path} ({exc})") from exc
