"""Configuration for student / Knowledge Distillation training.

Data settings deliberately mirror ``teacher_model.config.TeacherConfig`` so the student reads the same
manifests, image size and normalisation as the teacher; ``to_data_config()`` hands them to the existing
teacher data pipeline instead of duplicating it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from teacher_model.config import (DEFAULT_CHECKPOINT_NAME as TEACHER_CHECKPOINT_NAME,
                                  DEFAULT_OUTPUT_DIR, TeacherConfig)
from preprocessing.config import DEFAULT_PROCESSED_DIR

from .distillation import DEFAULT_ALPHA, DEFAULT_TEMPERATURE, validate_kd_hyperparameters

ARCHITECTURE_ID = "mobilenet_v2"
DEFAULT_TEACHER_CHECKPOINT = DEFAULT_OUTPUT_DIR / TEACHER_CHECKPOINT_NAME
DEFAULT_CHECKPOINT_NAME = "student_mobilenetv2_kd_best.pth"   # distinct from teacher_* and student_best_aug.pth


@dataclass(frozen=True)
class StudentConfig:
    """All tunable settings of the student / KD pipeline."""

    # --- data (produced by `python -m preprocessing.main`) ---
    processed_dir: Path = DEFAULT_PROCESSED_DIR
    manifest_dir: Path | None = None
    labels_path: Path | None = None
    dataset_root: Path | None = None
    image_size: int = 224                 # must equal the teacher checkpoint's input size
    rotation_deg: float = 15.0
    # --- teacher ---
    teacher_checkpoint: Path = DEFAULT_TEACHER_CHECKPOINT
    # --- student model ---
    pretrained: bool = True
    dropout: float = 0.2
    # --- distillation ---
    temperature: float = DEFAULT_TEMPERATURE
    alpha: float = DEFAULT_ALPHA          # weight of the supervised CE term; (1 - alpha) weights KD
    label_smoothing: float = 0.0          # applied to the CE term only
    # --- optimisation ---
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"              # "adamw" | "sgd"
    min_lr: float = 1e-6
    monitor: str = "val_loss"             # "val_loss" (min) | "val_accuracy" (max); CE on validation
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    # --- runtime ---
    num_workers: int = 2
    device: str = "auto"                  # "auto" | "cpu" | "cuda"
    mixed_precision: bool = True          # only effective on CUDA
    seed: int = 42
    # --- outputs ---
    output_dir: Path = DEFAULT_OUTPUT_DIR
    checkpoint_name: str = DEFAULT_CHECKPOINT_NAME
    metrics_dir: Path | None = None
    resume: Path | None = None
    overwrite: bool = False
    max_train_batches: int | None = None  # debugging / smoke runs only

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.output_dir) / self.checkpoint_name

    @property
    def reports_dir(self) -> Path:
        return Path(self.metrics_dir) if self.metrics_dir else Path(self.output_dir) / "student_mobilenetv2_reports"

    def to_data_config(self) -> TeacherConfig:
        """The data-related subset as a ``TeacherConfig`` so the existing P2 dataset code can be reused."""
        return TeacherConfig(
            processed_dir=self.processed_dir, manifest_dir=self.manifest_dir, labels_path=self.labels_path,
            dataset_root=self.dataset_root, image_size=self.image_size, rotation_deg=self.rotation_deg,
            pretrained=False, epochs=self.epochs, batch_size=self.batch_size,
            learning_rate=self.learning_rate, weight_decay=self.weight_decay, optimizer=self.optimizer,
            monitor=self.monitor, num_workers=self.num_workers, device=self.device, seed=self.seed,
            output_dir=self.output_dir)

    def validate(self) -> None:
        validate_kd_hyperparameters(self.temperature, self.alpha)
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.to_data_config().validate()      # image_size, rotation, epochs, batch, lr, optimizer, ...
        if self.monitor not in ("val_loss", "val_accuracy"):
            raise ValueError("monitor must be 'val_loss' or 'val_accuracy'")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in data.items()}
