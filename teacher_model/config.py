"""Configuration for teacher training. Paths default to the preprocessing pipeline's outputs."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from preprocessing.config import DEFAULT_PROCESSED_DIR, PROJECT_ROOT

ARCHITECTURE_ID = "resnet50"
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "model_weights"
DEFAULT_CHECKPOINT_NAME = "teacher_resnet50_best.pth"  # NOT teacher_best_aug.pth (pre-existing file)


@dataclass(frozen=True)
class TeacherConfig:
    """All tunable settings of the teacher pipeline."""

    # --- data (produced by `python -m preprocessing.main`) ---
    processed_dir: Path = DEFAULT_PROCESSED_DIR
    manifest_dir: Path | None = None      # default: <processed_dir>/manifests
    labels_path: Path | None = None       # default: <processed_dir>/metadata/labels.json
    dataset_root: Path | None = None      # class root; default: read from dataset_summary.json
    image_size: int = 224
    rotation_deg: float = 15.0            # max |angle| of scipy.ndimage.rotate augmentation
    # --- model ---
    pretrained: bool = True
    # --- optimisation ---
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    optimizer: str = "adamw"              # "adamw" | "sgd"
    min_lr: float = 1e-6                  # cosine annealing floor
    monitor: str = "val_loss"             # model-selection metric: "val_loss" (min) | "val_accuracy" (max)
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
    metrics_dir: Path | None = None       # default: <output_dir>/teacher_resnet50_reports
    resume: Path | None = None
    overwrite: bool = False               # allow replacing an existing best checkpoint
    max_train_batches: int | None = None  # debugging / smoke runs only

    # ---------------------------------------------------------------- derived
    @property
    def manifests(self) -> Path:
        return Path(self.manifest_dir) if self.manifest_dir else Path(self.processed_dir) / "manifests"

    @property
    def labels_file(self) -> Path:
        return Path(self.labels_path) if self.labels_path else Path(self.processed_dir) / "metadata" / "labels.json"

    @property
    def summary_file(self) -> Path:
        return Path(self.processed_dir) / "metadata" / "dataset_summary.json"

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.output_dir) / self.checkpoint_name

    @property
    def reports_dir(self) -> Path:
        return Path(self.metrics_dir) if self.metrics_dir else Path(self.output_dir) / "teacher_resnet50_reports"

    def validate(self) -> None:
        if self.image_size < 32:
            raise ValueError("image_size must be >= 32")
        if self.rotation_deg < 0:
            raise ValueError("rotation_deg must be >= 0")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.optimizer not in ("adamw", "sgd"):
            raise ValueError("optimizer must be 'adamw' or 'sgd'")
        if self.monitor not in ("val_loss", "val_accuracy"):
            raise ValueError("monitor must be 'val_loss' or 'val_accuracy'")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be 'auto', 'cpu' or 'cuda'")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in data.items()}
