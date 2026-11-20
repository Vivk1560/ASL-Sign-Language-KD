"""Centralised configuration for the preprocessing pipeline.

Every path, threshold, seed and class definition used by the other modules
lives here. Nothing in this module touches the filesystem at import time.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = Path(os.environ.get("ASL_RAW_DIR", PROJECT_ROOT / "data" / "raw"))
DEFAULT_PROCESSED_DIR = Path(
    os.environ.get("ASL_PROCESSED_DIR", PROJECT_ROOT / "data" / "processed")
)

ALPHABET: tuple[str, ...] = tuple(chr(c) for c in range(ord("A"), ord("Z") + 1))
EXTRA_CLASSES: tuple[str, ...] = ("del", "nothing", "space")
ALL_CLASSES: tuple[str, ...] = ALPHABET + EXTRA_CLASSES
CLASS_SETS = ("26", "29")

EXTRA_CLASS_EXCLUSION_REASON = (
    "Excluded from the default 26-class (A-Z) objective; the existing "
    "classifier heads have 26 outputs. Enable class_set='29' to include it."
)

IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
SPLIT_NAMES: tuple[str, str, str] = ("train", "validation", "test")
DHASH_BITS = 8  # 8x8 difference hash -> 64-bit hash (fixed, see duplicate_detection)


@dataclass(frozen=True)
class AugmentationConfig:
    """Training-only augmentation parameters (no horizontal flip by design)."""

    rotation_deg: float = 15.0
    scale_range: tuple[float, float] = (0.90, 1.10)
    translate_frac: float = 0.08
    brightness_range: tuple[float, float] = (0.75, 1.25)
    contrast_range: tuple[float, float] = (0.75, 1.25)
    saturation_range: tuple[float, float] = (0.80, 1.20)
    channel_gain: float = 0.05
    occlusion_prob: float = 0.20
    occlusion_area_frac: tuple[float, float] = (0.02, 0.10)
    occlusion_max_holes: int = 2
    # Offline generation is optional; on-the-fly use is the default approach.
    offline_enabled: bool = False
    offline_copies_per_image: int = 2
    offline_jpeg_quality: int = 95


def _default_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


@dataclass(frozen=True)
class PreprocessConfig:
    """All tunable settings of the pipeline."""

    raw_dir: Path = DEFAULT_RAW_DIR
    processed_dir: Path = DEFAULT_PROCESSED_DIR
    kaggle_dataset: str = "grassknoted/asl-alphabet"
    class_set: str = "26"
    seed: int = 42
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    split_tolerance: float = 0.05
    image_extensions: tuple[str, ...] = IMAGE_EXTENSIONS
    image_size: int = 224  # informational for later partitions; raw is never resized
    min_image_dim: int = 16
    max_image_dim: int = 8192
    min_class_folders_to_detect: int = 3
    near_duplicate_detection: bool = True
    near_duplicate_hamming: int = 4
    max_group_fraction: float = 0.10  # validation fails if one group holds > this share of a class
    samples_per_class: int = 3
    workers: int = field(default_factory=_default_workers)
    decode_check_in_validation: bool = True
    generate_visualizations: bool = True
    # If True, a missing official external test folder is a critical failure (isolation cannot be
    # verified). If False it is reported as an explicit NOT-VERIFIED warning. Does not change outputs.
    require_external_test: bool = False
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    # ------------------------------------------------------------------ paths
    @property
    def manifests_dir(self) -> Path:
        return Path(self.processed_dir) / "manifests"

    @property
    def metadata_dir(self) -> Path:
        return Path(self.processed_dir) / "metadata"

    @property
    def reports_dir(self) -> Path:
        return Path(self.processed_dir) / "reports"

    @property
    def logs_dir(self) -> Path:
        return self.reports_dir / "logs"

    @property
    def augmented_dir(self) -> Path:
        return Path(self.processed_dir) / "augmented"

    @property
    def archive_dir(self) -> Path:
        return Path(self.processed_dir) / "_archive"

    @property
    def labels_path(self) -> Path:
        return self.metadata_dir / "labels.json"

    @property
    def summary_path(self) -> Path:
        return self.metadata_dir / "dataset_summary.json"

    @property
    def config_path(self) -> Path:
        return self.metadata_dir / "preprocessing_config.json"

    @property
    def run_state_path(self) -> Path:
        return self.metadata_dir / "run_state.json"

    def manifest_path(self, split: str) -> Path:
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split '{split}'")
        return self.manifests_dir / f"{split}.csv"

    @property
    def augmented_manifest_path(self) -> Path:
        return self.manifests_dir / "train_augmented.csv"

    # ---------------------------------------------------------------- classes
    @property
    def classes(self) -> tuple[str, ...]:
        return ALPHABET if self.class_set == "26" else ALL_CLASSES

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.classes)}

    @property
    def excluded_classes(self) -> dict[str, str]:
        if self.class_set == "26":
            return {name: EXTRA_CLASS_EXCLUSION_REASON for name in EXTRA_CLASSES}
        return {}

    # ------------------------------------------------------------- utilities
    def validate(self) -> None:
        """Raise ``ValueError`` if the configuration is inconsistent."""
        if self.class_set not in CLASS_SETS:
            raise ValueError(f"class_set must be one of {CLASS_SETS}, got {self.class_set!r}")
        if len(self.split_ratios) != 3 or any(r <= 0 for r in self.split_ratios):
            raise ValueError("split_ratios must be three positive numbers")
        if abs(sum(self.split_ratios) - 1.0) > 1e-6:
            raise ValueError(f"split_ratios must sum to 1.0, got {sum(self.split_ratios)}")
        if not 0 <= self.near_duplicate_hamming <= 16:
            raise ValueError("near_duplicate_hamming must be in [0, 16]")
        if not 0 < self.max_group_fraction <= 1:
            raise ValueError("max_group_fraction must be in (0, 1]")
        if self.min_image_dim < 1 or self.max_image_dim < self.min_image_dim:
            raise ValueError("invalid image dimension limits")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        raw, proc = Path(self.raw_dir).resolve(), Path(self.processed_dir).resolve()
        if raw == proc or raw in proc.parents or proc in raw.parents:
            raise ValueError(
                "raw_dir and processed_dir must be separate, non-nested directories "
                f"(raw={raw}, processed={proc})"
            )

    def with_overrides(self, **changes: Any) -> "PreprocessConfig":
        """Return a copy with ``changes`` applied (paths are coerced to ``Path``)."""
        for key in ("raw_dir", "processed_dir"):
            if key in changes and changes[key] is not None:
                changes[key] = Path(changes[key]).expanduser().resolve()
        changes = {k: v for k, v in changes.items() if v is not None}
        if "split_ratios" in changes:
            changes["split_ratios"] = tuple(float(x) for x in changes["split_ratios"])
        new = dataclasses.replace(self, **changes)
        new.validate()
        return new

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view (paths as strings)."""
        data = dataclasses.asdict(self)
        data["raw_dir"] = str(self.raw_dir)
        data["processed_dir"] = str(self.processed_dir)
        return data

    def config_hash(self) -> str:
        """Hash of every setting that changes generated outputs (not paths/workers)."""
        data = self.to_dict()
        for volatile in ("raw_dir", "processed_dir", "workers", "generate_visualizations",
                         "decode_check_in_validation", "kaggle_dataset", "require_external_test"):
            data.pop(volatile, None)
        blob = json.dumps(data, sort_keys=True, default=list)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_config(**overrides: Any) -> PreprocessConfig:
    """Build a validated default configuration with optional overrides."""
    cfg = PreprocessConfig()
    cfg = cfg.with_overrides(**overrides) if overrides else cfg
    cfg.validate()
    return cfg
