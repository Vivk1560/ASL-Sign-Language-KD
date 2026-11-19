"""Synthetic Kaggle-shaped fixtures. These are NOT real dataset statistics."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from preprocessing.config import ALL_CLASSES, load_config


def random_image(seed: int, size: int = 64) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    return Image.fromarray(base).resize((size, size), Image.Resampling.BICUBIC)


def build_dataset(root: Path, per_class: int = 14, classes=ALL_CLASSES,
                  nested: bool = True) -> Path:
    """Create <root>/asl_alphabet_train/asl_alphabet_train/<class>/*.jpg (+ test folder)."""
    class_root = root / "asl_alphabet_train" / "asl_alphabet_train" if nested else root
    seed = 0
    for cls in classes:
        d = class_root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            seed += 1
            random_image(seed).save(d / f"{cls}{i + 1}.jpg", quality=95)
    test_dir = root / "asl_alphabet_test" / "asl_alphabet_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    for cls in ("A", "B", "nothing", "space"):
        random_image(99999 + len(cls)).save(test_dir / f"{cls}_test.jpg")
    return class_root


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    """Raw dir containing a synthetic nested dataset."""
    raw = tmp_path / "raw"
    build_dataset(raw)
    return raw


@pytest.fixture()
def cfg(tmp_path: Path, dataset: Path):
    return load_config(raw_dir=dataset, processed_dir=tmp_path / "processed", workers=2)


def class_root_of(raw: Path) -> Path:
    return raw / "asl_alphabet_train" / "asl_alphabet_train"
