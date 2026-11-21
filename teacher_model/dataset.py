"""Lazy PyTorch dataset over the preprocessing manifests (train / validation / test).

The split assignment comes ONLY from the manifests written by ``python -m preprocessing.main``.
This module never re-splits, never reads ``external_test.csv`` and never loads all images into RAM.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

from preprocessing.config import SPLIT_NAMES

from .augmentation import EvalTransform, TrainTransform
from .config import TeacherConfig
from .utils import read_json, seed_worker

LOG = logging.getLogger(__name__)
REQUIRED_COLUMNS = ("image_path", "class_name", "class_index", "split")


class DatasetError(RuntimeError):
    """Raised for missing/invalid manifests, labels or images."""


def load_label_mapping(labels_path: Path) -> dict[str, Any]:
    """Read ``labels.json`` (authoritative class mapping) and sanity-check it."""
    labels_path = Path(labels_path)
    if not labels_path.is_file():
        raise DatasetError(f"labels.json not found: {labels_path}. Run `python -m preprocessing.main "
                           "--class-set 29 --force` first.")
    labels = read_json(labels_path)
    classes, c2i = labels.get("classes"), labels.get("class_to_idx")
    if not classes or not c2i or int(labels.get("num_classes", -1)) != len(classes):
        raise DatasetError(f"{labels_path} is malformed (classes/class_to_idx/num_classes).")
    if any(c2i.get(name) != i for i, name in enumerate(classes)):
        raise DatasetError(f"{labels_path}: class_to_idx is inconsistent with the class list.")
    return labels


def read_manifest(path: Path, split: str, class_to_idx: dict[str, int]) -> list[dict[str, str]]:
    """Read and validate one manifest CSV."""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"Manifest not found: {path}. Run the preprocessing pipeline first.")
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise DatasetError(f"Manifest is empty: {path}")
    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise DatasetError(f"{path} lacks required columns {missing}; found {list(rows[0])}")
    for n, row in enumerate(rows, start=2):
        name = row["class_name"]
        if name not in class_to_idx:
            raise DatasetError(f"{path}:{n} unknown class '{name}' (not in labels.json)")
        if int(row["class_index"]) != class_to_idx[name]:
            raise DatasetError(f"{path}:{n} class_index {row['class_index']} != labels.json index "
                               f"{class_to_idx[name]} for '{name}'")
        if row["split"] != split:
            raise DatasetError(f"{path}:{n} row says split='{row['split']}' but file is the '{split}' manifest")
    return rows


def resolve_class_root(cfg: TeacherConfig) -> Path:
    """Dataset class root: explicit override, else ``class_root`` from dataset_summary.json."""
    if cfg.dataset_root:
        root = Path(cfg.dataset_root)
    else:
        if not cfg.summary_file.is_file():
            raise DatasetError(f"{cfg.summary_file} not found and --dataset-root not given.")
        recorded = read_json(cfg.summary_file).get("class_root")
        if not recorded:
            raise DatasetError(f"'class_root' missing in {cfg.summary_file}; pass --dataset-root.")
        root = Path(recorded)
    if not root.is_dir():
        raise DatasetError(f"Dataset class root does not exist: {root}. It is the folder containing the "
                           "A, B, ... class folders; pass --dataset-root <path> if the dataset was moved.")
    return root


def assert_no_split_leakage(manifests: dict[str, list[dict[str, str]]]) -> None:
    """Fail loudly if any image path, SHA-256 or duplicate group appears in two splits."""
    for key in ("image_path", "sha256", "group_id"):
        owner: dict[str, str] = {}
        for split, rows in manifests.items():
            for row in rows:
                value = row.get(key, "")
                if not value:
                    continue
                if owner.setdefault(value, split) != split:
                    raise DatasetError(f"Split leakage: {key}='{value}' is in both "
                                       f"'{owner[value]}' and '{split}'.")


class ASLManifestDataset(Dataset):
    """Lazy dataset: reads one image from disk per ``__getitem__``."""

    def __init__(self, rows: list[dict[str, str]], class_root: Path, transform: Callable,
                 split: str, check_files: bool = True) -> None:
        self.rows = rows
        self.class_root = Path(class_root)
        self.transform = transform
        self.split = split
        if check_files:
            absent = [r["image_path"] for r in rows if not (self.class_root / r["image_path"]).is_file()]
            if absent:
                raise DatasetError(f"{len(absent)} image(s) of the '{split}' split are missing under "
                                   f"{self.class_root}; first: {absent[:3]}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        path = self.class_root / row["image_path"]
        try:
            with Image.open(path) as img:
                img.load()
                tensor = self.transform(img)
        except FileNotFoundError as exc:
            raise DatasetError(f"Image file missing: {path}") from exc
        except (UnidentifiedImageError, OSError) as exc:
            raise DatasetError(f"Unreadable image: {path} ({exc})") from exc
        return tensor, int(row["class_index"])

    @property
    def targets(self) -> list[int]:
        return [int(r["class_index"]) for r in self.rows]


def build_datasets(cfg: TeacherConfig, check_files: bool = True
                   ) -> tuple[dict[str, ASLManifestDataset], dict[str, Any]]:
    """Create train/validation/test datasets. Only the train dataset gets random rotation."""
    labels = load_label_mapping(cfg.labels_file)
    c2i = labels["class_to_idx"]
    class_root = resolve_class_root(cfg)
    manifests = {s: read_manifest(cfg.manifests / f"{s}.csv", s, c2i) for s in SPLIT_NAMES}
    assert_no_split_leakage(manifests)
    transforms = {
        "train": TrainTransform(cfg.image_size, cfg.rotation_deg, seed=cfg.seed),
        "validation": EvalTransform(cfg.image_size),
        "test": EvalTransform(cfg.image_size),
    }
    datasets = {s: ASLManifestDataset(manifests[s], class_root, transforms[s], s, check_files)
                for s in SPLIT_NAMES}
    LOG.info("Datasets: %s", {s: len(d) for s, d in datasets.items()})
    return datasets, labels


def build_dataloaders(datasets: dict[str, Dataset], cfg: TeacherConfig, use_cuda: bool
                      ) -> dict[str, DataLoader]:
    """Mini-batch loaders. Shuffle only for training; Windows-safe worker settings."""
    generator = torch.Generator().manual_seed(cfg.seed)
    common: dict[str, Any] = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                                  pin_memory=use_cuda, worker_init_fn=seed_worker)
    if cfg.num_workers > 0:
        common["persistent_workers"] = True
    loaders = {}
    for split, ds in datasets.items():
        loaders[split] = DataLoader(ds, shuffle=(split == "train"), drop_last=False,
                                    generator=generator if split == "train" else None, **common)
    return loaders
