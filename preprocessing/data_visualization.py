"""Charts generated strictly from inspected data (saved under reports/)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless; safe on servers and Windows without a display
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from .config import SPLIT_NAMES, PreprocessConfig  # noqa: E402

LOG = logging.getLogger(__name__)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_class_distribution(cfg: PreprocessConfig, summary: dict[str, Any]) -> Path:
    counts = summary["per_class_supported_image_counts"]
    names = sorted(counts, key=lambda n: (len(n) > 1, n))
    colors = ["#2b7bba" if n in cfg.classes else "#bbbbbb" for n in names]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(names, [counts[n] for n in names], color=colors)
    ax.set_title("Supported image files per class folder (grey = excluded from current class set)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Images")
    return _save(fig, cfg.reports_dir / "class_distribution.png")


def plot_sample_images(cfg: PreprocessConfig, class_root: Path, summary: dict[str, Any]) -> Path:
    samples = {c: p for c, p in summary["samples"].items() if c in cfg.classes}
    k = max((len(v) for v in samples.values()), default=1)
    per_row = 4
    n_rows = max(1, -(-len(samples) // per_row))
    fig, axes = plt.subplots(n_rows, per_row * k, figsize=(per_row * k * 1.6, n_rows * 1.8),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ci, (cls, paths) in enumerate(sorted(samples.items())):
        r, c0 = divmod(ci, per_row)
        for j, rel in enumerate(paths):
            ax = axes[r][c0 * k + j]
            with Image.open(Path(class_root) / rel) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_title(cls if j == 0 else "", fontsize=9)
    return _save(fig, cfg.reports_dir / "sample_images.png")


def plot_image_dimensions(cfg: PreprocessConfig, records) -> Path:
    ws = [r.width for r in records if r.header_ok]
    hs = [r.height for r in records if r.header_ok]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].hist(ws, bins=30, color="#2b7bba")
    axes[0].set_title("Width (px)")
    axes[1].hist(hs, bins=30, color="#d95f02")
    axes[1].set_title("Height (px)")
    axes[2].scatter(ws, hs, s=6, alpha=0.4)
    axes[2].set_title("Width vs height")
    axes[2].set_xlabel("width")
    axes[2].set_ylabel("height")
    return _save(fig, cfg.reports_dir / "image_dimensions.png")


def plot_dataset_summary(cfg: PreprocessConfig, cleaning: dict[str, Any],
                         duplicates: dict[str, Any]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(["valid", "excluded", "not selected"],
                [cleaning["valid"], cleaning["excluded"], cleaning["not_selected"]],
                color=["#1b9e77", "#d95f02", "#999999"])
    axes[0].set_title("File status")
    hist = duplicates["group_size_histogram"]
    axes[1].bar(list(hist.keys()), list(hist.values()), color="#7570b3")
    axes[1].set_title("Duplicate-group size (images per group)")
    axes[1].set_yscale("log")
    reasons = cleaning["excluded_by_reason"] or {"none": 0}
    axes[2].bar(list(reasons.keys()), list(reasons.values()), color="#d95f02")
    axes[2].set_title("Exclusion reasons")
    axes[2].tick_params(axis="x", rotation=20)
    return _save(fig, cfg.reports_dir / "dataset_summary.png")


def plot_split_distribution(cfg: PreprocessConfig, per_class: dict[str, dict[str, int]]) -> Path:
    names = [c for c in cfg.classes if c in per_class]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    bottom = [0] * len(names)
    for split, color in zip(SPLIT_NAMES, ("#1b9e77", "#e6ab02", "#7570b3")):
        vals = [per_class[n][split] for n in names]
        ax.bar(names, vals, bottom=bottom, label=split, color=color)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_title("Images per class by split")
    ax.legend()
    return _save(fig, cfg.reports_dir / "split_distribution.png")
