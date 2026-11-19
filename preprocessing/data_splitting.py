"""Reproducible, stratified, group-aware train/validation/test splitting."""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import SPLIT_NAMES, PreprocessConfig
from .records import STATUS_VALID, ImageRecord, read_csv, write_csv

LOG = logging.getLogger(__name__)
MANIFEST_FIELDS = ["image_path", "class_name", "class_index", "sha256", "dhash",
                   "group_id", "group_size", "split", "width", "height"]


def _assign_class_groups(groups: list[tuple[str, int]], ratios: tuple[float, ...],
                         rng: random.Random) -> dict[str, int]:
    """Greedy largest-first assignment of whole groups to the split with biggest deficit."""
    items = sorted(groups)
    rng.shuffle(items)
    items.sort(key=lambda g: -g[1])  # stable: ties keep the seeded shuffle order
    total = sum(s for _, s in items)
    targets = [r * total for r in ratios]
    current = [0.0] * len(ratios)
    assign: dict[str, int] = {}
    sizes = dict(items)
    for gid, size in items:
        k = max(range(len(ratios)), key=lambda i: (targets[i] - current[i], -i))
        assign[gid] = k
        current[k] += size
    for k in range(len(ratios)):  # repair: never leave a split empty if avoidable
        if not any(v == k for v in assign.values()):
            counts = [sum(1 for v in assign.values() if v == s) for s in range(len(ratios))]
            donor = max(range(len(ratios)), key=lambda s: counts[s])
            if counts[donor] >= 2:
                gid = min((g for g, v in assign.items() if v == donor), key=lambda g: (sizes[g], g))
                assign[gid] = k
    return assign


def build_splits(cfg: PreprocessConfig, records: list[ImageRecord]) -> dict[str, Any]:
    """Split valid images of the selected classes. Returns rows, issues and counts."""
    class_to_idx = cfg.class_to_idx
    valid = [r for r in records if r.status == STATUS_VALID and r.class_name in class_to_idx]
    by_class: dict[str, dict[str, list[ImageRecord]]] = defaultdict(lambda: defaultdict(list))
    for r in valid:
        by_class[r.class_name][r.group_id].append(r)

    rows: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLIT_NAMES}
    issues: list[dict[str, Any]] = []
    per_class: dict[str, Any] = {}
    for cls in cfg.classes:
        groups = by_class.get(cls, {})
        if not groups:
            issues.append({"severity": "critical", "class_name": cls, "code": "class_has_no_images",
                           "detail": "No valid images for this class."})
            per_class[cls] = {s: 0 for s in SPLIT_NAMES}
            continue
        rng = random.Random(f"{cfg.seed}:split:{cls}")
        assign = _assign_class_groups([(g, len(m)) for g, m in groups.items()], cfg.split_ratios, rng)
        counts = [0, 0, 0]
        for gid, k in assign.items():
            for r in groups[gid]:
                counts[k] += 1
                rows[SPLIT_NAMES[k]].append({
                    "image_path": r.rel_path, "class_name": cls, "class_index": class_to_idx[cls],
                    "sha256": r.sha256, "dhash": r.dhash, "group_id": r.group_id,
                    "group_size": r.group_size, "split": SPLIT_NAMES[k],
                    "width": r.width, "height": r.height})
        per_class[cls] = dict(zip(SPLIT_NAMES, counts))
        total = sum(counts)
        for k, name in enumerate(SPLIT_NAMES):
            if counts[k] == 0:
                code = "insufficient_independent_groups" if len(groups) < 3 else "empty_split"
                issues.append({"severity": "critical", "class_name": cls, "code": code,
                               "detail": f"{len(groups)} independent group(s); '{name}' got 0 images."})
            elif abs(counts[k] / total - cfg.split_ratios[k]) > cfg.split_tolerance:
                issues.append({"severity": "warning", "class_name": cls, "code": "ratio_deviation",
                               "detail": f"'{name}' fraction {counts[k] / total:.3f} vs target "
                                         f"{cfg.split_ratios[k]:.2f} (large duplicate groups)."})
    for s in SPLIT_NAMES:
        rows[s].sort(key=lambda r: r["image_path"])
    sizes = {s: len(rows[s]) for s in SPLIT_NAMES}
    LOG.info("Splits: %s; %d issue(s)", sizes, len(issues))
    return {"rows": rows, "issues": issues, "per_class": per_class, "sizes": sizes,
            "seed": cfg.seed, "ratios": list(cfg.split_ratios),
            "method": "per-class, group-aware greedy assignment (seeded), stratified by class"}


def write_manifests(cfg: PreprocessConfig, rows: dict[str, list[dict[str, Any]]]) -> list[Path]:
    return [write_csv(cfg.manifest_path(s), MANIFEST_FIELDS, rows[s]) for s in SPLIT_NAMES]


def read_manifest(path: Path) -> list[dict[str, Any]]:
    """Read a manifest and restore numeric column types."""
    out = []
    for row in read_csv(path):
        row["class_index"] = int(row["class_index"])
        row["group_size"] = int(row["group_size"])
        for key in ("width", "height"):
            row[key] = int(row[key]) if row[key] else None
        out.append(row)
    return out


def resolve_image_path(class_root: Path, row: dict[str, Any]) -> Path:
    """Absolute path of a manifest row (``image_path`` is relative to the class root)."""
    return Path(class_root) / row["image_path"]
