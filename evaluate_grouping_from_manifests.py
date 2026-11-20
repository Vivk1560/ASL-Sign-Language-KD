"""Hash-level evaluation of duplicate grouping + splitting (no images needed).

Reads sha256/dhash from the committed manifests, re-runs the *current*
``assign_duplicate_groups`` and ``build_splits`` on them and reports group-size,
per-class split balance and residual cross-split near-duplicate pairs.
This is NOT a substitute for ``python -m preprocessing.main`` on the real images.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from preprocessing.config import SPLIT_NAMES, load_config
from preprocessing.data_splitting import build_splits
from preprocessing.duplicate_detection import _popcount64, assign_duplicate_groups
from preprocessing.records import STATUS_VALID, ImageRecord

ROOT = Path(__file__).resolve().parent


def load_records(manifest_dir: Path) -> list[ImageRecord]:
    recs = []
    for s in SPLIT_NAMES:
        with open(manifest_dir / f"{s}.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                recs.append(ImageRecord(
                    rel_path=r["image_path"], class_folder=r["class_name"],
                    class_name=r["class_name"], ext=".jpg", size_bytes=0,
                    width=int(r["width"]), height=int(r["height"]), header_ok=True,
                    status=STATUS_VALID, sha256=r["sha256"], dhash=r["dhash"]))
    recs.sort(key=lambda r: r.rel_path)
    return recs


def residual_cross_split_pairs(rows_by_split: dict, threshold: int) -> int:
    """Count image pairs (same class) with dHash distance <= threshold in different splits."""
    per_class = defaultdict(list)
    for s, rows in rows_by_split.items():
        for r in rows:
            per_class[r["class_name"]].append((int(r["dhash"], 16), s))
    total = 0
    for items in per_class.values():
        h = np.array([x[0] for x in items], dtype=np.uint64)
        sp = np.array([SPLIT_NAMES.index(x[1]) for x in items])
        for a in range(0, len(h), 500):
            d = _popcount64(h[a:a + 500, None] ^ h[None, :])
            close = d <= threshold
            diff = sp[a:a + 500, None] != sp[None, :]
            total += int((close & diff).sum())
    return total // 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", type=Path, default=ROOT / "data/processed/manifests")
    ap.add_argument("--class-set", default="29")
    ap.add_argument("--threshold", type=int, default=4)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    cfg = load_config(class_set=a.class_set, near_duplicate_hamming=a.threshold)
    recs = load_records(a.manifest_dir)
    rep = assign_duplicate_groups(cfg, recs)
    split = build_splits(cfg, recs)
    sizes = [g["size"] for g in rep["groups"]]
    out = {
        "images": rep["images_considered"], "total_groups": rep["total_groups"],
        "multi_image_groups": rep["duplicate_groups"],
        "images_in_multi_groups": rep["images_in_duplicate_groups"],
        "max_group_size": rep["max_group_size"],
        "top10_groups": [g["size"] for g in rep["largest_groups"]],
        "median_multi_group_size": float(np.median(sizes)) if sizes else 0,
        "oversized_groups": rep["oversized_groups"],
        "split_sizes": split["sizes"],
        "split_issues": split["issues"],
        "per_class_splits": split["per_class"],
        "residual_cross_split_near_pairs": residual_cross_split_pairs(split["rows"], a.threshold),
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    print(text)
    if a.out:
        a.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
