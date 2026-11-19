"""Exact (SHA-256) and near-duplicate (64-bit dHash) detection.

Nothing is deleted. Every valid image receives a ``group_id``; images that are
exact or near duplicates share one, so splitting can keep them together.
Near-duplicates are compared *within a class* using single-linkage grouping.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import PreprocessConfig
from .records import (STATUS_EXCLUDED, STATUS_VALID, ImageRecord, sha256_file,
                      write_csv, write_json)

LOG = logging.getLogger(__name__)
R_LABEL_CONFLICT = "label_conflict_exact_duplicate"
_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash_image(img: Image.Image) -> str:
    """64-bit difference hash as 16 hex chars (robust to re-encoding/brightness)."""
    small = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    arr = np.asarray(small, dtype=np.int16)
    bits = (arr[:, 1:] > arr[:, :-1]).flatten()
    return np.packbits(bits).tobytes().hex()


def _popcount64(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    return _POP8[np.ascontiguousarray(x).view(np.uint8).reshape(*x.shape, 8)].sum(-1)


def _hash_one(args: tuple[Path, ImageRecord]) -> None:
    root, rec = args
    path = root / rec.rel_path
    rec.sha256 = sha256_file(path)
    with Image.open(path) as img:
        rec.dhash = dhash_image(img)


def compute_hashes(cfg: PreprocessConfig, class_root: Path, records: list[ImageRecord]) -> None:
    """Fill ``sha256`` and ``dhash`` for all valid records."""
    valid = [r for r in records if r.status == STATUS_VALID]
    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        list(pool.map(_hash_one, ((Path(class_root), r) for r in valid)))


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def near_pairs(hashes: np.ndarray, threshold: int):
    """Yield index pairs (i<j) with Hamming distance <= threshold."""
    n = len(hashes)
    chunk = max(1, 4_000_000 // max(n, 1))
    for start in range(0, n, chunk):
        block = hashes[start:start + chunk]
        dist = _popcount64(block[:, None] ^ hashes[None, :])
        rows, cols = np.nonzero(dist <= threshold)
        for r, c in zip(rows.tolist(), cols.tolist()):
            if c > start + r:
                yield start + r, c


def assign_duplicate_groups(cfg: PreprocessConfig, records: list[ImageRecord]) -> dict[str, Any]:
    """Resolve label conflicts, build groups, set group ids, return the report."""
    # Exact duplicates carrying different labels cannot be split safely -> exclude + report.
    classes_by_sha: dict[str, set[str]] = defaultdict(set)
    for r in records:
        if r.status == STATUS_VALID:
            classes_by_sha[r.sha256].add(r.class_name or "")
    conflicts = []
    for r in records:
        if r.status == STATUS_VALID and len(classes_by_sha[r.sha256]) > 1:
            r.status, r.reason = STATUS_EXCLUDED, R_LABEL_CONFLICT
            conflicts.append({"image_path": r.rel_path, "class_name": r.class_name,
                              "sha256": r.sha256,
                              "classes_sharing_hash": sorted(classes_by_sha[r.sha256])})

    valid = [r for r in records if r.status == STATUS_VALID]  # already sorted by path
    uf = _UnionFind(len(valid))
    index_by_sha: dict[str, int] = {}
    by_class: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(valid):
        if r.sha256 in index_by_sha:
            uf.union(index_by_sha[r.sha256], i)
        else:
            index_by_sha[r.sha256] = i
        by_class[r.class_name or ""].append(i)

    if cfg.near_duplicate_detection:
        for idxs in by_class.values():
            hv = np.array([int(valid[i].dhash, 16) for i in idxs], dtype=np.uint64)
            for a, b in near_pairs(hv, cfg.near_duplicate_hamming):
                uf.union(idxs[a], idxs[b])

    members: dict[int, list[int]] = defaultdict(list)
    for i in range(len(valid)):
        members[uf.find(i)].append(i)
    ordered = sorted(members.values(), key=lambda m: m[0])
    groups_out = []
    for n, m in enumerate(ordered, start=1):
        gid = f"g{n:06d}"
        for i in m:
            valid[i].group_id, valid[i].group_size = gid, len(m)
        if len(m) > 1:
            shas = {valid[i].sha256 for i in m}
            kind = "exact" if len(shas) == 1 else ("near" if len(shas) == len(m) else "mixed")
            groups_out.append({"group_id": gid, "type": kind, "size": len(m),
                               "class_name": valid[m[0]].class_name,
                               "members": [valid[i].rel_path for i in m]})

    sizes = Counter(len(m) for m in ordered)
    per_class = {}
    for cls, idxs in sorted(by_class.items()):
        gids = Counter(valid[i].group_id for i in idxs)
        per_class[cls] = {"images": len(idxs), "groups": len(gids),
                          "images_in_duplicate_groups": sum(c for c in gids.values() if c > 1),
                          "largest_group": max(gids.values()) if gids else 0}
    largest = sorted(groups_out, key=lambda g: -g["size"])[:10]
    report = {
        "images_considered": len(valid),
        "total_groups": len(ordered),
        "duplicate_groups": len(groups_out),
        "images_in_duplicate_groups": sum(g["size"] for g in groups_out),
        "exact_duplicate_extra_copies": len(valid) - len(index_by_sha),
        "group_size_histogram": {str(k): v for k, v in sorted(sizes.items())},
        "near_duplicate_detection": cfg.near_duplicate_detection,
        "near_duplicate_hamming_threshold": cfg.near_duplicate_hamming,
        "hash": "sha256 (exact) + 64-bit dHash 8x8 (near), compared within class",
        "label_conflicts": conflicts,
        "largest_groups": [{k: g[k] for k in ("group_id", "type", "size", "class_name")}
                           for g in largest],
        "per_class": per_class,
        "groups": groups_out,
        "note": "No image was deleted; groups are kept together by the splitter.",
    }
    LOG.info("Duplicates: %d groups of size>1 covering %d of %d images; %d label conflicts",
             len(groups_out), report["images_in_duplicate_groups"], len(valid), len(conflicts))
    return report


def write_duplicate_reports(cfg: PreprocessConfig, report: dict[str, Any]) -> None:
    write_json(cfg.reports_dir / "duplicate_report.json", report)
    rows = [{"group_id": g["group_id"], "type": g["type"], "class_name": g["class_name"],
             "group_size": g["size"], "image_path": m}
            for g in report["groups"] for m in g["members"]]
    write_csv(cfg.reports_dir / "duplicate_groups.csv",
              ["group_id", "type", "class_name", "group_size", "image_path"], rows)
