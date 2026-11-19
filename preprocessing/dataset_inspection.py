"""Dataset discovery and inspection (read-only on the raw dataset)."""

from __future__ import annotations

import logging
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import ALL_CLASSES, PreprocessConfig
from .labels import canonical_class_name
from .records import ImageRecord

LOG = logging.getLogger(__name__)
_SKIP_DIRS = {"__MACOSX"}


def discover_class_root(raw_dir: Path, min_matches: int = 3, max_depth: int = 4) -> Path | None:
    """Find the directory whose sub-folders are the class folders.

    The Kaggle archive nests data as ``asl_alphabet_train/asl_alphabet_train/<class>``;
    this search finds the level that actually holds ``A``, ``B`` ... so a nested
    layout can never silently yield zero images.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        return None
    best: tuple[tuple[int, int], Path] | None = None
    stack: list[tuple[Path, int]] = [(raw_dir, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            children = sorted(p for p in current.iterdir() if p.is_dir())
        except OSError:
            continue
        matches = sum(1 for c in children if canonical_class_name(c.name) is not None)
        if matches >= min_matches:
            key = (matches, -depth)
            if best is None or key > best[0]:
                best = (key, current)
        if depth < max_depth:
            for child in children:
                if child.name in _SKIP_DIRS or child.name.startswith("."):
                    continue
                stack.append((child, depth + 1))
    return best[1] if best else None


def find_kaggle_test_info(raw_dir: Path) -> dict[str, Any]:
    """Report (informationally) the small Kaggle test folder, if present."""
    info: dict[str, Any] = {"found": False, "path": None, "file_count": 0}
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        return info
    for dirpath, dirnames, filenames in os.walk(raw_dir):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        tests = [f for f in filenames if "_test." in f.lower()]
        if len(tests) >= 3:
            info.update(found=True, path=str(Path(dirpath)), file_count=len(tests))
            return info
    return info


def enumerate_records(class_root: Path) -> tuple[list[ImageRecord], dict[str, Any]]:
    """List every file inside class folders (sorted, deterministic).

    Returns the records plus structural facts (stray files, nested directories).
    """
    class_root = Path(class_root)
    records: list[ImageRecord] = []
    nested_dirs: list[str] = []
    stray_files: list[str] = []
    for entry in sorted(class_root.iterdir(), key=lambda p: p.name):
        if entry.is_file():
            stray_files.append(entry.name)
            continue
        if not entry.is_dir() or entry.name in _SKIP_DIRS:
            continue
        class_name = canonical_class_name(entry.name)
        for dirpath, dirnames, filenames in os.walk(entry):
            dirnames.sort()
            for d in dirnames:
                nested_dirs.append((Path(dirpath) / d).relative_to(class_root).as_posix())
            for fname in sorted(filenames):
                fpath = Path(dirpath) / fname
                try:
                    size = fpath.stat().st_size
                except OSError:
                    size = -1
                records.append(
                    ImageRecord(
                        rel_path=fpath.relative_to(class_root).as_posix(),
                        class_folder=entry.name,
                        class_name=class_name,
                        ext=fpath.suffix.lower(),
                        size_bytes=size,
                    )
                )
    records.sort(key=lambda r: r.rel_path)
    return records, {"nested_directories": nested_dirs, "stray_files_in_root": stray_files}


def _read_header(args: tuple[Path, ImageRecord]) -> None:
    root, rec = args
    try:
        with Image.open(root / rec.rel_path) as img:
            rec.width, rec.height = img.size
        rec.header_ok = True
    except Exception:  # noqa: BLE001 - any failure means unreadable header
        rec.header_ok = False


def inspect_dataset(
    cfg: PreprocessConfig, class_root: Path, records: list[ImageRecord], structure: dict[str, Any]
) -> dict[str, Any]:
    """Compute dataset statistics from the real files; fills width/height on records."""
    class_root = Path(class_root)
    supported = [r for r in records if r.ext in cfg.image_extensions]
    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        list(pool.map(_read_header, ((class_root, r) for r in supported)))

    folders = sorted({r.class_folder for r in records} | {
        p.name for p in class_root.iterdir() if p.is_dir() and p.name not in _SKIP_DIRS
    })
    selected = set(cfg.classes)
    class_folders = []
    for folder in folders:
        canon = canonical_class_name(folder)
        in_folder = [r for r in records if r.class_folder == folder]
        class_folders.append({
            "folder": folder,
            "canonical_class": canon,
            "file_count": len(in_folder),
            "supported_image_count": sum(1 for r in in_folder if r.ext in cfg.image_extensions),
            "selected": canon in selected if canon else False,
        })

    present = {c["canonical_class"] for c in class_folders if c["canonical_class"]}
    unexpected = [c["folder"] for c in class_folders if c["canonical_class"] is None]
    missing_selected = [c for c in cfg.classes if c not in present]
    missing_known = [c for c in ALL_CLASSES if c not in present]

    ext_counts = Counter(r.ext or "<none>" for r in records)
    per_class_all: dict[str, int] = {}
    for c in class_folders:
        if c["canonical_class"]:
            per_class_all[c["canonical_class"]] = (
                per_class_all.get(c["canonical_class"], 0) + c["supported_image_count"]
            )

    excluded_classes = {}
    for name, reason in cfg.excluded_classes.items():
        excluded_classes[name] = {
            "present_in_dataset": name in present,
            "supported_image_count": per_class_all.get(name, 0),
            "exclusion_reason": reason,
        }

    ok = [r for r in supported if r.header_ok]
    widths = np.array([r.width for r in ok], dtype=np.int64)
    heights = np.array([r.height for r in ok], dtype=np.int64)
    size_counter = Counter((int(w), int(h)) for w, h in zip(widths, heights))
    dims = {
        "images_measured": int(len(ok)),
        "header_unreadable": int(len(supported) - len(ok)),
        "width": _stats(widths),
        "height": _stats(heights),
        "unique_sizes": len(size_counter),
        "most_common_sizes": [
            {"width": w, "height": h, "count": n} for (w, h), n in size_counter.most_common(10)
        ],
    }

    rng = random.Random(f"{cfg.seed}:samples")
    samples: dict[str, list[str]] = {}
    by_class: dict[str, list[ImageRecord]] = {}
    for r in ok:
        if r.class_name:
            by_class.setdefault(r.class_name, []).append(r)
    for name in sorted(by_class):
        pool = by_class[name]
        samples[name] = [r.rel_path for r in rng.sample(pool, min(cfg.samples_per_class, len(pool)))]

    summary = {
        "class_root": str(class_root),
        "total_files": len(records),
        "supported_image_files": len(supported),
        "extension_counts": dict(sorted(ext_counts.items())),
        "class_folders": class_folders,
        "unexpected_folders": unexpected,
        "missing_selected_classes": missing_selected,
        "missing_known_classes": missing_known,
        "per_class_supported_image_counts": dict(sorted(per_class_all.items())),
        "excluded_classes": excluded_classes,
        "dimensions": dims,
        "samples": samples,
        **structure,
        "kaggle_test_folder": find_kaggle_test_info(cfg.raw_dir),
    }
    LOG.info(
        "Inspection: %d files, %d supported images, %d class folders (%d unexpected)",
        len(records), len(supported), len(class_folders), len(unexpected),
    )
    return summary


def _stats(arr: np.ndarray) -> dict[str, float | None]:
    if arr.size == 0:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": int(arr.min()), "max": int(arr.max()),
        "mean": round(float(arr.mean()), 3), "median": float(np.median(arr)),
    }


def compute_raw_fingerprint(class_root: Path) -> str:
    """Fingerprint of (relative path, size, mtime) for every file under the class root.

    Used to prove the raw dataset was not modified during a run.
    """
    import hashlib

    digest = hashlib.sha256()
    class_root = Path(class_root)
    lines = []
    for dirpath, dirnames, filenames in os.walk(class_root):
        dirnames.sort()
        for fname in sorted(filenames):
            p = Path(dirpath) / fname
            st = p.stat()
            lines.append(f"{p.relative_to(class_root).as_posix()}|{st.st_size}|{st.st_mtime_ns}")
    for line in sorted(lines):
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
