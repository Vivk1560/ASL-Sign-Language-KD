"""Identify files that must be excluded. Raw files are never deleted or modified."""

from __future__ import annotations

import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from .config import PreprocessConfig
from .labels import canonical_class_name
from .records import (STATUS_EXCLUDED, STATUS_NOT_SELECTED, STATUS_VALID, ImageRecord,
                      write_csv, write_json)

LOG = logging.getLogger(__name__)

R_UNSUPPORTED = "unsupported_file_type"
R_EMPTY_FILE = "empty_file"
R_CORRUPT = "corrupt_or_unreadable"
R_DIMENSIONS = "invalid_dimensions"
R_NOT_SELECTED = "class_not_in_selected_set"
R_UNEXPECTED_FOLDER = "unexpected_class_folder"


def check_image_file(path: Path, cfg: PreprocessConfig) -> tuple[str | None, str]:
    """Return ``(reason, detail)``; ``reason`` is ``None`` for a healthy image."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
            width, height = img.size
            img.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        return R_CORRUPT, f"{type(exc).__name__}: {str(exc)[:160]}"
    lo, hi = cfg.min_image_dim, cfg.max_image_dim
    if not (lo <= width <= hi and lo <= height <= hi):
        return R_DIMENSIONS, f"{width}x{height} outside [{lo}, {hi}]"
    return None, ""


def _classify(args: tuple[PreprocessConfig, Path, ImageRecord, frozenset[str]]) -> None:
    cfg, root, rec, selected = args
    if rec.class_name is None:
        rec.status, rec.reason = STATUS_NOT_SELECTED, R_UNEXPECTED_FOLDER
        return
    if rec.class_name not in selected:
        rec.status, rec.reason = STATUS_NOT_SELECTED, R_NOT_SELECTED
        return
    if rec.ext not in cfg.image_extensions:
        rec.status, rec.reason = STATUS_EXCLUDED, R_UNSUPPORTED
        return
    if rec.size_bytes == 0:
        rec.status, rec.reason = STATUS_EXCLUDED, R_EMPTY_FILE
        return
    reason, detail = check_image_file(root / rec.rel_path, cfg)
    if reason:
        rec.status, rec.reason = STATUS_EXCLUDED, reason
        rec.sha256 = ""  # detail is kept only in the report
        _DETAILS[rec.rel_path] = detail
        return
    rec.status, rec.reason = STATUS_VALID, ""


_DETAILS: dict[str, str] = {}


def clean_records(
    cfg: PreprocessConfig, class_root: Path, records: list[ImageRecord]
) -> dict[str, Any]:
    """Classify every record as valid / excluded / not_selected and build the report."""
    _DETAILS.clear()
    selected = frozenset(cfg.classes)
    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        list(pool.map(_classify, ((cfg, Path(class_root), r, selected) for r in records)))

    excluded = [
        {"image_path": r.rel_path, "class_name": r.class_name or r.class_folder,
         "reason": r.reason, "detail": _DETAILS.get(r.rel_path, "")}
        for r in records if r.status == STATUS_EXCLUDED
    ]
    by_reason = Counter(e["reason"] for e in excluded)
    valid_per_class = Counter(r.class_name for r in records if r.status == STATUS_VALID)
    present_folders = {
        canonical_class_name(d.name) for d in Path(class_root).iterdir() if d.is_dir()
    } - {None}
    empty_folders = [c for c in cfg.classes if c in present_folders and valid_per_class[c] == 0]
    missing_folders = [c for c in cfg.classes if c not in present_folders]

    report = {
        "total_records": len(records),
        "valid": sum(1 for r in records if r.status == STATUS_VALID),
        "excluded": len(excluded),
        "not_selected": sum(1 for r in records if r.status == STATUS_NOT_SELECTED),
        "excluded_by_reason": dict(sorted(by_reason.items())),
        "valid_per_class": dict(sorted(valid_per_class.items())),
        "selected_classes_without_valid_images": empty_folders,
        "selected_classes_missing_folder": missing_folders,
        "thresholds": {"min_image_dim": cfg.min_image_dim, "max_image_dim": cfg.max_image_dim,
                       "supported_extensions": list(cfg.image_extensions)},
        "excluded_files": excluded,
        "note": "Excluded files are only listed here; nothing was deleted or modified.",
    }
    LOG.info("Cleaning: %d valid, %d excluded (%s), %d not selected",
             report["valid"], report["excluded"], dict(by_reason), report["not_selected"])
    return report


def write_cleaning_report(cfg: PreprocessConfig, report: dict[str, Any]) -> None:
    """Persist ``corrupt_image_report.json`` and ``excluded_files.csv``."""
    write_json(cfg.reports_dir / "corrupt_image_report.json", report)
    write_csv(cfg.reports_dir / "excluded_files.csv",
              ["image_path", "class_name", "reason", "detail"], report["excluded_files"])
