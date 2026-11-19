"""Preprocessing orchestrator.  Usage:  python -m preprocessing.main [options]

Rerun policy
------------
* The raw dataset is never written to (verified by a before/after fingerprint).
* Processed outputs are deterministic and rewritten atomically on a rerun with
  identical dataset + settings.
* If the dataset fingerprint or any output-affecting setting changed since the
  last successful run, the run aborts unless ``--force`` is given; ``--force``
  first moves the previous outputs into ``processed/_archive/<timestamp>/``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import PIL

from . import __version__
from .config import PreprocessConfig, load_config
from .data_augmentation import TrainAugmentation, generate_offline_augmentations
from .data_cleaning import clean_records, write_cleaning_report
from .data_splitting import build_splits, write_manifests
from .data_visualization import (plot_class_distribution, plot_dataset_summary,
                                 plot_image_dimensions, plot_sample_images,
                                 plot_split_distribution)
from .dataset_inspection import (compute_raw_fingerprint, enumerate_records, inspect_dataset)
from .dataset_validation import validate_outputs
from .download_dataset import ensure_dataset
from .duplicate_detection import (assign_duplicate_groups, compute_hashes,
                                  write_duplicate_reports)
from .labels import write_labels, load_labels
from .records import atomic_write_text, read_json, write_json

LOG = logging.getLogger("preprocessing")
EXIT_OK, EXIT_FAILED, EXIT_NO_DATASET, EXIT_NEEDS_FORCE = 0, 1, 2, 3


def _setup_logging(cfg: PreprocessConfig, level: str) -> Path:
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.logs_dir / f"preprocessing_{dt.datetime.now():%Y%m%d_%H%M%S}.log"
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(fmt)
        root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    return log_path


def _archive_previous_outputs(cfg: PreprocessConfig) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = cfg.archive_dir / stamp
    dest.mkdir(parents=True, exist_ok=True)
    for src in (cfg.manifests_dir, cfg.metadata_dir, cfg.augmented_dir):
        if src.exists():
            shutil.move(str(src), str(dest / src.name))
    if cfg.reports_dir.exists():
        (dest / "reports").mkdir(exist_ok=True)
        for item in cfg.reports_dir.iterdir():
            if item.name != "logs":
                shutil.move(str(item), str(dest / "reports" / item.name))
    return dest


def _stage(n: int, title: str) -> float:
    LOG.info("=== Step %d: %s ===", n, title)
    return time.perf_counter()


def run_pipeline(cfg: PreprocessConfig, *, download: bool = False, extract_zip: bool = False,
                 force: bool = False, offline_augment: bool | None = None,
                 log_level: str = "INFO") -> int:
    """Execute the full pipeline and return a process exit code."""
    cfg.validate()
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    log_path = _setup_logging(cfg, log_level)
    LOG.info("ASL preprocessing v%s | python %s | class_set=%s seed=%d",
             __version__, platform.python_version(), cfg.class_set, cfg.seed)
    do_offline = cfg.augmentation.offline_enabled if offline_augment is None else offline_augment

    # 1-2. availability
    _stage(1, "Check dataset availability / locate or download")
    status = ensure_dataset(cfg, download=download, extract_zip=extract_zip)
    if not status.available:
        LOG.error("Dataset unavailable. %s", status.message)
        return EXIT_NO_DATASET
    class_root = status.class_root
    LOG.info("Dataset source: %s (%s)", status.source, class_root)

    # rerun policy
    fingerprint_before = compute_raw_fingerprint(class_root)
    if cfg.run_state_path.is_file():
        prev = read_json(cfg.run_state_path)
        if (prev.get("raw_fingerprint") != fingerprint_before
                or prev.get("config_hash") != cfg.config_hash()):
            if not force:
                LOG.error("Existing outputs were produced from a different dataset/configuration. "
                          "Re-run with --force to archive them and regenerate.")
                return EXIT_NEEDS_FORCE
            LOG.warning("--force: archiving previous outputs to %s", _archive_previous_outputs(cfg))
            _setup_logging(cfg, log_level)

    # 3. inspect
    t = _stage(3, "Inspect folder structure")
    records, structure = enumerate_records(class_root)
    summary = inspect_dataset(cfg, class_root, records, structure)
    if summary["missing_selected_classes"]:
        LOG.warning("Selected classes missing from dataset: %s", summary["missing_selected_classes"])
    if summary["unexpected_folders"]:
        LOG.warning("Unexpected class folders: %s", summary["unexpected_folders"])

    # 4. clean
    _stage(4, "Validate image files")
    cleaning = clean_records(cfg, class_root, records)
    write_cleaning_report(cfg, cleaning)

    # 5. duplicates
    _stage(5, "Detect exact and near duplicates")
    compute_hashes(cfg, class_root, records)
    duplicates = assign_duplicate_groups(cfg, records)
    write_duplicate_reports(cfg, duplicates)

    # 6. visualisation
    if cfg.generate_visualizations:
        _stage(6, "Generate dataset visualizations")
        plot_class_distribution(cfg, summary)
        if summary["samples"]:
            plot_sample_images(cfg, class_root, summary)
        if summary["dimensions"]["images_measured"]:
            plot_image_dimensions(cfg, records)
        plot_dataset_summary(cfg, cleaning, duplicates)

    # 7. labels
    _stage(7, "Generate class mapping (labels.json)")
    write_labels(cfg)
    labels = load_labels(cfg.labels_path)

    # 8. splits
    _stage(8, "Create reproducible group-aware splits")
    split = build_splits(cfg, records)
    write_json(cfg.reports_dir / "split_report.json",
               {k: v for k, v in split.items() if k != "rows"})
    critical = [i for i in split["issues"] if i["severity"] == "critical"]
    if critical:
        LOG.error("Split cannot be made reliably (%d critical issue(s)); see split_report.json. "
                  "First: %s", len(critical), critical[0])
        return EXIT_FAILED
    write_manifests(cfg, split["rows"])
    if cfg.generate_visualizations:
        plot_split_distribution(cfg, split["per_class"])

    # 9. augmentation (training split only)
    _stage(9, "Training-only augmentation")
    aug = TrainAugmentation(cfg.augmentation, seed=cfg.seed)
    aug_rows: list[dict[str, Any]] = []
    if do_offline:
        aug_rows = generate_offline_augmentations(cfg, class_root, split["rows"]["train"])
    else:
        LOG.info("On-the-fly augmentation selected (TrainAugmentation); no offline images written.")

    # 10. validation
    _stage(10, "Validate processed outputs")
    report = validate_outputs(cfg, class_root, labels, summary["unexpected_folders"],
                              fingerprint_before)
    write_json(cfg.reports_dir / "dataset_validation_report.json", report.to_dict())
    if not report.ok:
        failed_dir = cfg.manifests_dir / "_failed_validation"
        failed_dir.mkdir(exist_ok=True)
        for s in ("train", "validation", "test"):
            if cfg.manifest_path(s).exists():
                shutil.move(str(cfg.manifest_path(s)), str(failed_dir / f"{s}.csv"))
        LOG.error("Validation FAILED: %s. Manifests moved to %s - do not use them.",
                  [c.name for c in report.critical_failures], failed_dir)
        return EXIT_FAILED

    # 11. metadata + final reports
    _stage(11, "Write metadata and final reports")
    final = {
        **summary,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset_source": status.source,
        "cleaning": {k: v for k, v in cleaning.items() if k != "excluded_files"},
        "duplicates": {k: v for k, v in duplicates.items() if k not in ("groups", "per_class")},
        "splits": {"sizes": split["sizes"], "per_class": split["per_class"],
                   "issues": split["issues"], "seed": cfg.seed, "ratios": list(cfg.split_ratios)},
        "class_handling": {"class_set": cfg.class_set, "classes": list(cfg.classes),
                           "excluded_classes": summary["excluded_classes"]},
        "augmentation": {"mode": "offline" if do_offline else "on-the-fly",
                         "offline_images": len(aug_rows), "parameters": aug.signature()},
        "raw_fingerprint": fingerprint_before,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "pillow": PIL.__version__,
                        "pipeline_version": __version__},
        "validation_ok": report.ok,
    }
    write_json(cfg.summary_path, final)
    write_json(cfg.config_path, cfg.to_dict())
    write_json(cfg.run_state_path, {"raw_fingerprint": fingerprint_before,
                                    "config_hash": cfg.config_hash(), "status": "success",
                                    "completed_utc": final["generated_utc"]})
    atomic_write_text(cfg.reports_dir / "preprocessing_report.md", _markdown_report(final))
    LOG.info("Done. Outputs in %s (log: %s)", cfg.processed_dir, log_path)
    return EXIT_OK


def _markdown_report(f: dict[str, Any]) -> str:
    s = f["splits"]["sizes"]
    lines = [
        "# Preprocessing report (generated from this run)", "",
        f"- Generated (UTC): {f['generated_utc']}", f"- Class root: `{f['class_root']}`",
        f"- Dataset source: {f['dataset_source']}",
        f"- Class set: {f['class_handling']['class_set']} ({len(f['class_handling']['classes'])} classes)",
        f"- Total files: {f['total_files']}; valid: {f['cleaning']['valid']}; "
        f"excluded: {f['cleaning']['excluded']}; not selected: {f['cleaning']['not_selected']}",
        f"- Duplicate groups (size>1): {f['duplicates']['duplicate_groups']} covering "
        f"{f['duplicates']['images_in_duplicate_groups']} images",
        f"- Split sizes: train={s['train']}, validation={s['validation']}, test={s['test']}",
        f"- Augmentation: {f['augmentation']['mode']}", f"- Validation passed: {f['validation_ok']}", "",
        "## Excluded classes", "",
    ]
    for name, info in f["excluded_classes"].items():
        lines.append(f"- `{name}`: present={info['present_in_dataset']}, "
                     f"images={info['supported_image_count']}")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m preprocessing.main",
                                description="ASL dataset preprocessing pipeline (Partition 1)")
    p.add_argument("--raw-dir", type=Path, help="dataset location (default: data/raw)")
    p.add_argument("--processed-dir", type=Path, help="output location (default: data/processed)")
    p.add_argument("--seed", type=int)
    p.add_argument("--split-ratios", type=float, nargs=3, metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--class-set", choices=("26", "29"))
    p.add_argument("--near-threshold", type=int, dest="near_duplicate_hamming",
                   help="max dHash Hamming distance treated as near-duplicate (default 4)")
    p.add_argument("--no-near-duplicates", action="store_true")
    p.add_argument("--workers", type=int)
    p.add_argument("--download", action="store_true", help="try Kaggle API if dataset is missing")
    p.add_argument("--extract-zip", action="store_true", help="extract a .zip found in raw dir")
    p.add_argument("--offline-augment", action="store_true", help="write augmented train images")
    p.add_argument("--augment-copies", type=int, help="offline copies per train image")
    p.add_argument("--skip-visualization", action="store_true")
    p.add_argument("--no-decode-validation", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="archive previous outputs and regenerate if dataset/config changed")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides: dict[str, Any] = {
        "raw_dir": args.raw_dir, "processed_dir": args.processed_dir, "seed": args.seed,
        "split_ratios": args.split_ratios, "class_set": args.class_set,
        "near_duplicate_hamming": args.near_duplicate_hamming, "workers": args.workers,
    }
    if args.no_near_duplicates:
        overrides["near_duplicate_detection"] = False
    if args.skip_visualization:
        overrides["generate_visualizations"] = False
    if args.no_decode_validation:
        overrides["decode_check_in_validation"] = False
    cfg = load_config(**overrides)
    if args.augment_copies is not None:
        import dataclasses
        cfg = cfg.with_overrides(augmentation=dataclasses.replace(
            cfg.augmentation, offline_copies_per_image=args.augment_copies))
    return run_pipeline(cfg, download=args.download, extract_zip=args.extract_zip,
                        force=args.force, offline_augment=True if args.offline_augment else None,
                        log_level=args.log_level)


if __name__ == "__main__":
    sys.exit(main())
