"""Evaluation of a trained student on the validation / internal test split (never the external test set)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from torch import nn

from teacher_model.evaluate import collect_predictions, evaluate_split  # noqa: F401  (re-exported, reused)
from teacher_model.metrics import save_confusion_matrix_png
from teacher_model.model import count_parameters
from teacher_model.utils import resolve_device, setup_logging, write_json

from .checkpoint import load_student

LOG = logging.getLogger(__name__)


def model_footprint(model: nn.Module, checkpoint_path: Path | None = None) -> dict[str, Any]:
    """Measured size facts only: parameter counts, fp32 parameter bytes and (optionally) checkpoint file size.
    No speed or accuracy claim is made here."""
    counts = count_parameters(model)
    info: dict[str, Any] = {**counts, "parameter_bytes_fp32": sum(p.numel() * p.element_size()
                                                                  for p in model.parameters())}
    info["parameter_megabytes_fp32"] = round(info["parameter_bytes_fp32"] / 1e6, 3)
    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        info["checkpoint_file_megabytes"] = round(Path(checkpoint_path).stat().st_size / 1e6, 3)
    return info


def save_student_evaluation(metrics: dict[str, Any], reports_dir: Path, split: str,
                            class_names: list[str], model_name: str = "Student MobileNetV2") -> dict[str, Path]:
    """Write ``<split>_metrics.json``, ``<split>_classification_report.txt`` and the confusion-matrix PNG."""
    reports_dir = Path(reports_dir)
    paths = {
        "metrics": write_json(reports_dir / f"{split}_metrics.json", metrics),
        "confusion_matrix": save_confusion_matrix_png(
            metrics["confusion_matrix"], class_names, reports_dir / f"{split}_confusion_matrix.png",
            title=f"{model_name} - {split} confusion matrix"),
    }
    report_path = reports_dir / f"{split}_classification_report.txt"
    report_path.write_text(metrics["classification_report"], encoding="utf-8")
    paths["classification_report"] = report_path
    return paths


def main(argv: list[str] | None = None) -> int:
    """Evaluate an existing student checkpoint on validation and/or internal test."""
    from .config import StudentConfig
    from teacher_model.dataset import build_dataloaders, build_datasets

    d = StudentConfig()
    parser = argparse.ArgumentParser(prog="python -m student_model.evaluate",
                                     description="Evaluate a trained MobileNetV2 student.")
    parser.add_argument("--checkpoint", type=Path, default=d.checkpoint_path)
    parser.add_argument("--split", choices=["validation", "test", "both"], default="test")
    parser.add_argument("--processed-dir", type=Path, default=d.processed_dir)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--reports-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    setup_logging()
    device = resolve_device(args.device)
    model, ckpt = load_student(args.checkpoint, device)
    cfg = StudentConfig(processed_dir=args.processed_dir, dataset_root=args.dataset_root,
                        batch_size=args.batch_size, num_workers=args.num_workers, device=args.device,
                        image_size=int(ckpt.get("input_size", 224)), metrics_dir=args.reports_dir,
                        output_dir=args.checkpoint.parent)
    datasets, labels = build_datasets(cfg.to_data_config())
    class_names = list(ckpt["class_names"])
    if list(labels["classes"]) != class_names:
        raise SystemExit("labels.json classes differ from the checkpoint's class list.")
    splits = ["validation", "test"] if args.split == "both" else [args.split]
    loaders = build_dataloaders({s: datasets[s] for s in splits}, cfg.to_data_config(), device.type == "cuda")
    for split in splits:
        metrics = evaluate_split(model, loaders[split], device, class_names)
        metrics["model_footprint"] = model_footprint(model, args.checkpoint)
        save_student_evaluation(metrics, cfg.reports_dir, split, class_names)
        LOG.info("%s accuracy=%.4f macro-F1=%.4f (n=%d)", split, metrics["accuracy"], metrics["f1_macro"],
                 metrics["num_samples"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
