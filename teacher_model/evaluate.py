"""Evaluation of a trained teacher on the validation / internal test split."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .checkpoint import load_teacher
from .metrics import compute_metrics, save_confusion_matrix_png
from .utils import resolve_device, setup_logging, write_json

LOG = logging.getLogger(__name__)


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device
                        ) -> tuple[list[int], list[int]]:
    """Run the model in eval mode and return ``(y_true, y_pred)``."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, targets in loader:
        logits = model(images.to(device, non_blocking=True))
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        y_true.extend(targets.tolist())
    return y_true, y_pred


def evaluate_split(model: nn.Module, loader: DataLoader, device: torch.device,
                   class_names: list[str]) -> dict[str, Any]:
    y_true, y_pred = collect_predictions(model, loader, device)
    return compute_metrics(y_true, y_pred, class_names)


def save_evaluation(metrics: dict[str, Any], reports_dir: Path, split: str,
                    class_names: list[str]) -> dict[str, Path]:
    """Write ``<split>_metrics.json``, ``<split>_classification_report.txt`` and the confusion PNG."""
    reports_dir = Path(reports_dir)
    paths = {
        "metrics": write_json(reports_dir / f"{split}_metrics.json", metrics),
        "confusion_matrix": save_confusion_matrix_png(
            metrics["confusion_matrix"], class_names, reports_dir / f"{split}_confusion_matrix.png",
            title=f"Teacher ResNet50 - {split} confusion matrix"),
    }
    report_path = reports_dir / f"{split}_classification_report.txt"
    report_path.write_text(metrics["classification_report"], encoding="utf-8")
    paths["classification_report"] = report_path
    return paths


def main(argv: list[str] | None = None) -> int:
    """Evaluate an existing checkpoint on the internal test split (final-evaluation step only)."""
    from .config import TeacherConfig
    from .dataset import build_dataloaders, build_datasets

    parser = argparse.ArgumentParser(description="Evaluate a trained ResNet50 teacher.")
    parser.add_argument("--checkpoint", type=Path, default=TeacherConfig().checkpoint_path)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--processed-dir", type=Path, default=TeacherConfig().processed_dir)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--reports-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    setup_logging()
    device = resolve_device(args.device)
    model, ckpt = load_teacher(args.checkpoint, device)
    cfg = TeacherConfig(processed_dir=args.processed_dir, dataset_root=args.dataset_root,
                        batch_size=args.batch_size, num_workers=args.num_workers,
                        image_size=int(ckpt.get("input_size", 224)), metrics_dir=args.reports_dir,
                        device=args.device)
    datasets, labels = build_datasets(cfg)
    if list(labels["classes"]) != list(ckpt["class_names"]):
        raise SystemExit("labels.json classes differ from the checkpoint's class list.")
    loader = build_dataloaders({args.split: datasets[args.split]}, cfg, device.type == "cuda")[args.split]
    metrics = evaluate_split(model, loader, device, list(ckpt["class_names"]))
    save_evaluation(metrics, cfg.reports_dir, args.split, list(ckpt["class_names"]))
    LOG.info("%s accuracy=%.4f macro-F1=%.4f (n=%d)", args.split, metrics["accuracy"],
             metrics["f1_macro"], metrics["num_samples"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
