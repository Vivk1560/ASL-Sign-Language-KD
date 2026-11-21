"""Command-line entry point:  python -m teacher_model.train"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import TeacherConfig
from .utils import resolve_device, seed_everything, setup_logging, write_json

LOG = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    d = TeacherConfig()
    p = argparse.ArgumentParser(prog="python -m teacher_model.train",
                                description="Train the ResNet50 teacher on the preprocessing manifests.")
    p.add_argument("--processed-dir", type=Path, default=d.processed_dir,
                   help="Preprocessing output folder (contains manifests/ and metadata/).")
    p.add_argument("--manifest-dir", type=Path, default=None, help="Override <processed-dir>/manifests.")
    p.add_argument("--dataset-root", type=Path, default=None,
                   help="Folder containing the A, B, ... class folders (default: from dataset_summary.json).")
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--learning-rate", type=float, default=d.learning_rate)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--optimizer", choices=["adamw", "sgd"], default=d.optimizer)
    p.add_argument("--monitor", choices=["val_loss", "val_accuracy"], default=d.monitor)
    p.add_argument("--patience", type=int, default=d.early_stopping_patience)
    p.add_argument("--rotation-deg", type=float, default=d.rotation_deg,
                   help="Max |angle| for random scipy.ndimage.rotate augmentation (default 15).")
    p.add_argument("--image-size", type=int, default=d.image_size)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--no-pretrained", action="store_true",
                   help="Random init (no ImageNet download needed).")
    p.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=d.device)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--output-dir", type=Path, default=d.output_dir)
    p.add_argument("--checkpoint-name", default=d.checkpoint_name)
    p.add_argument("--resume", type=Path, default=None, help="Resume from a *_last.pth checkpoint.")
    p.add_argument("--overwrite", action="store_true", help="Allow replacing an existing best checkpoint.")
    p.add_argument("--max-train-batches", type=int, default=None, help="Smoke-test only: cap batches/epoch.")
    return p


def config_from_args(args: argparse.Namespace) -> TeacherConfig:
    cfg = TeacherConfig(
        processed_dir=args.processed_dir, manifest_dir=args.manifest_dir, dataset_root=args.dataset_root,
        image_size=args.image_size, rotation_deg=args.rotation_deg, pretrained=not args.no_pretrained,
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, optimizer=args.optimizer, monitor=args.monitor,
        early_stopping_patience=args.patience, num_workers=args.num_workers, device=args.device,
        mixed_precision=not args.no_amp, seed=args.seed, output_dir=args.output_dir,
        checkpoint_name=args.checkpoint_name, resume=args.resume, overwrite=args.overwrite,
        max_train_batches=args.max_train_batches)
    cfg.validate()
    return cfg


def run(cfg: TeacherConfig) -> dict:
    """Full pipeline: train -> select best on validation -> final evaluation on internal test."""
    from .checkpoint import load_teacher
    from .dataset import build_dataloaders, build_datasets
    from .evaluate import evaluate_split, save_evaluation
    from .model import build_teacher, count_parameters
    from .trainer import fit

    if cfg.checkpoint_path.exists() and not cfg.resume and not cfg.overwrite:
        raise SystemExit(f"{cfg.checkpoint_path} already exists. Use --overwrite to replace it, "
                         "--resume <last.pth> to continue, or --checkpoint-name for a new file.")
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    datasets, labels = build_datasets(cfg)
    class_names, class_to_idx = list(labels["classes"]), dict(labels["class_to_idx"])
    loaders = build_dataloaders(datasets, cfg, device.type == "cuda")
    model = build_teacher(len(class_names), pretrained=cfg.pretrained and cfg.resume is None)
    LOG.info("Parameters: %s", count_parameters(model))

    summary = fit(model, loaders, cfg, device, class_names, class_to_idx)
    write_json(cfg.reports_dir / "training_history.json", summary)

    # Final evaluation: best-on-validation checkpoint, internal test split, touched only now.
    best_model, _ = load_teacher(cfg.checkpoint_path, device)
    val_metrics = evaluate_split(best_model, loaders["validation"], device, class_names)
    test_metrics = evaluate_split(best_model, loaders["test"], device, class_names)
    save_evaluation(val_metrics, cfg.reports_dir, "validation", class_names)
    save_evaluation(test_metrics, cfg.reports_dir, "test", class_names)
    write_json(cfg.reports_dir / "run_config.json", cfg.to_dict())
    LOG.info("Best epoch %d | validation acc %.4f | INTERNAL TEST acc %.4f macro-F1 %.4f",
             summary["best_epoch"], val_metrics["accuracy"], test_metrics["accuracy"],
             test_metrics["f1_macro"])
    return {"training": summary, "validation": val_metrics["accuracy"], "test": test_metrics["accuracy"]}


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        cfg = config_from_args(args)
        run(cfg)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        LOG.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
