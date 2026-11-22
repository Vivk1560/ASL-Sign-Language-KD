"""Command-line entry point:  python -m student_model.train"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from teacher_model.utils import resolve_device, seed_everything, setup_logging, write_json

from .config import StudentConfig
from .distillation import DEFAULT_ALPHA, DEFAULT_TEMPERATURE

LOG = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    d = StudentConfig()
    p = argparse.ArgumentParser(prog="python -m student_model.train",
                                description="Distil the ResNet50 teacher into a MobileNetV2 student.")
    p.add_argument("--processed-dir", type=Path, default=d.processed_dir,
                   help="Preprocessing output folder (contains manifests/ and metadata/).")
    p.add_argument("--manifest-dir", type=Path, default=None, help="Override <processed-dir>/manifests.")
    p.add_argument("--dataset-root", type=Path, default=None,
                   help="Folder containing the A, B, ... class folders (default: from dataset_summary.json).")
    p.add_argument("--teacher-checkpoint", type=Path, default=d.teacher_checkpoint,
                   help="Trained ResNet50 teacher checkpoint (read-only; never modified).")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="KD temperature T (> 0).")
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                   help="Weight of the supervised CE loss in [0, 1]; KD loss gets (1 - alpha).")
    p.add_argument("--label-smoothing", type=float, default=d.label_smoothing)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--learning-rate", type=float, default=d.learning_rate)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--optimizer", choices=["adamw", "sgd"], default=d.optimizer)
    p.add_argument("--monitor", choices=["val_loss", "val_accuracy"], default=d.monitor)
    p.add_argument("--patience", type=int, default=d.early_stopping_patience)
    p.add_argument("--dropout", type=float, default=d.dropout)
    p.add_argument("--rotation-deg", type=float, default=d.rotation_deg)
    p.add_argument("--image-size", type=int, default=d.image_size,
                   help="Must equal the teacher checkpoint's input size.")
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--no-pretrained", action="store_true", help="Random student init (no ImageNet download).")
    p.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=d.device)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--output-dir", type=Path, default=d.output_dir)
    p.add_argument("--checkpoint-name", default=d.checkpoint_name)
    p.add_argument("--resume", type=Path, default=None, help="Resume from a *_last.pth student checkpoint.")
    p.add_argument("--overwrite", action="store_true", help="Allow replacing an existing best student checkpoint.")
    p.add_argument("--max-train-batches", type=int, default=None, help="Smoke-test only: cap batches/epoch.")
    return p


def config_from_args(args: argparse.Namespace) -> StudentConfig:
    cfg = StudentConfig(
        processed_dir=args.processed_dir, manifest_dir=args.manifest_dir, dataset_root=args.dataset_root,
        image_size=args.image_size, rotation_deg=args.rotation_deg, teacher_checkpoint=args.teacher_checkpoint,
        pretrained=not args.no_pretrained, dropout=args.dropout, temperature=args.temperature,
        alpha=args.alpha, label_smoothing=args.label_smoothing, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        optimizer=args.optimizer, monitor=args.monitor, early_stopping_patience=args.patience,
        num_workers=args.num_workers, device=args.device, mixed_precision=not args.no_amp, seed=args.seed,
        output_dir=args.output_dir, checkpoint_name=args.checkpoint_name, resume=args.resume,
        overwrite=args.overwrite, max_train_batches=args.max_train_batches)
    cfg.validate()
    return cfg


def _guard_checkpoint_paths(cfg: StudentConfig) -> None:
    """The student must never write over the teacher checkpoint (or anything the teacher run produced)."""
    from teacher_model.trainer import last_checkpoint_path

    teacher = Path(cfg.teacher_checkpoint).resolve()
    outputs = {cfg.checkpoint_path.resolve(), last_checkpoint_path(cfg.checkpoint_path).resolve()}
    if teacher in outputs:
        raise ValueError(f"Student output path collides with the teacher checkpoint {teacher}; "
                         "choose another --output-dir / --checkpoint-name.")
    if cfg.checkpoint_path.exists() and not cfg.resume and not cfg.overwrite:
        raise ValueError(f"{cfg.checkpoint_path} already exists. Use --overwrite to replace it, "
                         "--resume <last.pth> to continue, or --checkpoint-name for a new file.")


def run(cfg: StudentConfig) -> dict:
    """Full pipeline: load+verify teacher -> distil -> best-on-validation -> evaluate validation + internal test."""
    from teacher_model.dataset import build_dataloaders, build_datasets, load_label_mapping

    from .checkpoint import load_student
    from .evaluate import evaluate_split, model_footprint, save_student_evaluation
    from .model import build_student, count_parameters
    from .teacher import load_frozen_teacher
    from .trainer import fit

    cfg.validate()
    _guard_checkpoint_paths(cfg)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    data_cfg = cfg.to_data_config()

    # Fail fast on a missing / incompatible teacher before touching the 87k-image dataset.
    labels = load_label_mapping(data_cfg.labels_file)
    class_names, class_to_idx = list(labels["classes"]), dict(labels["class_to_idx"])
    teacher, teacher_info = load_frozen_teacher(cfg.teacher_checkpoint, device, class_names, class_to_idx,
                                                cfg.image_size)
    datasets, _ = build_datasets(data_cfg)          # train / validation / test manifests only
    loaders = build_dataloaders(datasets, data_cfg, device.type == "cuda")
    student = build_student(len(class_names), pretrained=cfg.pretrained and cfg.resume is None,
                            dropout=cfg.dropout)
    LOG.info("Student parameters: %s | teacher parameters: %s", count_parameters(student),
             count_parameters(teacher))

    summary = fit(student, teacher, loaders, cfg, device, class_names, class_to_idx, teacher_info)
    summary["teacher"] = teacher_info
    write_json(cfg.reports_dir / "training_history.json", summary)

    # Final evaluation: best-on-validation student, internal splits only (external_test.csv is never read).
    best, _ = load_student(cfg.checkpoint_path, device)
    results: dict = {"training": summary}
    for split in ("validation", "test"):
        metrics = evaluate_split(best, loaders[split], device, class_names)
        metrics["model_footprint"] = model_footprint(best, cfg.checkpoint_path)
        save_student_evaluation(metrics, cfg.reports_dir, split, class_names)
        ref = evaluate_split(teacher, loaders[split], device, class_names)   # measured teacher reference
        write_json(cfg.reports_dir / f"teacher_reference_{split}_metrics.json",
                   {k: ref[k] for k in ("num_samples", "accuracy", "precision_macro", "recall_macro", "f1_macro")}
                   | {"teacher": teacher_info})
        results[split] = {"student_accuracy": metrics["accuracy"], "student_f1_macro": metrics["f1_macro"],
                          "teacher_accuracy": ref["accuracy"], "teacher_f1_macro": ref["f1_macro"]}
    write_json(cfg.reports_dir / "run_config.json", cfg.to_dict())
    write_json(cfg.reports_dir / "final_summary.json", results)
    LOG.info("Best epoch %d | validation acc %.4f | INTERNAL TEST acc %.4f macro-F1 %.4f "
             "(teacher reference test acc %.4f)", summary["best_epoch"], results["validation"]["student_accuracy"],
             results["test"]["student_accuracy"], results["test"]["student_f1_macro"],
             results["test"]["teacher_accuracy"])
    return results


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
