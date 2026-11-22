"""Teacher vs student comparison on the validation or internal-test split (never the external test set).

Both models see exactly the same batches (one pass over one DataLoader), are evaluated with the same
``teacher_model.metrics.compute_metrics`` used by the Partition 2/3 evaluators, and their footprints come
from ``student_model.evaluate.model_footprint``. Nothing is hardcoded; every number is computed at run time.

    python -m deployment.compare --student-checkpoint student_output/student_mobilenetv2_kd_best.pth \\
        --teacher-checkpoint model_weights/teacher_resnet50_best.pth --split test
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from student_model.evaluate import model_footprint
from teacher_model.metrics import compute_metrics
from teacher_model.utils import resolve_device, setup_logging, write_json

from .predictor import DeploymentError, default_checkpoint, load_model

LOG = logging.getLogger(__name__)
SUMMARY_KEYS = ("accuracy", "precision_macro", "recall_macro", "f1_macro")


@torch.inference_mode()
def compare_models(teacher: nn.Module, student: nn.Module, loader: Iterable, device: torch.device,
                   class_names: list[str], max_batches: int | None = None) -> dict[str, Any]:
    """Run both models on identical batches and return metrics plus agreement statistics."""
    teacher.eval()
    student.eval()
    y_true: list[int] = []
    t_pred: list[int] = []
    s_pred: list[int] = []
    for n, (images, targets) in enumerate(loader):
        if max_batches is not None and n >= max_batches:
            break
        images = images.to(device)
        t_pred.extend(teacher(images).argmax(dim=1).cpu().tolist())
        s_pred.extend(student(images).argmax(dim=1).cpu().tolist())
        y_true.extend(int(t) for t in targets)
    if not y_true:
        raise DeploymentError("The comparison loader produced no samples.")
    t_ok = [a == b for a, b in zip(t_pred, y_true)]
    s_ok = [a == b for a, b in zip(s_pred, y_true)]
    n = len(y_true)
    return {
        "num_samples": n,
        "partial_run": max_batches is not None,
        "teacher": compute_metrics(y_true, t_pred, class_names),
        "student": compute_metrics(y_true, s_pred, class_names),
        "agreement": {
            "prediction_agreement": sum(a == b for a, b in zip(t_pred, s_pred)) / n,
            "both_correct": sum(a and b for a, b in zip(t_ok, s_ok)),
            "teacher_only_correct": sum(a and not b for a, b in zip(t_ok, s_ok)),
            "student_only_correct": sum(b and not a for a, b in zip(t_ok, s_ok)),
            "both_wrong": sum((not a) and (not b) for a, b in zip(t_ok, s_ok)),
        },
    }


def footprint_comparison(teacher: nn.Module, student: nn.Module, teacher_path: Path | None,
                         student_path: Path | None) -> dict[str, Any]:
    t = model_footprint(teacher, teacher_path)
    s = model_footprint(student, student_path)
    out: dict[str, Any] = {
        "teacher": t, "student": s,
        "parameter_reduction_percent": round(100.0 * (1 - s["total"] / t["total"]), 2),
        "parameter_ratio_teacher_over_student": round(t["total"] / s["total"], 2),
    }
    if "checkpoint_file_megabytes" in t and "checkpoint_file_megabytes" in s:
        out["checkpoint_size_ratio_teacher_over_student"] = round(
            t["checkpoint_file_megabytes"] / s["checkpoint_file_megabytes"], 2)
    return out


def format_markdown(result: dict[str, Any], split: str) -> str:
    rows = ["| Metric | Teacher (ResNet50) | Student (MobileNetV2) |", "|---|---|---|"]
    for key in SUMMARY_KEYS:
        rows.append(f"| {key} | {result['teacher'][key]:.4f} | {result['student'][key]:.4f} |")
    fp = result["footprint"]
    rows.append(f"| parameters | {fp['teacher']['total']:,} | {fp['student']['total']:,} |")
    if "checkpoint_file_megabytes" in fp["teacher"] and "checkpoint_file_megabytes" in fp["student"]:
        rows.append(f"| checkpoint size (MB) | {fp['teacher']['checkpoint_file_megabytes']} "
                    f"| {fp['student']['checkpoint_file_megabytes']} |")
    ag = result["agreement"]
    partial = " (PARTIAL RUN: limited number of batches)" if result["partial_run"] else ""
    return (f"## Teacher vs Student - {split} split, {result['num_samples']} samples{partial}\n\n"
            + "\n".join(rows)
            + f"\n\nParameter reduction: {fp['parameter_reduction_percent']}%. "
              f"Prediction agreement: {ag['prediction_agreement']:.4f} "
              f"(both correct {ag['both_correct']}, teacher-only {ag['teacher_only_correct']}, "
              f"student-only {ag['student_only_correct']}, both wrong {ag['both_wrong']}).\n")


def build_loader(split: str, processed_dir: Path, dataset_root: Path | None, batch_size: int,
                 num_workers: int, image_size: int, device: torch.device):
    """The same manifest-driven loader the Partition 2/3 evaluators use (EvalTransform, no shuffling)."""
    from student_model.config import StudentConfig
    from teacher_model.dataset import build_dataloaders, build_datasets

    cfg = StudentConfig(processed_dir=processed_dir, dataset_root=dataset_root, batch_size=batch_size,
                        num_workers=num_workers, image_size=image_size)
    datasets, labels = build_datasets(cfg.to_data_config())
    loaders = build_dataloaders({split: datasets[split]}, cfg.to_data_config(), device.type == "cuda")
    return loaders[split], list(labels["classes"])


def build_parser() -> argparse.ArgumentParser:
    from student_model.config import StudentConfig
    d = StudentConfig()
    p = argparse.ArgumentParser(prog="python -m deployment.compare",
                                description="Compare the trained teacher and student on one dataset split.")
    p.add_argument("--student-checkpoint", type=Path, default=None)
    p.add_argument("--teacher-checkpoint", type=Path, default=None)
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--processed-dir", type=Path, default=d.processed_dir)
    p.add_argument("--dataset-root", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--max-batches", type=int, default=None, help="smoke-test only; marks the result as partial")
    p.add_argument("--reports-dir", type=Path, default=Path("deployment_reports"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()
    s_path = args.student_checkpoint or default_checkpoint("student")
    t_path = args.teacher_checkpoint or default_checkpoint("teacher")
    try:
        device = resolve_device(args.device)
        student, s_ckpt = load_model(s_path, "student", device)
        teacher, t_ckpt = load_model(t_path, "teacher", device)
        if list(s_ckpt["class_names"]) != list(t_ckpt["class_names"]):
            raise DeploymentError("Teacher and student checkpoints have different class lists.")
        if int(s_ckpt["input_size"]) != int(t_ckpt["input_size"]):
            raise DeploymentError("Teacher and student checkpoints use different input sizes.")
        loader, class_names = build_loader(args.split, args.processed_dir, args.dataset_root, args.batch_size,
                                           args.num_workers, int(s_ckpt["input_size"]), device)
        if class_names != list(s_ckpt["class_names"]):
            raise DeploymentError("labels.json classes differ from the checkpoints' class list.")
    except DeploymentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # dataset/manifest problems from the Partition 2 loader
        print(f"error: could not prepare the {args.split} data: {exc}", file=sys.stderr)
        return 2
    result = compare_models(teacher, student, loader, device, class_names, args.max_batches)
    result["footprint"] = footprint_comparison(teacher, student, t_path, s_path)
    result["split"] = args.split
    result["checkpoints"] = {"teacher": str(t_path), "student": str(s_path)}
    write_json(args.reports_dir / f"teacher_vs_student_{args.split}.json", result)
    md = format_markdown(result, args.split)
    (args.reports_dir / f"teacher_vs_student_{args.split}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
