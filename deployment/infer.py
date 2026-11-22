"""Single-image inference CLI. Works without MediaPipe and without a camera.

    python -m deployment.infer path/to/image.jpg --checkpoint student_output/student_mobilenetv2_kd_best.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .predictor import (DeploymentError, MODEL_KINDS, Predictor, checkpoint_summary, default_checkpoint,
                        open_image)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m deployment.infer",
                                description="Classify one or more ASL alphabet images with a trained checkpoint.")
    p.add_argument("images", nargs="*", type=Path, help="image file(s) to classify")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="checkpoint path (default: $ASL_STUDENT_CHECKPOINT, else "
                        "student_output/student_mobilenetv2_kd_best.pth; for --model teacher: "
                        "$ASL_TEACHER_CHECKPOINT, else model_weights/teacher_resnet50_best.pth)")
    p.add_argument("--model", choices=MODEL_KINDS, default="student")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--json", action="store_true", help="print machine-readable JSON")
    p.add_argument("--info", action="store_true", help="print the checkpoint's stored metadata and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ckpt_path = args.checkpoint or default_checkpoint(args.model)
    try:
        predictor = Predictor.from_checkpoint(ckpt_path, args.model, args.device)
    except DeploymentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.info:
        print(json.dumps(checkpoint_summary(predictor.checkpoint, ckpt_path), indent=2, default=str))
        return 0
    if not args.images:
        print("error: no image given (use --info to inspect the checkpoint only)", file=sys.stderr)
        return 2
    results, failures = [], 0
    for path in args.images:
        try:
            pred = predictor.predict_image(open_image(path), args.top_k)
        except DeploymentError as exc:
            failures += 1
            print(f"error: {exc}", file=sys.stderr)
            results.append({"image": str(path), "error": str(exc)})
            continue
        results.append({"image": str(path), "label": pred.label, "confidence": pred.confidence,
                        "top_k": [{"label": l, "probability": p} for l, p in pred.top_k]})
        if not args.json:
            top = ", ".join(f"{l} {p:.1%}" for l, p in pred.top_k)
            print(f"{path}: {pred.label} ({pred.confidence:.1%})   top-{len(pred.top_k)}: {top}")
    if args.json:
        print(json.dumps({"model": args.model, "checkpoint": str(ckpt_path),
                          "device": str(predictor.device), "results": results}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
