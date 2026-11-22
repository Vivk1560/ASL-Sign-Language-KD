"""Reproducible inference benchmark (model forward pass only, synthetic input).

Method: fixed-seed random input of shape [batch, 3, S, S]; ``warmup`` untimed passes; then ``repeats`` blocks of
``iterations`` timed passes under ``torch.inference_mode`` (CUDA is synchronised around every timed pass).
Latency statistics are over all timed passes; throughput is ``batch_size / mean latency``. Preprocessing,
image decoding, camera I/O and hand detection are NOT included. Nothing is hardcoded: results exist only
after the benchmark has been run, on the machine that ran it.

    python -m deployment.benchmark --student-checkpoint ... --teacher-checkpoint ... --devices cpu cuda
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from teacher_model.model import count_parameters
from teacher_model.utils import setup_logging, write_json

from .predictor import DeploymentError, default_checkpoint, load_model


def _percentile(sorted_values: list[float], q: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def summarize_latencies(latencies_s: list[float], batch_size: int) -> dict[str, float]:
    if not latencies_s:
        raise ValueError("no latency samples")
    ms = sorted(x * 1000.0 for x in latencies_s)
    mean = statistics.fmean(ms)
    return {
        "mean_ms": mean, "median_ms": statistics.median(ms),
        "std_ms": statistics.pstdev(ms) if len(ms) > 1 else 0.0,
        "min_ms": ms[0], "max_ms": ms[-1], "p95_ms": _percentile(ms, 0.95),
        "throughput_images_per_s": batch_size / (mean / 1000.0) if mean > 0 else float("inf"),
        "samples": len(ms),
    }


def benchmark_model(model: nn.Module, device: str | torch.device, input_size: int = 224, batch_size: int = 1,
                    warmup: int = 20, iterations: int = 50, repeats: int = 3, seed: int = 0,
                    timer: Callable[[], float] = time.perf_counter) -> dict[str, Any]:
    """Benchmark one model on one device. ``timer`` is injectable so tests need no real timing."""
    if min(batch_size, iterations, repeats, input_size) < 1 or warmup < 0:
        raise ValueError("batch_size, iterations, repeats and input_size must be >= 1 and warmup >= 0")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise DeploymentError("CUDA benchmark requested but CUDA is not available.")
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    model = model.to(device).eval()
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, 3, input_size, input_size, generator=generator).to(device)
    all_samples: list[float] = []
    repeat_means: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        sync()
        for _ in range(repeats):
            block: list[float] = []
            for _ in range(iterations):
                sync()
                t0 = timer()
                model(x)
                sync()
                block.append(timer() - t0)
            repeat_means.append(statistics.fmean(block) * 1000.0)
            all_samples.extend(block)
    result = summarize_latencies(all_samples, batch_size)
    result.update({"repeat_mean_ms": repeat_means, "batch_size": batch_size, "input_size": input_size,
                   "warmup": warmup, "iterations": iterations, "repeats": repeats, "seed": seed,
                   "device": str(device),
                   "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or platform.machine()})
    return result


def environment_info() -> dict[str, Any]:
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "cpu_threads": torch.get_num_threads(), "logical_cpus": os.cpu_count()}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m deployment.benchmark",
                                description="Benchmark student and/or teacher inference latency.")
    p.add_argument("--models", nargs="+", choices=["student", "teacher"], default=["student", "teacher"])
    p.add_argument("--student-checkpoint", type=Path, default=None)
    p.add_argument("--teacher-checkpoint", type=Path, default=None)
    p.add_argument("--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu", "cuda"],
                   help="cuda is skipped (and reported as skipped) when no GPU is available")
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 32])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch default)")
    p.add_argument("--output", type=Path, default=Path("deployment_reports/benchmark.json"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()
    if args.threads:
        torch.set_num_threads(args.threads)
    paths = {"student": args.student_checkpoint or default_checkpoint("student"),
             "teacher": args.teacher_checkpoint or default_checkpoint("teacher")}
    report: dict[str, Any] = {"environment": environment_info(), "models": {}, "skipped": []}
    try:
        for kind in args.models:
            model, ckpt = load_model(paths[kind], kind, torch.device("cpu"))
            entry: dict[str, Any] = {
                "checkpoint": str(paths[kind]),
                "checkpoint_megabytes": round(Path(paths[kind]).stat().st_size / 1e6, 3),
                "parameters": count_parameters(model), "runs": []}
            for dev in args.devices:
                if dev == "cuda" and not torch.cuda.is_available():
                    report["skipped"].append(f"{kind}/cuda: CUDA not available on this machine")
                    continue
                for bs in args.batch_sizes:
                    run = benchmark_model(model, dev, int(ckpt["input_size"]), bs, args.warmup,
                                          args.iterations, args.repeats, args.seed)
                    entry["runs"].append(run)
                    print(f"{kind:8s} {run['device']:5s} batch={bs:<3d} mean={run['mean_ms']:.2f} ms "
                          f"median={run['median_ms']:.2f} ms  {run['throughput_images_per_s']:.1f} img/s")
                model.to("cpu")
            report["models"][kind] = entry
    except DeploymentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for note in report["skipped"]:
        print(f"skipped: {note}")
    write_json(args.output, report)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
