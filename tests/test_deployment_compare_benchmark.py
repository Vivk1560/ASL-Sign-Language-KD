"""Partition 4: teacher-vs-student comparison and the benchmark harness (mocks, no real timing assertions)."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from deployment import benchmark, compare
from deployment.predictor import DeploymentError
from preprocessing.config import ALL_CLASSES
from student_model.checkpoint import save_student_checkpoint
from student_model.model import build_student

CLASSES = ["A", "B", "C"]


class LookupModel(nn.Module):
    """Predicts ``answers[i]`` for the i-th sample seen (fake model with a known accuracy)."""

    def __init__(self, answers, n_params=1):
        super().__init__()
        self.answers = list(answers)
        self.seen = 0
        self.w = nn.Parameter(torch.zeros(n_params))

    def forward(self, x):
        out = torch.zeros(x.shape[0], len(CLASSES))
        for row in range(x.shape[0]):
            out[row, self.answers[self.seen % len(self.answers)]] = 1.0
            self.seen += 1
        return out


def _loader(targets, batch=2):
    xs = torch.zeros(len(targets), 3, 4, 4)
    ys = torch.tensor(targets)
    return [(xs[i:i + batch], ys[i:i + batch]) for i in range(0, len(targets), batch)]


# ------------------------------------------------------------------ compare
def test_compare_models_metrics_and_agreement():
    targets = [0, 1, 2, 0]
    teacher = LookupModel([0, 1, 2, 0])          # 4/4 correct
    student = LookupModel([0, 1, 1, 1])          # samples 0,1 correct only
    res = compare.compare_models(teacher, student, _loader(targets), torch.device("cpu"), CLASSES)
    assert res["num_samples"] == 4 and res["partial_run"] is False
    assert res["teacher"]["accuracy"] == 1.0 and res["student"]["accuracy"] == 0.5
    ag = res["agreement"]
    assert (ag["both_correct"], ag["teacher_only_correct"], ag["student_only_correct"], ag["both_wrong"]) == (2, 2, 0, 0)
    assert ag["prediction_agreement"] == 0.5
    for key in compare.SUMMARY_KEYS:
        assert key in res["teacher"] and key in res["student"]


def test_both_models_see_identical_inputs():
    seen = {"t": [], "s": []}

    class Rec(nn.Module):
        def __init__(self, key):
            super().__init__()
            self.key = key

        def forward(self, x):
            seen[self.key].append(x.clone())
            return torch.zeros(x.shape[0], 3)

    xs = torch.arange(4 * 3 * 2 * 2, dtype=torch.float32).reshape(4, 3, 2, 2)
    loader = [(xs[:2], torch.tensor([0, 1])), (xs[2:], torch.tensor([2, 0]))]
    compare.compare_models(Rec("t"), Rec("s"), loader, torch.device("cpu"), CLASSES)
    assert all(torch.equal(a, b) for a, b in zip(seen["t"], seen["s"])) and len(seen["t"]) == 2


def test_compare_partial_and_empty():
    t, s = LookupModel([0]), LookupModel([0])
    res = compare.compare_models(t, s, _loader([0] * 6), torch.device("cpu"), CLASSES, max_batches=1)
    assert res["num_samples"] == 2 and res["partial_run"] is True
    with pytest.raises(DeploymentError):
        compare.compare_models(t, s, [], torch.device("cpu"), CLASSES)


def test_footprint_comparison_math():
    big, small = nn.Linear(100, 10), nn.Linear(10, 10)
    fp = compare.footprint_comparison(big, small, None, None)
    assert fp["teacher"]["total"] == 1010 and fp["student"]["total"] == 110
    assert fp["parameter_reduction_percent"] == round(100 * (1 - 110 / 1010), 2)
    assert "checkpoint_size_ratio_teacher_over_student" not in fp


def test_footprint_includes_checkpoint_sizes(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"x" * 4000)
    b.write_bytes(b"x" * 1000)
    fp = compare.footprint_comparison(nn.Linear(4, 4), nn.Linear(2, 2), a, b)
    assert fp["checkpoint_size_ratio_teacher_over_student"] == 4.0


def test_markdown_report_flags_partial_runs():
    res = compare.compare_models(LookupModel([0]), LookupModel([0]), _loader([0, 0]), torch.device("cpu"), CLASSES, 1)
    res["footprint"] = compare.footprint_comparison(nn.Linear(4, 4), nn.Linear(2, 2), None, None)
    md = compare.format_markdown(res, "test")
    assert "PARTIAL RUN" in md and "Parameter reduction" in md and "f1_macro" in md


def test_compare_cli_missing_checkpoints_return_2(tmp_path, capsys):
    code = compare.main(["--student-checkpoint", str(tmp_path / "s.pth"), "--teacher-checkpoint", str(tmp_path / "t.pth"),
                         "--reports-dir", str(tmp_path / "r")])
    assert code == 2 and "error:" in capsys.readouterr().err


# ------------------------------------------------------------------ benchmark
def _fake_timer(step=0.001):
    counter = itertools.count()
    return lambda: next(counter) * step


def test_summarize_latencies_known_values():
    s = benchmark.summarize_latencies([0.001, 0.002, 0.003, 0.004], batch_size=2)
    assert s["mean_ms"] == pytest.approx(2.5) and s["median_ms"] == pytest.approx(2.5)
    assert s["min_ms"] == pytest.approx(1.0) and s["max_ms"] == pytest.approx(4.0)
    assert s["throughput_images_per_s"] == pytest.approx(2 / 0.0025)
    assert s["samples"] == 4 and s["p95_ms"] <= s["max_ms"]
    with pytest.raises(ValueError):
        benchmark.summarize_latencies([], 1)


def test_benchmark_counts_warmup_and_repeats_with_fake_timer():
    calls = {"n": 0}

    class Counter(nn.Module):
        def forward(self, x):
            calls["n"] += 1
            return x.mean(dim=(1, 2, 3))

    res = benchmark.benchmark_model(Counter(), "cpu", input_size=8, batch_size=4, warmup=3, iterations=5,
                                    repeats=2, timer=_fake_timer(0.002))
    assert calls["n"] == 3 + 5 * 2
    assert res["samples"] == 10 and len(res["repeat_mean_ms"]) == 2
    assert res["mean_ms"] == pytest.approx(2.0) and res["median_ms"] == pytest.approx(2.0)
    assert res["throughput_images_per_s"] == pytest.approx(4 / 0.002)
    assert res["device"] == "cpu" and res["batch_size"] == 4 and res["seed"] == 0


def test_benchmark_input_is_seeded_and_model_set_to_eval():
    inputs = []

    class Rec(nn.Module):
        def forward(self, x):
            inputs.append(x.clone())
            assert not self.training
            return x.flatten(1)

    m = Rec().train()
    benchmark.benchmark_model(m, "cpu", 8, 2, 0, 1, 1, seed=5)
    benchmark.benchmark_model(m, "cpu", 8, 2, 0, 1, 1, seed=5)
    assert torch.equal(inputs[0], inputs[1]) and inputs[0].shape == (2, 3, 8, 8)


def test_benchmark_validation_and_missing_cuda(monkeypatch):
    with pytest.raises(ValueError):
        benchmark.benchmark_model(nn.Identity(), "cpu", 8, 0)
    with pytest.raises(ValueError):
        benchmark.benchmark_model(nn.Identity(), "cpu", 8, 1, warmup=-1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(DeploymentError, match="CUDA"):
        benchmark.benchmark_model(nn.Identity(), "cuda", 8)


def test_benchmark_cli_end_to_end_with_tiny_student(tmp_path, monkeypatch, capsys):
    torch.manual_seed(0)
    ckpt = save_student_checkpoint(
        tmp_path / "student_mobilenetv2_kd_best.pth", build_student(29, pretrained=False), epoch=1,
        best_val_metric=0.1, best_metric_name="val_loss", class_names=list(ALL_CLASSES),
        class_to_idx={c: i for i, c in enumerate(ALL_CLASSES)}, config={"image_size": 32}, distillation={}, teacher={})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    out = tmp_path / "bench.json"
    code = benchmark.main(["--models", "student", "--student-checkpoint", str(ckpt), "--devices", "cpu", "cuda",
                           "--batch-sizes", "1", "--warmup", "1", "--iterations", "2", "--repeats", "2",
                           "--output", str(out)])
    assert code == 0
    report = json.loads(out.read_text())
    student = report["models"]["student"]
    assert student["parameters"]["total"] > 2_000_000 and student["checkpoint_megabytes"] > 0
    assert len(student["runs"]) == 1 and student["runs"][0]["device"] == "cpu"
    assert any("cuda" in s.lower() for s in report["skipped"])       # GPU honestly reported as skipped
    assert report["environment"]["torch"] == torch.__version__


def test_benchmark_cli_missing_checkpoint_returns_2(tmp_path, capsys):
    assert benchmark.main(["--models", "student", "--student-checkpoint", str(tmp_path / "x.pth"),
                           "--output", str(tmp_path / "o.json")]) == 2
    assert "error:" in capsys.readouterr().err
