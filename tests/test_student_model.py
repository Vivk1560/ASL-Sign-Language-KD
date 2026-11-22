"""Tests for the Partition 3 student (MobileNetV2) + Knowledge Distillation pipeline.

Synthetic data only: no ImageNet download, no real dataset and no real teacher checkpoint are needed.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from preprocessing.config import ALL_CLASSES
from student_model import train as student_train
from student_model.checkpoint import load_student, read_student_checkpoint, save_student_checkpoint
from student_model.config import DEFAULT_CHECKPOINT_NAME, StudentConfig
from student_model.distillation import DistillationLoss, kd_divergence, validate_kd_hyperparameters
from student_model.model import build_student, count_parameters
from student_model.teacher import (TeacherCompatibilityError, file_sha256, freeze_teacher,
                                   load_frozen_teacher, teacher_logits, verify_teacher_compatibility)
from student_model.trainer import fit, train_epoch
from teacher_model.checkpoint import load_teacher, read_checkpoint
from teacher_model.checkpoint import save_checkpoint as save_teacher_checkpoint
from teacher_model.config import TeacherConfig
from teacher_model.dataset import build_dataloaders, build_datasets
from teacher_model.model import build_teacher
from teacher_model.utils import write_json

FIELDS = ["image_path", "class_name", "class_index", "sha256", "dhash",
          "group_id", "group_size", "split", "width", "height"]
SIZE = 32
SPLITS = ("train", "validation", "test")
C2I = {c: i for i, c in enumerate(ALL_CLASSES)}


def _make_image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(base).resize((SIZE, SIZE), Image.Resampling.BICUBIC).save(path, quality=95)


@pytest.fixture(scope="module")
def teacher_ckpt(tmp_path_factory) -> Path:
    """A tiny randomly initialised ResNet50 saved with the *teacher's own* checkpoint writer."""
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("teacher") / "teacher_resnet50_best.pth"
    save_teacher_checkpoint(path, build_teacher(29, pretrained=False), epoch=3, best_val_metric=0.5,
                            best_metric_name="val_loss", class_names=list(ALL_CLASSES), class_to_idx=C2I,
                            config={"image_size": SIZE})
    return path


@pytest.fixture()
def synthetic(tmp_path: Path, teacher_ckpt: Path):
    """29 classes x 3 images (1 per split) + manifests / labels / summary, and a matching StudentConfig."""
    class_root = tmp_path / "raw" / "train"
    processed = tmp_path / "processed"
    rows = {s: [] for s in SPLITS}
    seed = 0
    for idx, cls in enumerate(ALL_CLASSES):
        for k, split in enumerate(SPLITS):
            seed += 1
            rel = f"{cls}/{cls}{k}.jpg"
            _make_image(class_root / rel, seed)
            rows[split].append({"image_path": rel, "class_name": cls, "class_index": idx,
                                "sha256": f"sha{seed}", "dhash": "0", "group_id": f"g{seed}",
                                "group_size": 1, "split": split, "width": SIZE, "height": SIZE})
    for split in SPLITS:
        path = processed / "manifests" / f"{split}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
            w.writeheader()
            w.writerows(rows[split])
    write_json(processed / "metadata" / "labels.json", {
        "classes": list(ALL_CLASSES), "num_classes": 29, "class_to_idx": C2I})
    write_json(processed / "metadata" / "dataset_summary.json", {"class_root": str(class_root)})
    cfg = StudentConfig(processed_dir=processed, image_size=SIZE, batch_size=8, num_workers=0, epochs=2,
                        pretrained=False, device="cpu", output_dir=tmp_path / "out",
                        teacher_checkpoint=teacher_ckpt, early_stopping_patience=2, mixed_precision=False)
    return cfg, processed


def _loaders(cfg: StudentConfig):
    datasets, _ = build_datasets(cfg.to_data_config())
    return build_dataloaders(datasets, cfg.to_data_config(), use_cuda=False)


def _state_fingerprint(model: torch.nn.Module) -> list[torch.Tensor]:
    return [v.detach().clone() for v in model.state_dict().values()]


# ------------------------------------------------------------------ 1, 2, 10: architecture
def test_mobilenetv2_output_shape_29_classes_cpu():
    model = build_student(29, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 29)
    assert out.device.type == "cpu"


def test_classifier_head_has_29_outputs_and_is_torchvision_mobilenetv2():
    model = build_student(29, pretrained=False)
    head = model.classifier[-1]
    assert isinstance(head, torch.nn.Linear)
    assert (head.in_features, head.out_features) == (1280, 29)
    assert count_parameters(model)["trainable"] == count_parameters(model)["total"]


def test_student_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        build_student(1, pretrained=False)
    with pytest.raises(ValueError):
        build_student(29, pretrained=False, dropout=1.0)


def test_failed_pretrained_download_raises_instead_of_falling_back(monkeypatch):
    from torchvision import models

    def boom(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr(models, "mobilenet_v2", boom)
    with pytest.raises(RuntimeError, match="ImageNet MobileNetV2"):
        build_student(29, pretrained=True)


# ------------------------------------------------------------------ 6, 7, 8: KD loss
def test_kd_loss_matches_reference_formula():
    torch.manual_seed(1)
    zs, zt = torch.randn(5, 29), torch.randn(5, 29)
    y = torch.randint(0, 29, (5,))
    T, alpha = 3.0, 0.3
    pt, ps = torch.softmax(zt / T, 1), torch.softmax(zs / T, 1)
    expected_kd = T ** 2 * (pt * (pt.log() - ps.log())).sum(1).mean()   # batch-mean KL, T^2 scaled
    out = DistillationLoss(T, alpha)(zs, zt, y)
    assert torch.allclose(out.kd, expected_kd, atol=1e-5)
    assert torch.allclose(out.ce, F.cross_entropy(zs, y), atol=1e-6)
    assert torch.allclose(out.total, alpha * out.ce + (1 - alpha) * out.kd, atol=1e-6)


def test_kd_loss_returns_finite_scalar():
    out = DistillationLoss(4.0, 0.5)(torch.randn(4, 29), torch.randn(4, 29), torch.randint(0, 29, (4,)))
    for t in out:
        assert t.ndim == 0 and torch.isfinite(t)


def test_kd_loss_is_zero_when_student_equals_teacher():
    z = torch.randn(4, 29)
    assert kd_divergence(z, z.clone(), 2.0).item() == pytest.approx(0.0, abs=1e-6)


def test_kd_loss_changes_with_alpha_and_temperature():
    torch.manual_seed(2)
    zs, zt = torch.randn(6, 29), torch.randn(6, 29) * 3
    y = torch.randint(0, 29, (6,))
    ce_only = DistillationLoss(4.0, 1.0)(zs, zt, y)
    kd_only = DistillationLoss(4.0, 0.0)(zs, zt, y)
    mid = DistillationLoss(4.0, 0.5)(zs, zt, y)
    assert torch.allclose(ce_only.total, ce_only.ce)        # alpha=1 -> pure CE
    assert torch.allclose(kd_only.total, kd_only.kd)        # alpha=0 -> pure KD
    assert mid.total.item() == pytest.approx(0.5 * (ce_only.total + kd_only.total).item(), rel=1e-5)
    assert kd_divergence(zs, zt, 1.0).item() != pytest.approx(kd_divergence(zs, zt, 8.0).item(), rel=1e-3)
    assert ce_only.ce.item() == pytest.approx(DistillationLoss(8.0, 0.5)(zs, zt, y).ce.item())  # CE ignores T


@pytest.mark.parametrize("temperature, alpha", [
    (0.0, 0.5), (-1.0, 0.5), (float("nan"), 0.5), (float("inf"), 0.5),
    (4.0, -0.01), (4.0, 1.01), (4.0, float("nan")), (4.0, float("inf")), ("4", 0.5), (4.0, None), (True, 0.5)])
def test_invalid_temperature_or_alpha_rejected(temperature, alpha):
    with pytest.raises(ValueError):
        validate_kd_hyperparameters(temperature, alpha)
    with pytest.raises(ValueError):
        DistillationLoss(temperature, alpha)


def test_loss_validates_shapes_and_label_smoothing():
    loss = DistillationLoss(2.0, 0.5)
    with pytest.raises(ValueError):
        loss(torch.randn(3, 29), torch.randn(3, 10), torch.zeros(3, dtype=torch.long))
    with pytest.raises(ValueError):
        loss(torch.randn(3, 29), torch.randn(3, 29), torch.zeros(2, dtype=torch.long))
    with pytest.raises(ValueError):
        DistillationLoss(2.0, 0.5, label_smoothing=1.0)


def test_teacher_logits_receive_no_gradient_through_loss():
    zs = torch.randn(3, 29, requires_grad=True)
    zt = torch.randn(3, 29, requires_grad=True)
    DistillationLoss(4.0, 0.5)(zs, zt, torch.zeros(3, dtype=torch.long)).total.backward()
    assert zs.grad is not None and zt.grad is None


# ------------------------------------------------------------------ 3, 4: frozen teacher
def test_teacher_is_frozen_and_eval_and_stays_eval(synthetic, teacher_ckpt):
    cfg, _ = synthetic
    teacher, info = load_frozen_teacher(teacher_ckpt, "cpu", list(ALL_CLASSES), C2I, SIZE)
    assert not teacher.training
    assert all(not p.requires_grad for p in teacher.parameters())
    assert info["sha256"] == file_sha256(teacher_ckpt) and info["architecture"] == "resnet50"
    teacher.train()                                    # a stray .train() must not leak into KD forward
    with torch.enable_grad():
        out = teacher_logits(teacher, torch.randn(2, 3, SIZE, SIZE))
    assert not teacher.training
    assert out.shape == (2, 29) and not out.requires_grad


def test_teacher_batchnorm_statistics_unchanged_by_teacher_forward(teacher_ckpt):
    teacher, _ = load_frozen_teacher(teacher_ckpt, "cpu", list(ALL_CLASSES), C2I, SIZE)
    before = _state_fingerprint(teacher)
    teacher.train()
    teacher_logits(teacher, torch.randn(4, 3, SIZE, SIZE))
    assert all(torch.equal(a, b) for a, b in zip(before, _state_fingerprint(teacher)))


def test_missing_or_foreign_teacher_checkpoint_raises(tmp_path, teacher_ckpt):
    with pytest.raises(FileNotFoundError, match="Teacher checkpoint not found"):
        load_frozen_teacher(tmp_path / "nope.pth", "cpu", list(ALL_CLASSES), C2I, SIZE)
    student_ckpt = tmp_path / "s.pth"
    save_student_checkpoint(student_ckpt, build_student(29, pretrained=False), epoch=1, best_val_metric=1.0,
                            best_metric_name="val_loss", class_names=list(ALL_CLASSES), class_to_idx=C2I,
                            config={"image_size": SIZE}, distillation={}, teacher={})
    with pytest.raises(ValueError, match="architecture"):      # a student file is not a teacher
        load_frozen_teacher(student_ckpt, "cpu", list(ALL_CLASSES), C2I, SIZE)


# ------------------------------------------------------------------ 12: compatibility
def test_teacher_student_output_dimensions_and_class_mapping_compatible(teacher_ckpt):
    teacher, ckpt = load_teacher(teacher_ckpt)
    student = build_student(len(ckpt["class_names"]), pretrained=False).eval()
    x = torch.randn(2, 3, SIZE, SIZE)
    assert teacher(x).shape == student(x).shape == (2, 29)
    verify_teacher_compatibility(teacher, ckpt, list(ALL_CLASSES), C2I, SIZE)   # does not raise


def test_incompatible_teacher_rejected(teacher_ckpt):
    teacher, ckpt = load_teacher(teacher_ckpt)
    swapped = list(ALL_CLASSES)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(TeacherCompatibilityError, match="class list"):
        verify_teacher_compatibility(teacher, ckpt, swapped, {c: i for i, c in enumerate(swapped)}, SIZE)
    with pytest.raises(TeacherCompatibilityError, match="class_to_idx"):
        verify_teacher_compatibility(teacher, ckpt, list(ALL_CLASSES), {**C2I, "A": 5}, SIZE)
    with pytest.raises(TeacherCompatibilityError, match="image size"):
        verify_teacher_compatibility(teacher, ckpt, list(ALL_CLASSES), C2I, 224)
    wrong_head = {**ckpt, "class_names": list(ALL_CLASSES)[:10], "class_to_idx": {c: i for i, c in enumerate(ALL_CLASSES[:10])}}
    with pytest.raises(TeacherCompatibilityError, match="outputs 29 classes"):
        verify_teacher_compatibility(teacher, wrong_head, list(ALL_CLASSES)[:10], wrong_head["class_to_idx"], SIZE)
    bad_norm = {**ckpt, "normalization": {"mean": [0.5] * 3, "std": [0.5] * 3}}
    with pytest.raises(TeacherCompatibilityError, match="normalisation"):
        verify_teacher_compatibility(teacher, bad_norm, list(ALL_CLASSES), C2I, SIZE)


# ------------------------------------------------------------------ 5, 11: training step
def test_student_gets_gradients_teacher_does_not(synthetic, teacher_ckpt):
    teacher, _ = load_frozen_teacher(teacher_ckpt, "cpu", list(ALL_CLASSES), C2I, SIZE)
    student = build_student(29, pretrained=False).train()
    x, y = torch.randn(4, 3, SIZE, SIZE), torch.randint(0, 29, (4,))
    out = DistillationLoss(4.0, 0.5)(student(x), teacher_logits(teacher, x), y)
    out.total.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters())
    assert any(p.grad.abs().sum() > 0 for p in student.parameters())
    assert all(p.grad is None for p in teacher.parameters())


def test_one_synthetic_training_step_updates_only_the_student(synthetic, teacher_ckpt):
    cfg, _ = synthetic
    teacher, _ = load_frozen_teacher(teacher_ckpt, "cpu", list(ALL_CLASSES), C2I, SIZE)
    student = build_student(29, pretrained=False)
    teacher_before, student_before = _state_fingerprint(teacher), _state_fingerprint(student)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-2)
    stats = train_epoch(student, teacher, _loaders(cfg)["train"], DistillationLoss(4.0, 0.5), optimizer,
                        torch.device("cpu"), max_batches=1, log_every=0)
    assert all(math.isfinite(v) for v in stats.values())
    assert all(torch.equal(a, b) for a, b in zip(teacher_before, _state_fingerprint(teacher)))
    assert any(not torch.equal(a, b) for a, b in zip(student_before, _state_fingerprint(student)))


def test_fit_writes_best_and_last_checkpoints_without_touching_teacher(synthetic, teacher_ckpt):
    cfg, _ = synthetic
    teacher_hash = file_sha256(teacher_ckpt)
    class_names = list(ALL_CLASSES)
    teacher, info = load_frozen_teacher(teacher_ckpt, "cpu", class_names, C2I, SIZE)
    student = build_student(29, pretrained=False)
    summary = fit(student, teacher, _loaders(cfg), cfg, torch.device("cpu"), class_names, C2I, info)
    best, last = cfg.checkpoint_path, Path(summary["last_checkpoint"])
    assert best.name == DEFAULT_CHECKPOINT_NAME and best.is_file() and last.is_file()
    assert last.name == "student_mobilenetv2_kd_last.pth"
    assert summary["epochs_run"] >= 1 and len(summary["history"]) == summary["epochs_run"]
    assert {"train_ce", "train_kd", "val_loss", "val_accuracy", "teacher_train_accuracy"} <= set(summary["history"][0])
    assert file_sha256(teacher_ckpt) == teacher_hash          # teacher checkpoint byte-identical
    assert not teacher.training and all(not p.requires_grad for p in teacher.parameters())


# ------------------------------------------------------------------ 9: checkpoint round trip
def test_student_checkpoint_save_and_load_roundtrip(tmp_path):
    torch.manual_seed(3)
    student = build_student(29, pretrained=False, dropout=0.3).eval()
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3)
    path = save_student_checkpoint(
        tmp_path / "sub" / "student.pth", student, epoch=4, best_val_metric=0.25, best_metric_name="val_loss",
        class_names=list(ALL_CLASSES), class_to_idx=C2I, config={"image_size": SIZE, "dropout": 0.3},
        distillation={"temperature": 4.0, "alpha": 0.5, "label_smoothing": 0.0},
        teacher={"sha256": "abc", "file_name": "t.pth"}, optimizer=opt, history=[{"epoch": 4}])
    assert not path.with_suffix(".pth.tmp").exists()
    ckpt = read_student_checkpoint(path)
    assert (ckpt["architecture"], ckpt["num_classes"], ckpt["epoch"], ckpt["input_size"]) == ("mobilenet_v2", 29, 4, SIZE)
    assert ckpt["class_names"] == list(ALL_CLASSES) and ckpt["class_to_idx"] == C2I
    assert ckpt["distillation"]["temperature"] == 4.0 and ckpt["teacher"]["sha256"] == "abc"
    assert "optimizer_state_dict" in ckpt and ckpt["best_metric_name"] == "val_loss"
    loaded, _ = load_student(path)
    assert not loaded.training and loaded.classifier[0].p == pytest.approx(0.3)
    x = torch.randn(2, 3, SIZE, SIZE)
    with torch.no_grad():
        assert torch.allclose(student(x), loaded(x), atol=1e-6)


def test_student_and_teacher_checkpoints_are_not_interchangeable(tmp_path, teacher_ckpt):
    with pytest.raises(ValueError, match="architecture"):
        read_student_checkpoint(teacher_ckpt)
    student_path = save_student_checkpoint(
        tmp_path / "s.pth", build_student(29, pretrained=False), epoch=1, best_val_metric=1.0,
        best_metric_name="val_loss", class_names=list(ALL_CLASSES), class_to_idx=C2I,
        config={"image_size": SIZE}, distillation={}, teacher={})
    with pytest.raises(ValueError, match="architecture"):
        read_checkpoint(student_path)
    with pytest.raises(FileNotFoundError):
        read_student_checkpoint(tmp_path / "missing.pth")


# ------------------------------------------------------------------ resume / config / CLI / end to end
def test_resume_continues_from_last_checkpoint_and_rejects_other_teacher(synthetic, teacher_ckpt, tmp_path):
    import dataclasses
    cfg, _ = synthetic
    cfg1 = dataclasses.replace(cfg, epochs=1)
    class_names = list(ALL_CLASSES)
    teacher, info = load_frozen_teacher(teacher_ckpt, "cpu", class_names, C2I, SIZE)
    s1 = fit(build_student(29, pretrained=False), teacher, _loaders(cfg1), cfg1, torch.device("cpu"),
             class_names, C2I, info)
    cfg2 = dataclasses.replace(cfg, epochs=2, resume=Path(s1["last_checkpoint"]))
    s2 = fit(build_student(29, pretrained=False), teacher, _loaders(cfg2), cfg2, torch.device("cpu"),
             class_names, C2I, info)
    assert [r["epoch"] for r in s2["history"]] == [1, 2]
    with pytest.raises(ValueError, match="different teacher"):
        fit(build_student(29, pretrained=False), teacher, _loaders(cfg2), cfg2, torch.device("cpu"),
            class_names, C2I, {**info, "sha256": "0" * 64})


def test_config_validation_and_paths():
    cfg = StudentConfig()
    cfg.validate()
    assert cfg.checkpoint_path.name == "student_mobilenetv2_kd_best.pth"
    assert cfg.checkpoint_path.name not in ("teacher_resnet50_best.pth", "student_best_aug.pth")
    for bad in (dict(temperature=0), dict(alpha=2), dict(epochs=0), dict(optimizer="adam"),
                dict(image_size=8), dict(dropout=1.5), dict(monitor="f1")):
        with pytest.raises(ValueError):
            StudentConfig(**bad).validate()
    assert isinstance(cfg.to_data_config(), TeacherConfig)
    assert all(isinstance(v, (str, int, float, bool, type(None))) for v in cfg.to_dict().values())


def test_cli_parser_builds_valid_config():
    args = student_train.build_parser().parse_args(
        ["--temperature", "2", "--alpha", "0.7", "--epochs", "3", "--no-pretrained", "--device", "cpu"])
    cfg = student_train.config_from_args(args)
    assert (cfg.temperature, cfg.alpha, cfg.epochs, cfg.pretrained) == (2.0, 0.7, 3, False)
    with pytest.raises(ValueError):
        student_train.config_from_args(student_train.build_parser().parse_args(["--alpha", "1.5"]))


def test_main_returns_error_code_for_missing_teacher(synthetic, tmp_path):
    cfg, processed = synthetic
    code = student_train.main(["--processed-dir", str(processed), "--teacher-checkpoint",
                               str(tmp_path / "missing.pth"), "--output-dir", str(tmp_path / "o"),
                               "--no-pretrained", "--device", "cpu", "--num-workers", "0"])
    assert code == 1
    assert not (tmp_path / "o").exists()          # failed before writing anything


def test_run_refuses_to_overwrite_teacher_or_existing_student(synthetic, teacher_ckpt):
    import dataclasses
    cfg, _ = synthetic
    clash = dataclasses.replace(cfg, output_dir=teacher_ckpt.parent, checkpoint_name=teacher_ckpt.name)
    with pytest.raises(ValueError, match="collides with the teacher"):
        student_train.run(clash)
    cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint_path.write_bytes(b"x")
    with pytest.raises(ValueError, match="already exists"):
        student_train.run(cfg)


def test_end_to_end_run_ignores_external_test_and_writes_reports(synthetic, teacher_ckpt):
    cfg, processed = synthetic
    # A poisoned external manifest must never be read by training or evaluation.
    (processed / "manifests" / "external_test.csv").write_text("garbage,not,a,manifest\n", encoding="utf-8")
    teacher_hash = file_sha256(teacher_ckpt)
    results = student_train.run(cfg)
    reports = cfg.reports_dir
    for split in ("validation", "test"):
        for name in (f"{split}_metrics.json", f"{split}_classification_report.txt",
                     f"{split}_confusion_matrix.png", f"teacher_reference_{split}_metrics.json"):
            assert (reports / name).is_file(), name
        assert 0.0 <= results[split]["student_accuracy"] <= 1.0
    for name in ("training_history.json", "run_config.json", "final_summary.json"):
        assert (reports / name).is_file()
    import json
    metrics = json.loads((reports / "test_metrics.json").read_text(encoding="utf-8"))
    assert len(metrics["confusion_matrix"]) == 29 and metrics["num_samples"] == 29
    assert metrics["model_footprint"]["total"] == count_parameters(build_student(29, pretrained=False))["total"]
    assert file_sha256(teacher_ckpt) == teacher_hash
    model, ckpt = load_student(cfg.checkpoint_path)     # best checkpoint usable for later inference
    assert ckpt["teacher"]["sha256"] == teacher_hash and not model.training
