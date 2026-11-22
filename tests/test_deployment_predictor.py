"""Partition 4: Predictor, checkpoint loading, error handling and the single-image CLI.

Synthetic data only: no real checkpoint, dataset, GPU, camera or MediaPipe is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from deployment import infer
from deployment.predictor import (CheckpointError, CheckpointNotFoundError, InvalidImageError, Predictor,
                                  checkpoint_summary, default_checkpoint, open_image)
from preprocessing.config import ALL_CLASSES
from student_model.checkpoint import save_student_checkpoint
from student_model.model import build_student
from teacher_model.augmentation import EvalTransform
from teacher_model.checkpoint import save_checkpoint as save_teacher_checkpoint
from teacher_model.model import build_teacher

SIZE = 32
C2I = {c: i for i, c in enumerate(ALL_CLASSES)}


def _save_student(path: Path, size: int = SIZE) -> Path:
    torch.manual_seed(0)
    return save_student_checkpoint(
        path, build_student(29, pretrained=False), epoch=7, best_val_metric=0.1, best_metric_name="val_loss",
        class_names=list(ALL_CLASSES), class_to_idx=C2I, config={"image_size": size, "pretrained": False},
        distillation={"temperature": 4.0, "alpha": 0.5}, teacher={})


@pytest.fixture(scope="module")
def student_ckpt(tmp_path_factory) -> Path:
    return _save_student(tmp_path_factory.mktemp("stu") / "student_mobilenetv2_kd_best.pth")


@pytest.fixture(scope="module")
def teacher_ckpt(tmp_path_factory) -> Path:
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tea") / "teacher_resnet50_best.pth"
    save_teacher_checkpoint(path, build_teacher(29, pretrained=False), epoch=3, best_val_metric=0.5,
                            best_metric_name="val_loss", class_names=list(ALL_CLASSES), class_to_idx=C2I,
                            config={"image_size": SIZE})
    return path


@pytest.fixture()
def image_file(tmp_path: Path) -> Path:
    arr = np.random.default_rng(1).integers(0, 256, (50, 70, 3), dtype=np.uint8)
    path = tmp_path / "hand.png"
    Image.fromarray(arr).save(path)
    return path


# ------------------------------------------------------------------ loading
def test_student_checkpoint_roundtrip_29_classes(student_ckpt, image_file):
    pred = Predictor.from_checkpoint(student_ckpt, "student", "cpu")
    assert pred.class_names == list(ALL_CLASSES) and len(pred.class_names) == 29
    assert {"del", "nothing", "space"} <= set(pred.class_names)
    out = pred.predict_path(image_file, top_k=3)
    assert out.label in ALL_CLASSES and out.label == pred.class_names[out.index]
    assert 0.0 <= out.confidence <= 1.0 and len(out.top_k) == 3
    assert sum(out.probabilities) == pytest.approx(1.0, abs=1e-5)
    assert out.top_k[0][0] == out.label and out.top_k[0][1] == pytest.approx(out.confidence)


def test_teacher_checkpoint_loads_through_teacher_loader(teacher_ckpt, image_file):
    pred = Predictor.from_checkpoint(teacher_ckpt, "teacher", "cpu")
    assert len(pred.class_names) == 29
    assert pred.predict_path(image_file).label in ALL_CLASSES


def test_input_size_comes_from_checkpoint(tmp_path):
    path = _save_student(tmp_path / "s.pth", size=48)
    assert Predictor.from_checkpoint(path, "student", "cpu").input_size == 48


def test_preprocessing_is_exactly_eval_transform(student_ckpt, image_file):
    pred = Predictor.from_checkpoint(student_ckpt, "student", "cpu")
    img = open_image(image_file)
    expected = EvalTransform(SIZE)(img)
    assert isinstance(pred.transform, EvalTransform)
    assert torch.equal(pred._to_batch([img])[0], expected)


def test_inference_is_deterministic_and_eval_mode(student_ckpt, image_file):
    pred = Predictor.from_checkpoint(student_ckpt, "student", "cpu")
    assert not pred.model.training
    a, b = pred.predict_path(image_file), pred.predict_path(image_file)
    assert a.probabilities == b.probabilities


def test_checkpoint_summary_reports_only_stored_metadata(student_ckpt):
    pred = Predictor.from_checkpoint(student_ckpt, "student", "cpu")
    info = checkpoint_summary(pred.checkpoint, student_ckpt)
    assert info["architecture"] == "mobilenet_v2" and info["epoch"] == 7
    assert info["pretrained_flag_in_config"] is False and info["file_megabytes"] > 0


def test_batch_prediction_matches_single(student_ckpt, image_file):
    pred = Predictor.from_checkpoint(student_ckpt, "student", "cpu")
    img = open_image(image_file)
    batch = pred.predict_images([img, img])
    assert len(batch) == 2 and batch[0].label == pred.predict_image(img).label
    assert pred.predict_images([]) == []


# ------------------------------------------------------------------ checkpoint errors
def test_missing_checkpoint_is_a_clear_error(tmp_path):
    with pytest.raises(CheckpointNotFoundError) as err:
        Predictor.from_checkpoint(tmp_path / "absent.pth", "student", "cpu")
    assert "absent.pth" in str(err.value) and "not stored in Git" in str(err.value)


@pytest.mark.parametrize("name", ["student_best_aug.pth", "teacher_best_aug.pth"])
def test_legacy_checkpoints_are_refused(tmp_path, name):
    legacy = tmp_path / name
    legacy.write_bytes(b"x")
    with pytest.raises(CheckpointError, match="legacy"):
        Predictor.from_checkpoint(legacy, "student" if name.startswith("student") else "teacher", "cpu")


def test_corrupt_checkpoint_file(tmp_path):
    bad = tmp_path / "bad.pth"
    bad.write_bytes(b"this is not a torch archive")
    with pytest.raises(CheckpointError):
        Predictor.from_checkpoint(bad, "student", "cpu")


def test_raw_state_dict_without_metadata_is_rejected(tmp_path):
    raw = tmp_path / "raw.pth"
    torch.save(build_student(29, pretrained=False).state_dict(), raw)
    with pytest.raises(CheckpointError):
        Predictor.from_checkpoint(raw, "student", "cpu")


def test_wrong_architecture_is_rejected(tmp_path, student_ckpt):
    ckpt = torch.load(student_ckpt, weights_only=True)
    ckpt["architecture"] = "resnet50"
    path = tmp_path / "wrong.pth"
    torch.save(ckpt, path)
    with pytest.raises(CheckpointError, match="architecture"):
        Predictor.from_checkpoint(path, "student", "cpu")


def test_normalization_mismatch_is_rejected(tmp_path, student_ckpt):
    ckpt = torch.load(student_ckpt, weights_only=True)
    ckpt["normalization"] = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
    path = tmp_path / "norm.pth"
    torch.save(ckpt, path)
    with pytest.raises(CheckpointError, match="normalization"):
        Predictor.from_checkpoint(path, "student", "cpu")


def test_model_output_class_mismatch_is_detected():
    pred = Predictor(nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 5)), ["a", "b", "c"], 8, "cpu")
    with pytest.raises(CheckpointError, match="does not match"):
        pred.predict_image(Image.new("RGB", (10, 10)))


def test_default_checkpoint_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ASL_STUDENT_CHECKPOINT", str(tmp_path / "x.pth"))
    monkeypatch.setenv("ASL_TEACHER_CHECKPOINT", str(tmp_path / "t.pth"))
    assert default_checkpoint("student") == tmp_path / "x.pth"
    assert default_checkpoint("teacher") == tmp_path / "t.pth"
    monkeypatch.delenv("ASL_STUDENT_CHECKPOINT")
    assert default_checkpoint("student").name == "student_mobilenetv2_kd_best.pth"
    with pytest.raises(ValueError):
        default_checkpoint("other")


# ------------------------------------------------------------------ image errors
def test_invalid_images_raise_invalid_image_error(tmp_path):
    with pytest.raises(InvalidImageError, match="not found"):
        open_image(tmp_path / "missing.png")
    fake = tmp_path / "fake.png"
    fake.write_text("definitely not a png")
    with pytest.raises(InvalidImageError, match="readable"):
        open_image(fake)
    with pytest.raises(InvalidImageError):
        open_image(tmp_path)          # a directory


def test_bad_arrays_are_rejected(student_ckpt):
    pred = Predictor.from_checkpoint(student_ckpt, "student", "cpu")
    for bad in (np.zeros((10, 10), np.uint8), np.zeros((10, 10, 4), np.uint8), np.zeros((0, 5, 3), np.uint8)):
        with pytest.raises(InvalidImageError):
            pred.predict_rgb_array(bad)


def test_grayscale_and_rgba_files_are_converted(student_ckpt, tmp_path):
    pred = Predictor.from_checkpoint(student_ckpt, "student", "cpu")
    for mode in ("L", "RGBA"):
        path = tmp_path / f"{mode}.png"
        Image.new(mode, (40, 40)).save(path)
        assert pred.predict_path(path).label in ALL_CLASSES


# ------------------------------------------------------------------ CLI
def test_cli_prints_prediction(student_ckpt, image_file, capsys):
    assert infer.main([str(image_file), "--checkpoint", str(student_ckpt), "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert str(image_file) in out and "top-3" in out


def test_cli_json_output(student_ckpt, image_file, capsys):
    assert infer.main([str(image_file), "--checkpoint", str(student_ckpt), "--device", "cpu", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["model"] == "student" and data["results"][0]["label"] in ALL_CLASSES


def test_cli_missing_checkpoint_returns_2_without_traceback(tmp_path, image_file, capsys):
    assert infer.main([str(image_file), "--checkpoint", str(tmp_path / "no.pth")]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_invalid_image_does_not_stop_other_images(student_ckpt, image_file, tmp_path, capsys):
    bad = tmp_path / "bad.jpg"
    bad.write_text("nope")
    code = infer.main([str(bad), str(image_file), "--checkpoint", str(student_ckpt), "--device", "cpu"])
    captured = capsys.readouterr()
    assert code == 1 and "error:" in captured.err and str(image_file) in captured.out


def test_cli_info_and_no_image(student_ckpt, capsys):
    assert infer.main(["--info", "--checkpoint", str(student_ckpt), "--device", "cpu"]) == 0
    assert json.loads(capsys.readouterr().out)["architecture"] == "mobilenet_v2"
    assert infer.main(["--checkpoint", str(student_ckpt), "--device", "cpu"]) == 2


def test_cli_uses_environment_default(student_ckpt, image_file, monkeypatch, capsys):
    monkeypatch.setenv("ASL_STUDENT_CHECKPOINT", str(student_ckpt))
    assert infer.main([str(image_file), "--device", "cpu"]) == 0
