"""Tests for the Partition 2 teacher model. Synthetic data only; no ImageNet download, no real dataset."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from preprocessing.config import ALL_CLASSES
from teacher_model.augmentation import (EvalTransform, RandomScipyRotation, TrainTransform,
                                        rotate_image_array)
from teacher_model.checkpoint import load_teacher, read_checkpoint, save_checkpoint
from teacher_model.config import TeacherConfig
from teacher_model.dataset import (ASLManifestDataset, DatasetError, build_dataloaders,
                                   build_datasets, read_manifest)
from teacher_model.metrics import compute_metrics
from teacher_model.model import build_teacher
from teacher_model.trainer import EarlyStopping, fit
from teacher_model.utils import write_json

FIELDS = ["image_path", "class_name", "class_index", "sha256", "dhash",
          "group_id", "group_size", "split", "width", "height"]
SIZE = 32


def _make_image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(base).resize((SIZE, SIZE), Image.Resampling.BICUBIC).save(path, quality=95)


@pytest.fixture()
def synthetic(tmp_path: Path):
    """Class root with 29 classes x 3 images (1 per split) + manifests/labels/summary in processed/."""
    class_root = tmp_path / "raw" / "train"
    processed = tmp_path / "processed"
    splits = ("train", "validation", "test")
    rows = {s: [] for s in splits}
    seed = 0
    for idx, cls in enumerate(ALL_CLASSES):
        for k, split in enumerate(splits):
            seed += 1
            rel = f"{cls}/{cls}{k}.jpg"
            _make_image(class_root / rel, seed)
            rows[split].append({"image_path": rel, "class_name": cls, "class_index": idx,
                                "sha256": f"sha{seed}", "dhash": "0", "group_id": f"g{seed}",
                                "group_size": 1, "split": split, "width": SIZE, "height": SIZE})
    for split in splits:
        path = processed / "manifests" / f"{split}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
            w.writeheader(); w.writerows(rows[split])
    write_json(processed / "metadata" / "labels.json", {
        "classes": list(ALL_CLASSES), "num_classes": 29,
        "class_to_idx": {c: i for i, c in enumerate(ALL_CLASSES)}})
    write_json(processed / "metadata" / "dataset_summary.json", {"class_root": str(class_root)})
    cfg = TeacherConfig(processed_dir=processed, image_size=SIZE, batch_size=8, num_workers=0,
                        epochs=2, pretrained=False, device="cpu", output_dir=tmp_path / "out",
                        early_stopping_patience=2)
    return cfg, rows, class_root


# ------------------------------------------------------------------ model
def test_resnet50_output_shape_29_classes_cpu():
    model = build_teacher(29, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 29)
    assert out.device.type == "cpu"


# ------------------------------------------------------------ augmentation
@pytest.mark.parametrize("angle", [15.0, -15.0, 8.0, -12.0, 0.0])
def test_rotation_preserves_shape_dtype_both_signs(angle):
    img = np.random.default_rng(0).integers(0, 256, (40, 40, 3), dtype=np.uint8)
    out = rotate_image_array(img, angle)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_rotation_positive_and_negative_differ_and_rotate_spatially():
    img = np.zeros((41, 41, 3), dtype=np.uint8)
    img[5:10, 20] = 255  # a short vertical bar near the top centre
    pos, neg = rotate_image_array(img, 30.0, fill=0), rotate_image_array(img, -30.0, fill=0)
    assert not np.array_equal(pos, neg)
    # mirror symmetry: +30 and -30 bars land on opposite sides of the vertical centre line
    xs_pos = np.argwhere(pos[..., 0] > 100)[:, 1].mean()
    xs_neg = np.argwhere(neg[..., 0] > 100)[:, 1].mean()
    assert (xs_pos - 20) * (xs_neg - 20) < 0
    # channels are not mixed: identical channels stay identical
    assert np.array_equal(pos[..., 0], pos[..., 1])


def test_random_rotation_samples_both_signs_within_range():
    rot = RandomScipyRotation(15.0, seed=1)
    angles = [rot.sample_angle() for _ in range(300)]
    assert min(angles) < 0 < max(angles)
    assert all(-15.0 <= a <= 15.0 for a in angles)
    assert RandomScipyRotation(0.0).sample_angle() == 0.0


def test_train_transform_is_random_eval_transform_is_deterministic():
    img = Image.fromarray(np.random.default_rng(3).integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8))
    train = TrainTransform(SIZE, 25.0, seed=0)
    outs = [train(img) for _ in range(4)]
    assert any(not torch.equal(outs[0], o) for o in outs[1:])
    ev = EvalTransform(SIZE)
    assert torch.equal(ev(img), ev(img))
    assert outs[0].shape == (3, SIZE, SIZE) and outs[0].dtype == torch.float32
    assert torch.isfinite(outs[0]).all()


# ----------------------------------------------------------------- dataset
def test_datasets_labels_and_split_separation(synthetic):
    cfg, rows, _ = synthetic
    datasets, labels = build_datasets(cfg)
    assert {s: len(d) for s, d in datasets.items()} == {"train": 29, "validation": 29, "test": 29}
    for split, ds in datasets.items():
        paths = [r["image_path"] for r in ds.rows]
        assert paths == [r["image_path"] for r in rows[split]]
        assert all(r["split"] == split for r in ds.rows)
    all_paths = [r["image_path"] for d in datasets.values() for r in d.rows]
    assert len(all_paths) == len(set(all_paths))
    _, y = datasets["train"][ALL_CLASSES.index("space")]
    assert y == labels["class_to_idx"]["space"] == 28


def test_augmentation_only_on_training_dataset(synthetic):
    cfg, _, _ = synthetic
    datasets, _ = build_datasets(cfg)
    assert datasets["train"].transform.is_random is True
    assert datasets["validation"].transform.is_random is False
    assert datasets["test"].transform.is_random is False
    a, _ = datasets["validation"][0]; b, _ = datasets["validation"][0]
    assert torch.equal(a, b)
    c, _ = datasets["test"][3]; d, _ = datasets["test"][3]
    assert torch.equal(c, d)


def test_leakage_across_splits_is_rejected(synthetic):
    cfg, rows, _ = synthetic
    path = cfg.manifests / "validation.csv"
    text = path.read_text(encoding="utf-8").splitlines()
    first_train = rows["train"][0]
    # replace validation row 1's sha256 with the one from a train row
    header, first = text[0], text[1].replace(rows["validation"][0]["sha256"], first_train["sha256"])
    path.write_text("\n".join([header, first] + text[2:]) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="leakage"):
        build_datasets(cfg)


def test_manifest_validation_errors(synthetic):
    cfg, _, _ = synthetic
    with pytest.raises(DatasetError, match="not found"):
        read_manifest(cfg.manifests / "nope.csv", "train", {})
    bad = cfg.manifests / "train.csv"
    bad.write_text(bad.read_text(encoding="utf-8").replace(",A,0,", ",A,5,", 1), encoding="utf-8")
    with pytest.raises(DatasetError, match="class_index"):
        build_datasets(cfg)


def test_missing_and_unreadable_images_raise_clear_errors(synthetic):
    cfg, rows, class_root = synthetic
    (class_root / rows["train"][0]["image_path"]).unlink()
    with pytest.raises(DatasetError, match="missing"):
        build_datasets(cfg)
    ds = ASLManifestDataset(rows["test"], class_root, EvalTransform(SIZE), "test", check_files=False)
    (class_root / rows["test"][1]["image_path"]).write_bytes(b"not an image")
    with pytest.raises(DatasetError, match="Unreadable"):
        ds[1]


def test_dataloader_batches_are_valid(synthetic):
    cfg, _, _ = synthetic
    datasets, _ = build_datasets(cfg)
    loaders = build_dataloaders(datasets, cfg, use_cuda=False)
    images, targets = next(iter(loaders["train"]))
    assert images.shape == (8, 3, SIZE, SIZE) and images.dtype == torch.float32
    assert targets.shape == (8,) and targets.dtype == torch.int64
    assert 0 <= int(targets.min()) and int(targets.max()) < 29
    assert torch.isfinite(images).all()


def test_external_test_manifest_is_never_read(synthetic):
    cfg, _, _ = synthetic
    (cfg.manifests / "external_test.csv").write_text("garbage", encoding="utf-8")
    datasets, _ = build_datasets(cfg)  # must not touch / choke on the external manifest
    assert set(datasets) == {"train", "validation", "test"}


# -------------------------------------------------------------- checkpoint
def test_checkpoint_roundtrip_without_imagenet(tmp_path):
    model = build_teacher(29, pretrained=False)
    path = save_checkpoint(tmp_path / "t.pth", model, epoch=3, best_val_metric=0.12,
                           best_metric_name="val_loss", class_names=list(ALL_CLASSES),
                           class_to_idx={c: i for i, c in enumerate(ALL_CLASSES)},
                           config={"image_size": 224})
    ckpt = read_checkpoint(path)
    assert ckpt["architecture"] == "resnet50" and ckpt["num_classes"] == 29
    assert ckpt["epoch"] == 3 and ckpt["best_val_metric"] == pytest.approx(0.12)
    loaded, _ = load_teacher(path)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(model.eval()(x), loaded(x), atol=1e-6)


# ----------------------------------------------------------------- metrics
def test_metrics_values():
    names = ["A", "B", "C"]
    m = compute_metrics([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0], names)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["per_class"]["B"]["precision"] == pytest.approx(2 / 3)
    assert m["per_class"]["A"]["recall"] == pytest.approx(0.5)
    assert m["f1_macro"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3, abs=1e-6)
    assert m["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert "macro avg" in m["classification_report"]


def test_early_stopping_modes():
    es = EarlyStopping("min", patience=2, min_delta=0.0)
    assert es.step(1.0) and not es.step(1.1) and not es.should_stop
    assert not es.step(1.2) and es.should_stop
    es2 = EarlyStopping("max", patience=1, min_delta=0.0)
    assert es2.step(0.5) and es2.step(0.6) and not es2.step(0.6) and es2.should_stop


# ------------------------------------------------ tiny end-to-end smoke run
def test_fit_smoke_and_resume_on_cpu(synthetic):
    cfg, _, _ = synthetic
    datasets, labels = build_datasets(cfg)
    loaders = build_dataloaders(datasets, cfg, use_cuda=False)
    model = build_teacher(29, pretrained=False)
    names, c2i = list(labels["classes"]), dict(labels["class_to_idx"])
    summary = fit(model, loaders, cfg, torch.device("cpu"), names, c2i)
    assert summary["epochs_run"] == 2 and len(summary["history"]) == 2
    assert cfg.checkpoint_path.is_file()
    last = Path(summary["last_checkpoint"])
    assert last.is_file() and last != cfg.checkpoint_path
    assert "optimizer_state_dict" in read_checkpoint(last)
    import dataclasses
    cfg2 = dataclasses.replace(cfg, epochs=3, resume=last)
    summary2 = fit(build_teacher(29, pretrained=False), loaders, cfg2, torch.device("cpu"), names, c2i)
    assert summary2["epochs_run"] == 3 and summary2["history"][-1]["epoch"] == 3
