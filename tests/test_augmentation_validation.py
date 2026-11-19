import csv
import shutil

import numpy as np
import pytest
from PIL import Image

from preprocessing.config import AugmentationConfig, SPLIT_NAMES
from preprocessing.data_augmentation import (TrainAugmentation, generate_offline_augmentations)
from preprocessing.data_cleaning import clean_records
from preprocessing.data_splitting import build_splits, read_manifest, write_manifests
from preprocessing.dataset_inspection import (compute_raw_fingerprint, enumerate_records,
                                              inspect_dataset)
from preprocessing.dataset_validation import ValidationError, validate_outputs
from preprocessing.duplicate_detection import assign_duplicate_groups, compute_hashes
from preprocessing.labels import load_labels, write_labels
from tests.conftest import class_root_of, random_image


def test_augmentation_is_reproducible_and_size_preserving():
    aug = TrainAugmentation(seed=5)
    img = random_image(3, 80)
    a = aug.for_sample(img, "A/A1.jpg|0")
    b = TrainAugmentation(seed=5).for_sample(img, "A/A1.jpg|0")
    assert a.size == img.size and a.mode == "RGB"
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(a), np.asarray(aug.for_sample(img, "other")))
    assert aug.signature()["horizontal_flip"] is False


def test_no_flip_in_geometry():
    """Identity-range config must return the image unchanged (proves no hidden flip)."""
    ident = AugmentationConfig(rotation_deg=0, scale_range=(1, 1), translate_frac=0,
                               brightness_range=(1, 1), contrast_range=(1, 1),
                               saturation_range=(1, 1), channel_gain=0, occlusion_prob=0)
    img = random_image(11, 64)
    out = TrainAugmentation(ident, seed=1).apply(img, np.random.default_rng(0))
    assert np.abs(np.asarray(out, dtype=int) - np.asarray(img, dtype=int)).max() <= 1


def test_occlusion_actually_modifies_pixels():
    """Regression for the ignored CoarseDropout args (Phase-1 finding #11)."""
    cfg = AugmentationConfig(rotation_deg=0, scale_range=(1, 1), translate_frac=0,
                             brightness_range=(1, 1), contrast_range=(1, 1),
                             saturation_range=(1, 1), channel_gain=0, occlusion_prob=1.0,
                             occlusion_area_frac=(0.05, 0.10), occlusion_max_holes=1)
    img = Image.new("RGB", (100, 100), (128, 128, 128))
    out = np.asarray(TrainAugmentation(cfg, seed=2).apply(img, np.random.default_rng(3)))
    changed = (np.abs(out.astype(int) - 128).sum(axis=2) > 0).mean()
    assert 0.02 <= changed <= 0.12


def _processed(cfg, dataset, offline=True):
    root = class_root_of(dataset)
    records, structure = enumerate_records(root)
    summary = inspect_dataset(cfg, root, records, structure)
    clean_records(cfg, root, records)
    compute_hashes(cfg, root, records)
    assign_duplicate_groups(cfg, records)
    write_labels(cfg)
    split = build_splits(cfg, records)
    write_manifests(cfg, split["rows"])
    aug_rows = generate_offline_augmentations(cfg, root, split["rows"]["train"]) if offline else []
    return root, summary, split, aug_rows


def _validate(cfg, root, summary, fp=None):
    return validate_outputs(cfg, root, load_labels(cfg.labels_path),
                            summary["unexpected_folders"], fp)


def test_offline_augmentation_train_only_with_source_links(cfg, dataset):
    root, summary, split, aug_rows = _processed(cfg, dataset)
    n_train = len(split["rows"]["train"])
    assert len(aug_rows) == n_train * cfg.augmentation.offline_copies_per_image
    train_paths = {r["image_path"] for r in split["rows"]["train"]}
    assert {r["source_path"] for r in aug_rows} <= train_paths
    held_out = {r["image_path"] for s in ("validation", "test") for r in split["rows"][s]}
    assert not ({r["source_path"] for r in aug_rows} & held_out)
    files = {p.relative_to(cfg.augmented_dir).as_posix() for p in cfg.augmented_dir.rglob("*.jpg")}
    assert files == {r["aug_path"] for r in aug_rows}
    assert all(f.startswith("train/") for f in files)
    with open(cfg.augmented_manifest_path, newline="") as fh:
        assert len(list(csv.DictReader(fh))) == len(aug_rows)
    assert _validate(cfg, root, summary).ok


def test_offline_augmentation_rejects_non_train_rows(cfg, dataset):
    root, _, split, _ = _processed(cfg, dataset, offline=False)
    with pytest.raises(ValueError):
        generate_offline_augmentations(cfg, root, split["rows"]["validation"])


def test_validation_passes_on_clean_outputs_and_raw_unchanged(cfg, dataset):
    root = class_root_of(dataset)
    fp = compute_raw_fingerprint(root)
    root, summary, _, _ = _processed(cfg, dataset, offline=False)
    rep = _validate(cfg, root, summary, fp)
    assert rep.ok, [c for c in rep.checks if not c.passed]
    rep.raise_if_critical()


def _rewrite(cfg, split, mutate):
    rows = read_manifest(cfg.manifest_path(split))
    mutate(rows)
    from preprocessing.data_splitting import MANIFEST_FIELDS
    from preprocessing.records import write_csv
    write_csv(cfg.manifest_path(split), MANIFEST_FIELDS, rows)


@pytest.mark.parametrize("scenario, expected", [
    ("leak_path", "no_path_overlap"),
    ("leak_group", "no_group_leakage"),
    ("bad_label", "labels_valid_and_consistent"),
    ("bad_index", "labels_valid_and_consistent"),
    ("missing_file", "files_exist"),
    ("empty_split", "splits_not_empty"),
    ("corrupt_file", "images_decodable"),
    ("aug_in_test", "no_augmentation_leakage"),
    ("orphan_aug", "no_augmentation_leakage"),
    ("raw_modified", "raw_dataset_unchanged"),
])
def test_validation_catches_failures(cfg, dataset, scenario, expected):
    root = class_root_of(dataset)
    fp = compute_raw_fingerprint(root)
    root, summary, split, _ = _processed(cfg, dataset, offline=(scenario == "aug_in_test"))
    if scenario == "leak_path":
        dup = dict(split["rows"]["train"][0], split="validation")
        _rewrite(cfg, "validation", lambda r: r.append(dup))
    elif scenario == "leak_group":
        t = split["rows"]["train"][0]
        extra = dict(split["rows"]["test"][0], group_id=t["group_id"])
        _rewrite(cfg, "test", lambda r: r.__setitem__(0, extra))
    elif scenario == "bad_label":
        _rewrite(cfg, "train", lambda r: r[0].__setitem__("class_name", "ZZ"))
    elif scenario == "bad_index":
        _rewrite(cfg, "train", lambda r: r[0].__setitem__("class_index", 99))
    elif scenario == "missing_file":
        (root / split["rows"]["train"][0]["image_path"]).unlink()
    elif scenario == "empty_split":
        _rewrite(cfg, "test", lambda r: r.clear())
    elif scenario == "corrupt_file":
        (root / split["rows"]["test"][0]["image_path"]).write_bytes(b"junk")
    elif scenario == "aug_in_test":
        aug_path = next(cfg.augmented_dir.rglob("*.jpg")).relative_to(cfg.augmented_dir).as_posix()
        _rewrite(cfg, "test", lambda r: r[0].__setitem__("image_path", "augmented/" + aug_path))
    elif scenario == "orphan_aug":
        orphan = cfg.augmented_dir / "validation" / "A" / "x.jpg"
        orphan.parent.mkdir(parents=True)
        random_image(1).save(orphan)
    elif scenario == "raw_modified":
        shutil.copy(root / "A" / "A1.jpg", root / "A" / "sneaky.jpg")
    rep = _validate(cfg, root, summary, fp)
    assert not rep.ok
    assert expected in [c.name for c in rep.critical_failures], [c.name for c in rep.critical_failures]
    with pytest.raises(ValidationError):
        rep.raise_if_critical()
