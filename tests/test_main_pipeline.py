import hashlib
import json
from pathlib import Path

from preprocessing.main import (EXIT_FAILED, EXIT_NEEDS_FORCE, EXIT_NO_DATASET, EXIT_OK, main)
from preprocessing.records import read_json
from tests.conftest import build_dataset, class_root_of


def _tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _args(dataset, proc, *extra):
    return ["--raw-dir", str(dataset), "--processed-dir", str(proc), "--workers", "2", *extra]


EXPECTED_OUTPUTS = [
    "metadata/labels.json", "metadata/dataset_summary.json", "metadata/preprocessing_config.json",
    "manifests/train.csv", "manifests/validation.csv", "manifests/test.csv",
    "reports/class_distribution.png", "reports/sample_images.png", "reports/image_dimensions.png",
    "reports/dataset_summary.png", "reports/split_distribution.png",
    "reports/corrupt_image_report.json", "reports/duplicate_report.json",
    "reports/dataset_validation_report.json", "reports/split_report.json",
    "reports/preprocessing_report.md",
]


def test_end_to_end_outputs_raw_untouched_and_logs(tmp_path, dataset):
    proc = tmp_path / "proc"
    before = _tree_hash(dataset)
    assert main(_args(dataset, proc)) == EXIT_OK
    assert _tree_hash(dataset) == before  # byte-identical raw data
    for rel in EXPECTED_OUTPUTS:
        assert (proc / rel).is_file(), rel
    assert list((proc / "reports" / "logs").glob("*.log"))
    summary = read_json(proc / "metadata" / "dataset_summary.json")
    assert summary["validation_ok"] and summary["splits"]["sizes"]["train"] > 0
    assert summary["excluded_classes"]["nothing"]["supported_image_count"] == 14
    assert summary["class_handling"]["class_set"] == "26"
    assert not list(proc.glob("augmented/**/*.jpg"))  # on-the-fly is default
    labels = read_json(proc / "metadata" / "labels.json")
    import csv
    for s in ("train", "validation", "test"):
        for row in csv.DictReader(open(proc / "manifests" / f"{s}.csv", newline="")):
            assert labels["class_to_idx"][row["class_name"]] == int(row["class_index"])


def test_rerun_same_seed_gives_identical_manifests(tmp_path, dataset):
    p1, p2 = tmp_path / "p1", tmp_path / "p2"
    assert main(_args(dataset, p1)) == EXIT_OK
    assert main(_args(dataset, p2)) == EXIT_OK
    for s in ("train", "validation", "test"):
        assert (p1 / "manifests" / f"{s}.csv").read_bytes() == (p2 / "manifests" / f"{s}.csv").read_bytes()
    assert main(_args(dataset, p1)) == EXIT_OK  # in-place rerun is safe
    assert (p1 / "manifests/train.csv").read_bytes() == (p2 / "manifests/train.csv").read_bytes()


def test_changed_config_needs_force_and_archives(tmp_path, dataset):
    proc = tmp_path / "proc"
    assert main(_args(dataset, proc)) == EXIT_OK
    old = (proc / "manifests/train.csv").read_bytes()
    assert main(_args(dataset, proc, "--seed", "7")) == EXIT_NEEDS_FORCE
    assert (proc / "manifests/train.csv").read_bytes() == old  # untouched
    assert main(_args(dataset, proc, "--seed", "7", "--force")) == EXIT_OK
    archived = list((proc / "_archive").glob("*/manifests/train.csv"))
    assert archived and archived[0].read_bytes() == old
    assert (proc / "manifests/train.csv").read_bytes() != old


def test_offline_augment_via_cli(tmp_path, dataset):
    proc = tmp_path / "proc"
    assert main(_args(dataset, proc, "--offline-augment", "--augment-copies", "1")) == EXIT_OK
    aug = read_json(proc / "metadata/dataset_summary.json")["augmentation"]
    assert aug["mode"] == "offline" and aug["offline_images"] > 0
    assert not list(proc.glob("augmented/validation")) and not list(proc.glob("augmented/test"))


def test_29_class_set(tmp_path, dataset):
    proc = tmp_path / "proc"
    assert main(_args(dataset, proc, "--class-set", "29")) == EXIT_OK
    assert read_json(proc / "metadata/labels.json")["num_classes"] == 29


def test_missing_dataset_exit_code(tmp_path):
    assert main(_args(tmp_path / "none", tmp_path / "proc")) == EXIT_NO_DATASET
    assert not (tmp_path / "proc" / "manifests").exists()


def test_insufficient_data_stops_pipeline(tmp_path):
    raw = tmp_path / "raw"
    build_dataset(raw, per_class=2)
    assert main(_args(raw, tmp_path / "proc")) == EXIT_FAILED
    assert not (tmp_path / "proc" / "manifests" / "train.csv").exists()
    assert (tmp_path / "proc" / "reports" / "split_report.json").is_file()
