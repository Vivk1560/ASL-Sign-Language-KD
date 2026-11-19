from pathlib import Path

from PIL import Image

from preprocessing.data_cleaning import clean_records
from preprocessing.dataset_inspection import (compute_raw_fingerprint, discover_class_root,
                                              enumerate_records, inspect_dataset)
from preprocessing.download_dataset import ensure_dataset, extract_archive
from tests.conftest import class_root_of, random_image


def test_discovers_nested_class_root(dataset):
    assert discover_class_root(dataset) == class_root_of(dataset)
    assert discover_class_root(dataset / "nope") is None


def test_inspection_counts_and_excluded_classes(cfg, dataset):
    root = class_root_of(dataset)
    records, structure = enumerate_records(root)
    s = inspect_dataset(cfg, root, records, structure)
    assert s["total_files"] == 29 * 14
    assert s["per_class_supported_image_counts"]["A"] == 14
    assert s["excluded_classes"]["del"] == {
        "present_in_dataset": True, "supported_image_count": 14,
        "exclusion_reason": cfg.excluded_classes["del"]}
    assert s["missing_selected_classes"] == []
    assert s["dimensions"]["width"]["min"] == 64
    assert s["kaggle_test_folder"]["found"]
    assert all(len(v) <= cfg.samples_per_class for v in s["samples"].values())


def test_corrupt_unsupported_empty_and_bad_dims_detected(cfg, dataset):
    root = class_root_of(dataset)
    (root / "A" / "corrupt.jpg").write_bytes(b"not an image at all")
    (root / "A" / "empty.jpg").write_bytes(b"")
    (root / "A" / "notes.txt").write_text("hi")
    Image.new("RGB", (4, 4)).save(root / "B" / "tiny.png")
    (root / "C" / "trunc.jpg").write_bytes((root / "C" / "C1.jpg").read_bytes()[:200])
    records, structure = enumerate_records(root)
    inspect_dataset(cfg, root, records, structure)
    rep = clean_records(cfg, root, records)
    reasons = {e["image_path"]: e["reason"] for e in rep["excluded_files"]}
    assert reasons["A/corrupt.jpg"] == "corrupt_or_unreadable"
    assert reasons["A/empty.jpg"] == "empty_file"
    assert reasons["A/notes.txt"] == "unsupported_file_type"
    assert reasons["B/tiny.png"] == "invalid_dimensions"
    assert reasons["C/trunc.jpg"] == "corrupt_or_unreadable"
    assert rep["not_selected"] == 3 * 14  # del / nothing / space
    assert (root / "A" / "corrupt.jpg").exists()  # never deleted


def test_empty_class_folder_reported(cfg, tmp_path):
    from tests.conftest import build_dataset
    raw = tmp_path / "raw2"
    root = build_dataset(raw, per_class=4)
    for f in (root / "D").iterdir():
        f.unlink()
    records, structure = enumerate_records(root)
    rep = clean_records(cfg, root, records)
    assert "D" in rep["selected_classes_without_valid_images"]


def test_raw_fingerprint_detects_change(dataset):
    root = class_root_of(dataset)
    before = compute_raw_fingerprint(root)
    assert before == compute_raw_fingerprint(root)
    (root / "A" / "extra.jpg").write_bytes(b"x")
    assert before != compute_raw_fingerprint(root)


def test_ensure_dataset_missing_and_zip(cfg, tmp_path, dataset):
    missing = cfg.with_overrides(raw_dir=tmp_path / "empty_raw")
    st = ensure_dataset(missing)
    assert not st.available and st.source == "missing" and "kaggle.com" in st.message
    assert ensure_dataset(cfg).source == "existing"
    # zip extraction path
    import shutil
    archive = shutil.make_archive(str(tmp_path / "bundle"), "zip", dataset)
    target = tmp_path / "from_zip"
    target.mkdir()
    shutil.copy(archive, target / "archive.zip")
    st = ensure_dataset(cfg.with_overrides(raw_dir=target), extract_zip=True)
    assert st.available and st.source == "extracted"


def test_extract_zip_slip_guard(tmp_path):
    import zipfile
    import pytest
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with pytest.raises(ValueError):
        extract_archive(z, tmp_path / "out")
