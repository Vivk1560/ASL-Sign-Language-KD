import shutil

from PIL import Image

from preprocessing.data_cleaning import clean_records
from preprocessing.data_splitting import build_splits, write_manifests
from preprocessing.dataset_inspection import enumerate_records, inspect_dataset
from preprocessing.duplicate_detection import assign_duplicate_groups, compute_hashes
from preprocessing.external_test import build_external_test_manifest, external_manifest_path
from preprocessing.records import read_csv, read_json
from tests.conftest import class_root_of, random_image


def _test_dir(dataset):
    return dataset / "asl_alphabet_test" / "asl_alphabet_test"


def test_external_manifest_maps_labels_and_flags_dynamic_signs(cfg, dataset):
    d = _test_dir(dataset)
    for name, seed in (("J", 7001), ("Z", 7002), ("space", 7003)):
        random_image(seed).save(d / f"{name}_test.jpg")
    (d / "notes.txt").write_text("x")
    random_image(7004).save(d / "weird_name.jpg")
    rep26 = build_external_test_manifest(cfg, dataset)  # default 26-class set
    assert "space_test.jpg" in rep26["unmapped_files"]  # extra classes not in this class set
    cfg = cfg.with_overrides(class_set="29")
    rep = build_external_test_manifest(cfg, dataset)
    rows = {r["image_path"]: r for r in read_csv(external_manifest_path(cfg))}
    assert rows["J_test.jpg"]["class_name"] == "J" and rows["J_test.jpg"]["is_dynamic_sign"] == "True"
    assert rows["Z_test.jpg"]["is_dynamic_sign"] == "True"
    assert rows["A_test.jpg"]["is_dynamic_sign"] == "False"
    assert rows["space_test.jpg"]["class_name"] == "space"
    assert "weird_name.jpg" in rep["unmapped_files"]
    assert "del" not in {r["class_name"] for r in rows.values()}
    saved = read_json(cfg.reports_dir / "external_test_report.json")
    # Partition 2: discovery/mapping success is no longer reported as a generic "ok";
    # no split manifests exist here, so isolation was not checked.
    assert saved["found"] is True and saved["mapped"] == len(rows)
    assert saved["isolation_checked"] is False and saved["isolation_passed"] is None
    assert saved["status"] == "isolation_not_checked"


def test_external_images_never_enter_train_val_test_manifests(cfg, dataset):
    build_external_test_manifest(cfg, dataset)
    root = class_root_of(dataset)
    records, structure = enumerate_records(root)
    inspect_dataset(cfg, root, records, structure)
    clean_records(cfg, root, records)
    compute_hashes(cfg, root, records)
    assign_duplicate_groups(cfg, records)
    split = build_splits(cfg, records)
    write_manifests(cfg, split["rows"])
    split_paths = {r["image_path"] for rows in split["rows"].values() for r in rows}
    assert not any(p.endswith("_test.jpg") for p in split_paths)
    assert external_manifest_path(cfg).name not in {cfg.manifest_path(s).name for s in
                                                    ("train", "validation", "test")}


def test_external_overlap_with_training_is_reported(cfg, dataset):
    root = class_root_of(dataset)
    shutil.copy(_test_dir(dataset) / "A_test.jpg", root / "A" / "leaky.jpg")
    records, structure = enumerate_records(root)
    inspect_dataset(cfg, root, records, structure)
    clean_records(cfg, root, records)
    compute_hashes(cfg, root, records)
    assign_duplicate_groups(cfg, records)
    write_manifests(cfg, build_splits(cfg, records)["rows"])
    rep = build_external_test_manifest(cfg, dataset)
    assert "A_test.jpg" in rep["training_sha256_overlap"]


def test_missing_test_folder_is_reported_not_fatal(tmp_path, cfg):
    empty = tmp_path / "empty_raw"
    empty.mkdir()
    rep = build_external_test_manifest(cfg, empty)
    assert rep["found"] is False and rep["status"] == "test folder not found"
