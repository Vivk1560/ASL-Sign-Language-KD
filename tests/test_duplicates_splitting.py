import shutil
from collections import Counter

import numpy as np
from PIL import Image, ImageEnhance

from preprocessing.data_cleaning import clean_records
from preprocessing.data_splitting import (build_splits, read_manifest, write_manifests)
from preprocessing.dataset_inspection import enumerate_records, inspect_dataset
from preprocessing.duplicate_detection import (assign_duplicate_groups, compute_hashes,
                                               dhash_image, near_pairs)
from tests.conftest import build_dataset, class_root_of


def _prepare(cfg, root):
    records, structure = enumerate_records(root)
    inspect_dataset(cfg, root, records, structure)
    clean_records(cfg, root, records)
    compute_hashes(cfg, root, records)
    return records


def test_exact_and_near_duplicates_grouped(cfg, dataset):
    root = class_root_of(dataset)
    shutil.copy(root / "A" / "A1.jpg", root / "A" / "A1_copy.jpg")  # exact
    with Image.open(root / "B" / "B1.jpg") as im:  # near: brightness + re-encode
        ImageEnhance.Brightness(im.convert("RGB")).enhance(1.05).save(root / "B" / "B1_near.jpg", quality=80)
    records = _prepare(cfg, root)
    rep = assign_duplicate_groups(cfg, records)
    by_path = {r.rel_path: r for r in records}
    assert by_path["A/A1.jpg"].group_id == by_path["A/A1_copy.jpg"].group_id
    assert by_path["A/A1.jpg"].sha256 == by_path["A/A1_copy.jpg"].sha256
    assert by_path["B/B1.jpg"].group_id == by_path["B/B1_near.jpg"].group_id
    assert by_path["B/B1.jpg"].sha256 != by_path["B/B1_near.jpg"].sha256
    types = {g["class_name"]: g["type"] for g in rep["groups"]}
    assert types["A"] == "exact" and types["B"] == "near"
    assert rep["duplicate_groups"] == 2
    # unrelated images stay separate
    assert by_path["C/C1.jpg"].group_id != by_path["C/C2.jpg"].group_id


def test_cross_class_exact_duplicate_is_label_conflict(cfg, dataset):
    root = class_root_of(dataset)
    shutil.copy(root / "A" / "A2.jpg", root / "B" / "B_from_A.jpg")
    records = _prepare(cfg, root)
    rep = assign_duplicate_groups(cfg, records)
    assert {c["image_path"] for c in rep["label_conflicts"]} == {"A/A2.jpg", "B/B_from_A.jpg"}
    assert all(r.status == "excluded" for r in records
               if r.rel_path in ("A/A2.jpg", "B/B_from_A.jpg"))


def test_near_pairs_hamming_threshold():
    h = np.array([0, 0b1111, 0b11111111], dtype=np.uint64)
    assert list(near_pairs(h, 3)) == []
    assert list(near_pairs(h, 4)) == [(0, 1), (1, 2)]  # d(0,1)=4, d(1,2)=4, d(0,2)=8
    assert list(near_pairs(h, 8)) == [(0, 1), (0, 2), (1, 2)]


def _split(cfg, dataset, with_dups=False):
    root = class_root_of(dataset)
    if with_dups:
        for cls in ("A", "B"):
            for i in (1, 2, 3):
                shutil.copy(root / cls / f"{cls}{i}.jpg", root / cls / f"{cls}{i}_dup.jpg")
    records = _prepare(cfg, root)
    assign_duplicate_groups(cfg, records)
    return build_splits(cfg, records)


def test_split_is_reproducible_and_seed_sensitive(cfg, dataset):
    a, b = _split(cfg, dataset), _split(cfg, dataset)
    assert a["rows"] == b["rows"]
    c = _split(cfg.with_overrides(seed=1234), dataset)
    assert a["rows"]["train"] != c["rows"]["train"]


def test_split_no_overlap_stratified_and_only_selected_classes(cfg, dataset):
    s = _split(cfg, dataset)
    paths = [r["image_path"] for rows in s["rows"].values() for r in rows]
    assert len(paths) == len(set(paths)) == 26 * 14
    assert not s["issues"] or all(i["severity"] == "warning" for i in s["issues"])
    for cls, counts in s["per_class"].items():
        assert all(counts[k] > 0 for k in counts), cls
    assert {r["class_name"] for rows in s["rows"].values() for r in rows} == set(cfg.classes)


def test_group_aware_split_keeps_duplicates_together(cfg, dataset):
    s = _split(cfg, dataset, with_dups=True)
    owner = {}
    for split, rows in s["rows"].items():
        for r in rows:
            owner.setdefault(r["group_id"], set()).add(split)
    assert all(len(v) == 1 for v in owner.values())
    # duplicated images actually exist together in one split
    by_path = {r["image_path"]: r["split"] for rows in s["rows"].values() for r in rows}
    for i in (1, 2, 3):
        assert by_path[f"A/A{i}.jpg"] == by_path[f"A/A{i}_dup.jpg"]


def test_insufficient_groups_is_reported_not_hidden(tmp_path, cfg):
    raw = tmp_path / "tiny_raw"
    root = build_dataset(raw, per_class=2, classes=list(cfg.classes))
    records = _prepare(cfg.with_overrides(raw_dir=raw), root)
    assign_duplicate_groups(cfg, records)
    s = build_splits(cfg, records)
    crit = [i for i in s["issues"] if i["severity"] == "critical"]
    assert crit and all(i["code"] == "insufficient_independent_groups" for i in crit)


def test_manifest_roundtrip_and_columns(cfg, dataset):
    s = _split(cfg, dataset)
    write_manifests(cfg, s["rows"])
    rows = read_manifest(cfg.manifest_path("train"))
    assert rows == s["rows"]["train"] or [dict(r, width=int(r["width"]), height=int(r["height"])) for r in s["rows"]["train"]] == rows
    first = rows[0]
    assert set(first) == {"image_path", "class_name", "class_index", "sha256", "dhash",
                          "group_id", "group_size", "split", "width", "height"}
    assert first["split"] == "train" and len(first["sha256"]) == 64
    assert cfg.manifest_path("train").read_bytes() == cfg.manifest_path("train").read_bytes()


def test_dhash_stable_and_different():
    from tests.conftest import random_image
    assert dhash_image(random_image(1)) == dhash_image(random_image(1))
    assert dhash_image(random_image(1)) != dhash_image(random_image(2))
