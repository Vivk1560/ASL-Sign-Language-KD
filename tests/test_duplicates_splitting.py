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


# ---------------------------------------------------------------- anchor grouping (Partition 1 correction)
from preprocessing.config import load_config  # noqa: E402
from preprocessing.duplicate_detection import anchor_groups  # noqa: E402
from preprocessing.records import STATUS_VALID, ImageRecord  # noqa: E402


def _rec(path, cls, sha, dhash):
    return ImageRecord(rel_path=path, class_folder=cls, class_name=cls, ext=".jpg", size_bytes=1,
                       width=200, height=200, header_ok=True, status=STATUS_VALID,
                       sha256=sha, dhash=f"{dhash:016x}")


def _mk(specs, tmp_path, **kw):
    cfg = load_config(raw_dir=tmp_path / "r", processed_dir=tmp_path / "p", workers=1, **kw)
    recs = [_rec(p, c, s, h) for p, c, s, h in specs]
    return cfg, recs


def test_chaining_is_bounded_a_b_c(tmp_path):
    # d(A,B)=4, d(B,C)=4, d(A,C)=8: single linkage would merge all three.
    a, b, c = 0, 0b1111, 0b11111111
    cfg, recs = _mk([("A/1.jpg", "A", "s1", a), ("A/2.jpg", "A", "s2", b),
                     ("A/3.jpg", "A", "s3", c)], tmp_path)
    assign_duplicate_groups(cfg, recs)
    g = {r.rel_path: r.group_id for r in recs}
    assert g["A/1.jpg"] == g["A/2.jpg"]      # A~B grouped
    assert g["A/3.jpg"] != g["A/1.jpg"]      # C not chained to A


def test_long_chain_does_not_form_one_group(tmp_path):
    # h_i = i low bits set: neighbours differ by 1 bit, so single linkage (threshold 4)
    # would merge all 40 into one group. Anchor grouping caps each group at 5 members.
    specs = [(f"A/{i:03d}.jpg", "A", f"s{i}", (1 << i) - 1) for i in range(40)]
    cfg, recs = _mk(specs, tmp_path)
    rep = assign_duplicate_groups(cfg, recs)
    assert rep["max_group_size"] <= 5
    assert rep["total_groups"] >= 8


def test_members_within_threshold_of_anchor(tmp_path):
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2 ** 63, size=300, dtype=np.uint64)
    anchor_of = anchor_groups(base, 4)
    for i, a in enumerate(anchor_of):
        assert bin(int(base[i]) ^ int(base[a])).count("1") <= 4


def test_identical_dhash_always_same_group_and_deterministic(tmp_path):
    specs = [(f"A/{i}.jpg", "A", f"s{i}", 0xABCDEF if i % 2 else 0x123456) for i in range(10)]
    cfg, recs = _mk(specs, tmp_path)
    assign_duplicate_groups(cfg, recs)
    first = [r.group_id for r in recs]
    cfg2, recs2 = _mk(specs, tmp_path)
    assign_duplicate_groups(cfg2, recs2)
    assert first == [r.group_id for r in recs2]
    assert len({r.group_id for r in recs if r.dhash.endswith("abcdef")}) == 1


def test_near_duplicates_never_merge_across_classes(tmp_path):
    cfg, recs = _mk([("A/1.jpg", "A", "s1", 0b1), ("B/1.jpg", "B", "s2", 0b1)], tmp_path)
    assign_duplicate_groups(cfg, recs)
    assert recs[0].group_id != recs[1].group_id


def test_exact_duplicates_stay_grouped_even_with_far_hash_noise(tmp_path):
    cfg, recs = _mk([("A/1.jpg", "A", "same", 0), ("A/2.jpg", "A", "same", 0),
                     ("A/3.jpg", "A", "other", 2 ** 40 - 1)], tmp_path)
    rep = assign_duplicate_groups(cfg, recs)
    assert recs[0].group_id == recs[1].group_id != recs[2].group_id
    assert rep["exact_duplicate_extra_copies"] == 1


def test_cross_class_exact_conflict_excluded_and_not_in_groups(tmp_path):
    cfg, recs = _mk([("A/1.jpg", "A", "dup", 5), ("B/1.jpg", "B", "dup", 5),
                     ("A/2.jpg", "A", "u", 999)], tmp_path)
    rep = assign_duplicate_groups(cfg, recs)
    assert len(rep["label_conflicts"]) == 2
    assert [r.status for r in recs] == ["excluded", "excluded", "valid"]
    assert rep["images_considered"] == 1


def test_empty_and_singleton_inputs(tmp_path):
    cfg, _ = _mk([], tmp_path)
    rep = assign_duplicate_groups(cfg, [])
    assert rep["images_considered"] == 0 and rep["max_group_size"] == 0
    cfg, recs = _mk([("A/1.jpg", "A", "s", 1)], tmp_path)
    rep = assign_duplicate_groups(cfg, recs)
    assert rep["duplicate_groups"] == 0 and recs[0].group_size == 1
    assert len(anchor_groups(np.array([], dtype=np.uint64), 4)) == 0


def test_oversized_group_is_reported(tmp_path):
    specs = [(f"A/{i}.jpg", "A", f"s{i}", i % 2) for i in range(50)]  # all within 4 bits
    cfg, recs = _mk(specs, tmp_path, max_group_fraction=0.1)
    rep = assign_duplicate_groups(cfg, recs)
    assert rep["oversized_groups"] and rep["oversized_groups"][0]["class_name"] == "A"


def test_split_per_class_distribution_and_no_group_leakage(tmp_path):
    rng = np.random.default_rng(1)
    specs = []
    for cls in ("A", "B"):
        for i in range(600):
            specs.append((f"{cls}/{i:04d}.jpg", cls, f"{cls}{i}", int(rng.integers(0, 2 ** 63))))
    cfg, recs = _mk(specs, tmp_path, class_set="26")
    assign_duplicate_groups(cfg, recs)
    s = build_splits(cfg, recs)
    owner = {}
    for split, rows in s["rows"].items():
        for r in rows:
            owner.setdefault(r["group_id"], set()).add(split)
    assert all(len(v) == 1 for v in owner.values())
    for cls in ("A", "B"):
        c = s["per_class"][cls]
        t = sum(c.values())
        assert abs(c["train"] / t - 0.70) < 0.05 and abs(c["validation"] / t - 0.15) < 0.05


def test_validation_flags_oversized_group(cfg, dataset):
    from preprocessing.dataset_validation import validate_outputs
    from preprocessing.labels import write_labels, load_labels
    root = class_root_of(dataset)
    records = _prepare(cfg, root)
    assign_duplicate_groups(cfg, records)
    s = build_splits(cfg, records)
    for rows in s["rows"].values():
        for r in rows:
            if r["class_name"] == "A":
                r["group_size"] = 13  # 13 of 14 images claimed in one group
    write_manifests(cfg, s["rows"])
    write_labels(cfg)
    rep = validate_outputs(cfg, root, load_labels(cfg.labels_path), [])
    assert "group_size_sane" in [c.name for c in rep.critical_failures]
