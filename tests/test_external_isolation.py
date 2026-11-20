"""Partition 2 regression tests: official external test images must never reach the internal split.

All fixtures are synthetic (tests/conftest.py); no Kaggle download is needed.
"""

import shutil

import pytest

from preprocessing.data_cleaning import clean_records, refresh_cleaning_report
from preprocessing.data_splitting import build_splits, write_manifests
from preprocessing.dataset_inspection import enumerate_records, inspect_dataset
from preprocessing.dataset_validation import validate_outputs
from preprocessing.duplicate_detection import assign_duplicate_groups, compute_hashes
from preprocessing.external_test import (build_external_test_manifest, check_isolation,
                                         exclude_external_overlaps, exclusions_report_path,
                                         load_external_test_set)
from preprocessing.labels import load_labels, write_labels
from preprocessing.main import EXIT_FAILED, EXIT_OK, main
from preprocessing.records import R_EXTERNAL_TEST, read_csv, read_json, write_csv
from tests.conftest import class_root_of, random_image

SPLITS = ("train", "validation", "test")


def _test_dir(dataset):
    return dataset / "asl_alphabet_test" / "asl_alphabet_test"


def _add_c_test(dataset):
    """Add a distinct external C image (the default fixture only has A,B,nothing,space)."""
    random_image(424242).save(_test_dir(dataset) / "C_test.jpg", quality=95)
    return _test_dir(dataset) / "C_test.jpg"


def _prepare(cfg, dataset, *, exclude=True):
    """Run the Partition-2 stage order by hand; returns a dict of intermediate objects."""
    root = class_root_of(dataset)
    records, structure = enumerate_records(root)
    summary = inspect_dataset(cfg, root, records, structure)
    cleaning = clean_records(cfg, root, records)
    compute_hashes(cfg, root, records)
    ext = load_external_test_set(cfg, dataset)
    excl = exclude_external_overlaps(cfg, records, ext) if exclude else None
    if excl:
        cleaning = refresh_cleaning_report(cfg, root, records, excl["_details"])
    dups = assign_duplicate_groups(cfg, records)
    write_labels(cfg)
    split = build_splits(cfg, records)
    write_manifests(cfg, split["rows"])
    return dict(root=root, records=records, summary=summary, cleaning=cleaning, ext=ext,
                excl=excl, dups=dups, split=split)


def _validate(cfg, p, ext="default"):
    return validate_outputs(cfg, p["root"], load_labels(cfg.labels_path),
                            p["summary"]["unexpected_folders"], None,
                            external_test=p["ext"] if ext == "default" else ext)


def _check(rep, name="no_external_test_sha256_leakage"):
    return next(c for c in rep.checks if c.name == name)


# 1 ------------------------------------------------------------------ exact match excluded
def test_exact_external_sha_match_is_excluded(cfg, dataset):
    c = _add_c_test(dataset)
    shutil.copy(c, class_root_of(dataset) / "C" / "leaky_copy.jpg")
    p = _prepare(cfg, dataset)
    rec = next(r for r in p["records"] if r.rel_path == "C/leaky_copy.jpg")
    assert rec.status == "excluded" and rec.reason == R_EXTERNAL_TEST == "external_test_exact_duplicate"
    rep = read_json(exclusions_report_path(cfg))
    assert rep["total_internal_records_excluded"] == 1
    assert rep["total_exact_overlaps_found"] == 1
    row = rep["exclusions"][0]
    assert (row["external_file"], row["external_class"], row["internal_image_path"],
            row["internal_class"], row["class_matches"]) == ("C_test.jpg", "C", "C/leaky_copy.jpg",
                                                              "C", True)
    assert row["sha256"] == rec.sha256
    # report keeps the schema fields and lists unmatched external files (A/B/nothing/space)
    assert "A_test.jpg" in rep["external_files_without_internal_match"]
    assert rep["total_external_files_examined"] == 5


# 2 ------------------------------------------------------------ non-matching stays valid
def test_non_matching_internal_images_remain_valid(cfg, dataset):
    c = _add_c_test(dataset)
    shutil.copy(c, class_root_of(dataset) / "C" / "leaky_copy.jpg")
    p = _prepare(cfg, dataset)
    others = [r for r in p["records"] if r.rel_path != "C/leaky_copy.jpg"
              and r.class_name in cfg.classes]
    assert others and all(r.status == "valid" for r in others)


def test_exclusion_is_by_hash_never_by_filename(cfg, dataset):
    """An internal file NAMED like an external one but with different bytes stays valid."""
    random_image(31337).save(class_root_of(dataset) / "A" / "A_test.jpg", quality=95)
    p = _prepare(cfg, dataset)
    rec = next(r for r in p["records"] if r.rel_path == "A/A_test.jpg")
    assert rec.status == "valid"
    assert p["excl"]["total_internal_records_excluded"] == 0


# 3 --------------------------------------------------------- excluded before grouping
def test_exclusion_happens_before_duplicate_grouping(cfg, dataset):
    c = _add_c_test(dataset)
    for n in ("leak1.jpg", "leak2.jpg"):  # two internal copies would form an exact-duplicate group
        shutil.copy(c, class_root_of(dataset) / "C" / n)
    p = _prepare(cfg, dataset)
    leaked = [r for r in p["records"] if r.rel_path.startswith("C/leak")]
    assert len(leaked) == 2 and all(r.status == "excluded" and r.group_id == "" for r in leaked)
    members = {m for g in p["dups"]["groups"] for m in g["members"]}
    assert not any(r.rel_path in members for r in leaked)
    assert p["dups"]["images_considered"] == sum(1 for r in p["records"] if r.status == "valid")
    # control: without exclusion the two copies WOULD have been grouped together
    cfg2 = cfg.with_overrides(processed_dir=cfg.processed_dir.parent / "control")
    ctrl = _prepare(cfg2, dataset, exclude=False)
    members2 = {m for g in ctrl["dups"]["groups"] for m in g["members"]}
    assert {"C/leak1.jpg", "C/leak2.jpg"} <= members2


# cleaning report -------------------------------------------------------------------------
def test_cleaning_report_reflects_final_state_and_is_not_corruption(cfg, dataset):
    c = _add_c_test(dataset)
    shutil.copy(c, class_root_of(dataset) / "C" / "leaky_copy.jpg")
    root = class_root_of(dataset)
    records, _ = enumerate_records(root)
    before = clean_records(cfg, root, records)
    compute_hashes(cfg, root, records)
    ext = load_external_test_set(cfg, dataset)
    excl = exclude_external_overlaps(cfg, records, ext)
    after = refresh_cleaning_report(cfg, root, records, excl["_details"])
    assert after["valid"] == before["valid"] - 1 and after["excluded"] == before["excluded"] + 1
    assert after["excluded_by_reason"][R_EXTERNAL_TEST] == 1
    assert "corrupt_or_unreadable" not in after["excluded_by_reason"]
    assert after["external_test_exact_duplicate_paths"] == ["C/leaky_copy.jpg"]
    assert any(e["image_path"] == "C/leaky_copy.jpg" and "C_test.jpg" in e["detail"]
               for e in after["excluded_files"])


# 4 + hash reuse + raw untouched: full pipeline ---------------------------------------------
def test_full_pipeline_excludes_external_copies_from_all_manifests(tmp_path, dataset, monkeypatch):
    c = _add_c_test(dataset)
    for cls_dir_name, n in (("C", "dup_of_c_test.jpg"),):
        shutil.copy(c, class_root_of(dataset) / cls_dir_name / n)
    from preprocessing import external_test as et
    calls = []
    real = et.sha256_file
    monkeypatch.setattr(et, "sha256_file", lambda path, *a, **k: (calls.append(path.name), real(path, *a, **k))[1])
    before = sorted((q.relative_to(dataset).as_posix(), q.read_bytes()) for q in dataset.rglob("*") if q.is_file())
    proc = tmp_path / "proc"
    assert main(["--raw-dir", str(dataset), "--processed-dir", str(proc), "--workers", "2",
                 "--skip-visualization"]) == EXIT_OK
    after = sorted((q.relative_to(dataset).as_posix(), q.read_bytes()) for q in dataset.rglob("*") if q.is_file())
    assert before == after  # raw dataset untouched (excluded file still on disk)
    assert sorted(calls) == sorted(["A_test.jpg", "B_test.jpg", "C_test.jpg", "nothing_test.jpg",
                                    "space_test.jpg"])  # each external image hashed exactly once
    for s in SPLITS:
        assert not any(r["image_path"] == "C/dup_of_c_test.jpg" for r in read_csv(proc / "manifests" / f"{s}.csv"))
    ext_sha = {r["sha256"] for r in read_csv(proc / "manifests" / "external_test.csv")}
    for s in SPLITS:
        assert not ({r["sha256"] for r in read_csv(proc / "manifests" / f"{s}.csv")} & ext_sha)
    val = read_json(proc / "reports" / "dataset_validation_report.json")
    chk = next(c for c in val["checks"] if c["name"] == "no_external_test_sha256_leakage")
    assert val["ok"] and chk["passed"] and chk["severity"] == "critical"
    clean = read_json(proc / "reports" / "corrupt_image_report.json")
    assert clean["excluded_by_reason"][R_EXTERNAL_TEST] == 1
    assert read_json(proc / "reports" / "external_test_exclusions.json")["total_internal_records_excluded"] == 1
    rep = read_json(proc / "reports" / "external_test_report.json")
    assert rep["status"] == "isolated" and rep["isolation_checked"] and rep["isolation_passed"] is True
    assert rep["overlap_count"] == 0
    summ = read_json(proc / "metadata" / "dataset_summary.json")
    assert summ["external_test"]["internal_records_excluded"] == 1


# 5-8: leakage into each split is a critical failure ----------------------------------------
@pytest.mark.parametrize("split_name", SPLITS)
def test_external_sha_in_any_split_manifest_is_critical_failure(cfg, dataset, split_name):
    p = _prepare(cfg, dataset)
    path = cfg.manifest_path(split_name)
    rows = read_csv(path)
    ext_sha = next(f.sha256 for f in p["ext"].files if f.filename == "A_test.jpg")
    rows[0]["sha256"] = ext_sha  # simulate leakage ON DISK (in-memory data is untouched)
    write_csv(path, list(rows[0].keys()), rows)
    rep = _validate(cfg, p)
    chk = _check(rep)
    assert not chk.passed and chk.severity == "critical" and split_name in chk.items[0]
    assert "no_external_test_sha256_leakage" in [c.name for c in rep.critical_failures]
    assert not rep.ok
    assert check_isolation(cfg, p["ext"])["overlaps"][0]["split"] == split_name
    iso = build_external_test_manifest(cfg, dataset, external=p["ext"])
    assert iso["status"] == "isolation_failed" and iso["isolation_passed"] is False
    assert iso["overlap_count"] >= 1 and "A_test.jpg" in iso["training_sha256_overlap"]
    assert iso["found"] is True and iso["mapped"] > 0  # discovery fine, isolation failed


def test_pipeline_fails_if_unexcluded_overlap_reaches_split(cfg, dataset):
    """If exclusion is skipped, the leaked copy lands in a split and validation catches it."""
    c = _add_c_test(dataset)
    shutil.copy(c, class_root_of(dataset) / "C" / "leaky_copy.jpg")
    p = _prepare(cfg, dataset, exclude=False)
    rep = _validate(cfg, p)
    assert not rep.ok and "no_external_test_sha256_leakage" in [c.name for c in rep.critical_failures]


# 9 ---------------------------------------------------------------------- zero overlap passes
def test_zero_overlap_passes(cfg, dataset):
    p = _prepare(cfg, dataset)
    rep = _validate(cfg, p)
    assert _check(rep).passed and rep.ok
    iso = build_external_test_manifest(cfg, dataset, external=p["ext"])
    assert iso["status"] == "isolated" and iso["overlap_count"] == 0


# 10 ------------------------------------------------------------ missing external directory
def test_missing_external_dir_is_explicit_not_silent(tmp_path, cfg, dataset):
    shutil.rmtree(dataset / "asl_alphabet_test")
    p = _prepare(cfg, dataset)
    assert p["ext"].found is False and p["excl"]["total_internal_records_excluded"] == 0
    assert exclusions_report_path(cfg).is_file()
    rep = _validate(cfg, p)
    chk = _check(rep)
    assert not chk.passed and chk.severity == "warning" and "NOT VERIFIED" in chk.detail
    assert rep.ok  # optional by default
    strict = cfg.with_overrides(require_external_test=True)
    rep2 = _validate(strict, p)
    assert not rep2.ok and "no_external_test_sha256_leakage" in [c.name for c in rep2.critical_failures]
    # validation without any external info is never a silent pass either
    assert not _check(_validate(cfg, p, ext=None)).passed
    out = build_external_test_manifest(cfg, dataset, external=p["ext"])
    assert out["found"] is False and out["status"] == "test folder not found"
    assert out["isolation_checked"] is False and out["isolation_passed"] is None


def test_cli_require_external_test_fails_fast_when_folder_missing(tmp_path, dataset):
    shutil.rmtree(dataset / "asl_alphabet_test")
    args = ["--raw-dir", str(dataset), "--processed-dir", str(tmp_path / "p"), "--workers", "2",
            "--skip-visualization"]
    assert main(args + ["--require-external-test"]) == EXIT_FAILED
    assert main(args) == EXIT_OK  # default: reported as NOT VERIFIED warning, pipeline continues


# 11 ------------------------------------------------------ expected missing del is documented
def test_missing_del_is_documented_not_a_leakage_failure(cfg, dataset):
    cfg29 = cfg.with_overrides(class_set="29")
    p = _prepare(cfg29, dataset)
    rep = build_external_test_manifest(cfg29, dataset, external=p["ext"])
    assert "del" in rep["classes_without_test_image"]
    assert "del" not in rep["unexpected_missing_classes"]
    assert "del" in rep["expected_missing_classes"]
    assert rep["isolation_passed"] is True and rep["status"] == "isolated"
    assert _check(_validate(cfg29, p)).passed
    assert "del" in read_json(exclusions_report_path(cfg29))["expected_missing_classes"]


# 12 ----------------------------------------- existing conflict behaviour intact after change
def test_cross_class_conflict_still_detected_alongside_external_exclusion(cfg, dataset):
    root = class_root_of(dataset)
    shutil.copy(root / "A" / "A1.jpg", root / "B" / "conflict_copy.jpg")
    c = _add_c_test(dataset)
    shutil.copy(c, root / "C" / "leaky_copy.jpg")
    p = _prepare(cfg, dataset)
    reasons = {r.rel_path: r.reason for r in p["records"] if r.status == "excluded"}
    assert reasons["A/A1.jpg"] == "label_conflict_exact_duplicate"
    assert reasons["B/conflict_copy.jpg"] == "label_conflict_exact_duplicate"
    assert reasons["C/leaky_copy.jpg"] == R_EXTERNAL_TEST
    assert len(p["dups"]["label_conflicts"]) == 2  # external exclusion did not mask or add conflicts
