"""Official Kaggle test images (external evaluation only) and their isolation from the internal data.

The 28 official images (``A_test.jpg`` ... ``Z_test.jpg``, ``nothing_test.jpg``, ``space_test.jpg``;
there is deliberately no ``del_test.jpg``) are exact byte copies of files that also sit inside the
87,000-image internal training folder. They must therefore be removed from the *eligible* internal
population BEFORE duplicate grouping and splitting; otherwise the "external" test set is contaminated.

Flow
----
1. ``load_external_test_set``      - discover + SHA-256 / dHash the official files ONCE.
2. ``exclude_external_overlaps``   - mark internal VALID records with an identical SHA-256 as
                                     ``external_test_exact_duplicate`` (raw files are never touched).
3. ``build_external_test_manifest``- write ``external_test.csv`` + ``external_test_report.json``,
                                     reusing the hashes from step 1.
4. ``check_isolation``             - re-read the split manifests from disk and look for any external
                                     SHA-256 (also used by ``dataset_validation``).

Status semantics (independent, not one generic "ok")
----------------------------------------------------
``found``             external folder discovered.
``mapped``            number of files mapped to a configured class (from the file name).
``isolation_checked`` the three split manifests exist on disk and were compared.
``isolation_passed``  checked AND zero external SHA-256 in any split (``None`` if not checked).
``overlap_count``     number of external files whose SHA-256 appears in a split manifest.
``status``            ``test folder not found`` | ``isolation_not_checked`` | ``isolation_failed`` |
                      ``isolated``.

Limitations (documented, not hidden)
------------------------------------
* 28 images (one per class, no ``del``) is a smoke test, not a statistical benchmark. The missing
  ``del`` image is EXPECTED and is not an isolation failure.
* ``J`` and ``Z`` are dynamic signs; a single-frame classifier cannot recognise the motion, so
  accuracy on these two is not evidence of temporal sign recognition.
* Only exact (SHA-256) copies are removed. Near-duplicates of external images (e.g. neighbouring
  video frames) may still exist in the internal data.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .config import SPLIT_NAMES, PreprocessConfig
from .dataset_inspection import find_kaggle_test_info
from .duplicate_detection import dhash_image
from .labels import canonical_class_name
from .records import (R_EXTERNAL_TEST, STATUS_VALID, ImageRecord, read_csv, sha256_file,
                      write_csv, write_json)

LOG = logging.getLogger(__name__)
DYNAMIC_SIGNS = ("J", "Z")
FIELDS = ["image_path", "class_name", "class_index", "sha256", "dhash", "width", "height",
          "is_dynamic_sign", "overlaps_training_sha256"]
_NAME = re.compile(r"^(?P<label>.+)_test\.(jpg|jpeg|png|bmp)$", re.IGNORECASE)
EXPECTED_MISSING_CLASSES = {
    "del": "The official Kaggle test folder ships no del_test.jpg. This is expected and is NOT "
           "an isolation failure."}
DYNAMIC_NOTE = ("J and Z are motion signs; single-frame results do not show temporal recognition.")


@dataclass(frozen=True)
class ExternalFile:
    filename: str
    class_name: str | None  # canonical class parsed from the file name; None if not parseable
    sha256: str
    width: int | None
    height: int | None
    dhash: str  # "" if the image could not be decoded

    @property
    def is_dynamic_sign(self) -> bool:
        return self.class_name in DYNAMIC_SIGNS


@dataclass
class ExternalTestSet:
    """Official external files, hashed once and reused by exclusion, manifest and validation."""

    found: bool = False
    path: str | None = None
    files: list[ExternalFile] = field(default_factory=list)
    non_image_files: list[str] = field(default_factory=list)

    @property
    def sha256_set(self) -> set[str]:
        return {f.sha256 for f in self.files}

    def files_by_sha(self) -> dict[str, list[ExternalFile]]:
        out: dict[str, list[ExternalFile]] = {}
        for f in self.files:
            out.setdefault(f.sha256, []).append(f)
        return out


def external_manifest_path(cfg: PreprocessConfig) -> Path:
    return cfg.manifests_dir / "external_test.csv"


def exclusions_report_path(cfg: PreprocessConfig) -> Path:
    return cfg.reports_dir / "external_test_exclusions.json"


def load_external_test_set(cfg: PreprocessConfig, raw_dir: Path | None = None) -> ExternalTestSet:
    """Discover the official external folder and hash every image in it exactly once."""
    info = find_kaggle_test_info(Path(raw_dir or cfg.raw_dir))
    if not info["found"]:
        return ExternalTestSet(found=False, path=None)
    ext = ExternalTestSet(found=True, path=info["path"])
    for f in sorted(Path(info["path"]).iterdir(), key=lambda p: p.name):
        if not f.is_file():
            continue
        if f.suffix.lower() not in cfg.image_extensions:
            ext.non_image_files.append(f.name)
            continue
        m = _NAME.match(f.name)
        cls = canonical_class_name(m.group("label")) if m else None
        digest = sha256_file(f)
        w = h = None
        dh = ""
        try:
            with Image.open(f) as im:
                w, h = im.size
                dh = dhash_image(im)
        except Exception as exc:  # noqa: BLE001 - still hashed by bytes, so still protected
            LOG.warning("External image %s could not be decoded: %s", f.name, exc)
        ext.files.append(ExternalFile(f.name, cls, digest, w, h, dh))
    LOG.info("External test set: %d image files hashed in %s", len(ext.files), ext.path)
    return ext


def exclude_external_overlaps(
    cfg: PreprocessConfig, records: list[ImageRecord], ext: ExternalTestSet,
) -> dict[str, Any]:
    """Mark VALID internal records whose SHA-256 equals an official external image.

    Must run after internal SHA-256 values exist and BEFORE duplicate grouping/splitting.
    Matching is by content hash only (never by file name). Raw files are not touched.
    Writes ``external_test_exclusions.json`` and returns the report; the affected records get
    ``status="excluded"`` and ``reason="external_test_exact_duplicate"``.
    """
    by_sha = ext.files_by_sha()
    rows: list[dict[str, Any]] = []
    details: dict[str, str] = {}
    matched_files: set[str] = set()
    for r in records:
        if r.status != STATUS_VALID or not r.sha256 or r.sha256 not in by_sha:
            continue
        r.status, r.reason = "excluded", R_EXTERNAL_TEST
        for ef in by_sha[r.sha256]:
            matched_files.add(ef.filename)
            rows.append({"external_file": ef.filename, "external_class": ef.class_name,
                         "internal_image_path": r.rel_path, "internal_class": r.class_name,
                         "class_matches": ef.class_name == r.class_name, "sha256": r.sha256})
        details[r.rel_path] = "identical SHA-256 to external test image " + ", ".join(
            sorted(ef.filename for ef in by_sha[r.sha256]))
    excluded_paths = sorted({row["internal_image_path"] for row in rows})
    report = {
        "external_folder_found": ext.found,
        "external_folder": ext.path,
        "total_external_files_examined": len(ext.files),
        "total_exact_overlaps_found": len(matched_files),
        "total_internal_records_excluded": len(excluded_paths),
        "exclusion_reason": R_EXTERNAL_TEST,
        "exclusions": sorted(rows, key=lambda x: (x["external_file"], x["internal_image_path"])),
        "external_files_without_internal_match": sorted(
            f.filename for f in ext.files if f.filename not in matched_files),
        "unmapped_external_files": sorted(
            [f.filename for f in ext.files if f.class_name is None] + ext.non_image_files),
        "expected_missing_classes": EXPECTED_MISSING_CLASSES,
        "matching": "exact SHA-256 on VALID internal records only; file names are never used",
        "note": "Raw files were not modified. Records are excluded from grouping and splitting only.",
    }
    # The external manifest builder / cleaning report want these too.
    report["_details"] = details
    write_json(exclusions_report_path(cfg), {k: v for k, v in report.items() if k != "_details"})
    LOG.info("External exclusion: %d external files examined, %d with an internal copy, "
             "%d internal records excluded", len(ext.files), len(matched_files), len(excluded_paths))
    return report


def check_isolation(cfg: PreprocessConfig, ext: ExternalTestSet) -> dict[str, Any]:
    """Compare official external SHA-256 values with the split manifests ON DISK."""
    manifests = {s: cfg.manifest_path(s) for s in SPLIT_NAMES}
    missing = [s for s, p in manifests.items() if not p.is_file()]
    result: dict[str, Any] = {"checked": False, "missing_manifests": missing, "overlaps": []}
    if missing or not ext.found:
        return result
    by_sha = ext.files_by_sha()
    for s, p in manifests.items():
        for row in read_csv(p):
            for ef in by_sha.get(row["sha256"], ()):
                result["overlaps"].append({"external_file": ef.filename, "split": s,
                                           "image_path": row["image_path"]})
    result["checked"] = True
    return result


def build_external_test_manifest(
    cfg: PreprocessConfig, raw_dir: Path | None = None, external: ExternalTestSet | None = None,
) -> dict[str, Any]:
    """Write ``external_test.csv`` + ``external_test_report.json``; returns the report.

    Pass ``external`` to reuse hashes already computed (otherwise the folder is hashed here).
    """
    ext = external if external is not None else load_external_test_set(cfg, raw_dir)
    report: dict[str, Any] = {
        "found": ext.found, "path": ext.path, "mapped": 0, "unmapped_files": [],
        "classes_without_test_image": [], "expected_missing_classes": EXPECTED_MISSING_CLASSES,
        "unexpected_missing_classes": [], "training_sha256_overlap": [],
        "isolation_checked": False, "isolation_passed": None, "overlap_count": 0,
        "overlaps": [], "dynamic_sign_note": DYNAMIC_NOTE,
        "status_semantics": "found/mapped describe discovery and file-name mapping only; "
                            "isolation_checked/isolation_passed describe leakage against the split "
                            "manifests on disk. 'status' summarises isolation, not discovery.",
    }
    if not ext.found:
        report["status"] = "test folder not found"
        write_json(cfg.reports_dir / "external_test_report.json", report)
        return report
    iso = check_isolation(cfg, ext)
    leaked = {o["external_file"] for o in iso["overlaps"]}
    c2i = cfg.class_to_idx
    rows = []
    for f in ext.files:
        if f.class_name is None or f.class_name not in c2i:
            report["unmapped_files"].append(f.filename)
            continue
        rows.append({"image_path": f.filename, "class_name": f.class_name,
                     "class_index": c2i[f.class_name], "sha256": f.sha256, "dhash": f.dhash,
                     "width": f.width, "height": f.height, "is_dynamic_sign": f.is_dynamic_sign,
                     "overlaps_training_sha256": f.filename in leaked})
    report["unmapped_files"] = sorted(report["unmapped_files"] + ext.non_image_files)
    write_csv(external_manifest_path(cfg), FIELDS, rows)
    present = {r["class_name"] for r in rows}
    report["mapped"] = len(rows)
    report["classes_without_test_image"] = [c for c in cfg.classes if c not in present]
    report["unexpected_missing_classes"] = [c for c in report["classes_without_test_image"]
                                            if c not in EXPECTED_MISSING_CLASSES]
    report["training_sha256_overlap"] = sorted(leaked)
    report["overlaps"] = iso["overlaps"]
    report["overlap_count"] = len(leaked)
    report["isolation_checked"] = iso["checked"]
    report["isolation_passed"] = (not leaked) if iso["checked"] else None
    report["status"] = ("isolation_not_checked" if not iso["checked"]
                        else "isolation_failed" if leaked else "isolated")
    write_json(cfg.reports_dir / "external_test_report.json", report)
    LOG.info("External test manifest: %d mapped, %d unmapped, missing classes %s, status=%s",
             len(rows), len(report["unmapped_files"]), report["classes_without_test_image"],
             report["status"])
    return report
