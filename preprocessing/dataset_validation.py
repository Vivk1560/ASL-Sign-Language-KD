"""Post-processing validation. Critical failures stop the pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .config import SPLIT_NAMES, PreprocessConfig
from .data_splitting import MANIFEST_FIELDS
from .dataset_inspection import compute_raw_fingerprint
from .labels import labels_mismatches
from .records import is_within, read_csv

LOG = logging.getLogger(__name__)


class ValidationError(RuntimeError):
    """Raised when at least one critical validation check fails."""


@dataclass
class CheckResult:
    name: str
    severity: str  # "critical" | "warning"
    passed: bool
    detail: str = ""
    items: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, severity: str, problems: list[str], ok_detail: str = "ok") -> None:
        self.checks.append(CheckResult(
            name, severity, not problems,
            ok_detail if not problems else f"{len(problems)} problem(s)", problems[:20]))

    @property
    def critical_failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.severity == "critical" and not c.passed]

    @property
    def ok(self) -> bool:
        return not self.critical_failures

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok,
                "critical_failures": [c.name for c in self.critical_failures],
                "checks": [c.__dict__ for c in self.checks]}

    def raise_if_critical(self) -> None:
        if not self.ok:
            names = ", ".join(c.name for c in self.critical_failures)
            raise ValidationError(f"Critical validation check(s) failed: {names}")


def _decodable(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:  # noqa: BLE001
        return False


def validate_outputs(
    cfg: PreprocessConfig, class_root: Path, labels: dict[str, Any],
    unexpected_folders: list[str], raw_fingerprint_before: str | None = None,
) -> ValidationReport:
    """Run every check against what is on disk (manifests are re-read from CSV)."""
    rep = ValidationReport()
    class_root = Path(class_root)
    manifests: dict[str, list[dict[str, str]]] = {}
    problems: list[str] = []
    for s in SPLIT_NAMES:
        p = cfg.manifest_path(s)
        if not p.is_file():
            problems.append(f"missing manifest {p.name}")
        else:
            manifests[s] = read_csv(p)
    rep.add("manifests_exist", "critical", problems)
    if problems:
        return rep

    rep.add("splits_not_empty", "critical",
            [f"{s} is empty" for s in SPLIT_NAMES if not manifests[s]])

    cols = [f"{s}: columns differ" for s in SPLIT_NAMES
            if manifests[s] and list(manifests[s][0].keys()) != MANIFEST_FIELDS]
    cols += [f"{s}: wrong split value" for s in SPLIT_NAMES
             if any(r["split"] != s for r in manifests[s])]
    rep.add("manifest_consistency", "critical", cols)

    c2i = labels["class_to_idx"]
    bad = []
    for s in SPLIT_NAMES:
        for r in manifests[s]:
            if r["class_name"] not in c2i:
                bad.append(f"{s}:{r['image_path']} invalid label {r['class_name']!r}")
            elif int(r["class_index"]) != c2i[r["class_name"]]:
                bad.append(f"{s}:{r['image_path']} index {r['class_index']} != mapping")
    rep.add("labels_valid_and_consistent", "critical", bad)
    rep.add("labels_json_matches_config", "critical", labels_mismatches(labels, cfg))

    seen = {s: {r["class_name"] for r in manifests[s]} for s in SPLIT_NAMES}
    rep.add("every_class_in_every_split", "critical",
            [f"class {c} missing from {s}" for s in SPLIT_NAMES for c in cfg.classes
             if c not in seen[s]])

    for key, name in (("image_path", "no_path_overlap"), ("sha256", "no_exact_duplicate_leakage"),
                      ("group_id", "no_group_leakage")):
        owner: dict[str, str] = {}
        leaks = []
        for s in SPLIT_NAMES:
            for v in {r[key] for r in manifests[s]}:
                if v in owner and owner[v] != s:
                    leaks.append(f"{key} {v[:16]} in {owner[v]} and {s}")
                owner.setdefault(v, s)
        rep.add(name, "critical", leaks)
    dup_within = [f"{s}: duplicate path" for s in SPLIT_NAMES
                  if len({r['image_path'] for r in manifests[s]}) != len(manifests[s])]
    rep.add("no_duplicate_rows_within_split", "critical", dup_within)

    missing = [r["image_path"] for s in SPLIT_NAMES for r in manifests[s]
               if not (class_root / r["image_path"]).is_file()]
    rep.add("files_exist", "critical", missing)

    if cfg.decode_check_in_validation:
        from concurrent.futures import ThreadPoolExecutor
        paths = [class_root / r["image_path"] for s in SPLIT_NAMES for r in manifests[s]
                 if (class_root / r["image_path"]).is_file()]
        with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
            flags = list(pool.map(_decodable, paths))
        rep.add("images_decodable", "critical",
                [str(p) for p, ok in zip(paths, flags) if not ok])

    rep.add("unexpected_class_folders", "warning",
            [f"unexpected folder: {f}" for f in unexpected_folders])

    # Augmentation leakage: val/test must stay raw; augmented files must derive from train only.
    aug_problems: list[str] = []
    aug_root = cfg.augmented_dir
    for s in ("validation", "test"):
        aug_problems += [f"{s} row points into augmented dir: {r['image_path']}"
                         for r in manifests[s] if r["image_path"].startswith(("train/", "augmented"))
                         or is_within(class_root / r["image_path"], aug_root)]
    train_paths = {r["image_path"] for r in manifests["train"]}
    train_groups = {r["group_id"] for r in manifests["train"]}
    other_groups = {r["group_id"] for s in ("validation", "test") for r in manifests[s]}
    listed: set[str] = set()
    if cfg.augmented_manifest_path.is_file():
        for r in read_csv(cfg.augmented_manifest_path):
            listed.add(r["aug_path"])
            if r["split"] != "train" or r["source_path"] not in train_paths:
                aug_problems.append(f"augmented row from non-train source: {r['source_path']}")
            if r["group_id"] in other_groups or r["group_id"] not in train_groups:
                aug_problems.append(f"augmented row group leaks: {r['group_id']}")
            if not (aug_root / r["aug_path"]).is_file():
                aug_problems.append(f"augmented file missing: {r['aug_path']}")
    if aug_root.is_dir():
        on_disk = {p.relative_to(aug_root).as_posix() for p in aug_root.rglob("*") if p.is_file()}
        aug_problems += [f"augmented file not in manifest / outside train: {p}"
                         for p in sorted(on_disk - listed)]
    rep.add("no_augmentation_leakage", "critical", aug_problems)

    if raw_fingerprint_before is not None:
        now = compute_raw_fingerprint(class_root)
        rep.add("raw_dataset_unchanged", "critical",
                [] if now == raw_fingerprint_before else ["raw dataset fingerprint changed"])
    for c in rep.checks:
        LOG.log(logging.INFO if c.passed else logging.ERROR, "validation %-34s %s",
                c.name, "PASS" if c.passed else f"FAIL ({c.severity}) {c.detail}")
    return rep
