"""Shared record type and small, side-effect-free I/O helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

STATUS_VALID = "valid"
STATUS_EXCLUDED = "excluded"
STATUS_NOT_SELECTED = "not_selected"


@dataclass
class ImageRecord:
    """One file found under a class folder; enriched as the pipeline advances."""

    rel_path: str  # POSIX path relative to the class root, e.g. "A/A1.jpg"
    class_folder: str
    class_name: str | None  # canonical class name or None if folder is unexpected
    ext: str
    size_bytes: int
    width: int | None = None
    height: int | None = None
    header_ok: bool | None = None
    status: str = "pending"
    reason: str = ""
    sha256: str = ""
    dhash: str = ""
    group_id: str = ""
    group_size: int = 1


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):  # numpy scalars
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)!r}")


def atomic_write_text(path: Path, text: str) -> Path:
    """Write text via a temp file + rename so partial writes never appear."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def write_json(path: Path, data: Any) -> Path:
    """Write deterministic, human-readable JSON (sorted keys, UTF-8, LF)."""
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default)
    return atomic_write_text(path, text + "\n")


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> Path:
    """Write a CSV with LF line endings on every platform."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def is_within(path: Path, parent: Path) -> bool:
    """True if ``path`` is ``parent`` or located below it (after resolving)."""
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False
