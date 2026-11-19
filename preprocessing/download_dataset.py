"""Optional Kaggle acquisition and manual-dataset support.

Credentials are never stored in source. The Kaggle API reads them from
``KAGGLE_USERNAME``/``KAGGLE_KEY`` (or ``KAGGLE_API_TOKEN``) or from
``kaggle.json`` in the user's ``.kaggle`` directory.
"""

from __future__ import annotations

import logging
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .config import PreprocessConfig
from .dataset_inspection import discover_class_root

LOG = logging.getLogger(__name__)


@dataclass
class DatasetStatus:
    available: bool
    class_root: Path | None
    source: str  # "existing" | "downloaded" | "extracted" | "missing"
    message: str


def manual_instructions(cfg: PreprocessConfig) -> str:
    """Return exact manual-download instructions (Windows-friendly)."""
    return (
        "ASL Alphabet dataset not found.\n"
        f"  Expected a folder tree under: {cfg.raw_dir}\n"
        "Option A - manual download (no API needed):\n"
        f"  1. Open https://www.kaggle.com/datasets/{cfg.kaggle_dataset} and click Download.\n"
        f"  2. Extract the archive so that folders A, B, ..., Z, del, nothing, space exist under\n"
        f"     {cfg.raw_dir}  (nesting such as asl_alphabet_train\\asl_alphabet_train\\A is fine).\n"
        "  3. Or keep the dataset elsewhere:  python -m preprocessing.main --raw-dir D:\\path\\to\\asl\n"
        "  4. Or leave the .zip in the raw folder and add --extract-zip.\n"
        "Option B - Kaggle API:\n"
        "  1. pip install kaggle   2. create an API token on kaggle.com (Settings > API)\n"
        "  3. save kaggle.json to %USERPROFILE%\\.kaggle\\ (or set KAGGLE_USERNAME / KAGGLE_KEY)\n"
        "  4. python -m preprocessing.main --download"
    )


def kaggle_credentials_present() -> bool:
    """True if any supported Kaggle credential source appears to exist."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    return (config_dir / "kaggle.json").is_file()


def extract_archive(zip_path: Path, dest: Path) -> None:
    """Extract a zip into ``dest`` with a zip-slip guard. The zip itself is untouched."""
    dest = Path(dest).resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if dest != target and dest not in target.parents:
                raise ValueError(f"Unsafe path in archive: {member.filename}")
        zf.extractall(dest)


def download_from_kaggle(cfg: PreprocessConfig) -> tuple[bool, str]:
    """Attempt a Kaggle download. Returns ``(ok, message)``; never raises on auth problems."""
    if not kaggle_credentials_present():
        return False, "Kaggle credentials not found (see instructions)."
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # lazy: optional dependency
    except Exception as exc:  # noqa: BLE001
        return False, f"kaggle package unavailable ({type(exc).__name__}: {exc}). pip install kaggle"
    try:
        api = KaggleApi()
        api.authenticate()
        Path(cfg.raw_dir).mkdir(parents=True, exist_ok=True)
        LOG.info("Downloading %s into %s (large, ~1 GB compressed)...", cfg.kaggle_dataset, cfg.raw_dir)
        api.dataset_download_files(cfg.kaggle_dataset, path=str(cfg.raw_dir), unzip=True, quiet=False)
    except Exception as exc:  # noqa: BLE001
        return False, f"Kaggle download failed: {type(exc).__name__}: {exc}"
    return True, "download call completed"


def ensure_dataset(
    cfg: PreprocessConfig, download: bool = False, extract_zip: bool = False
) -> DatasetStatus:
    """Locate the dataset, optionally extracting a local zip or downloading from Kaggle.

    "available" is only reported after the class folders were actually found on disk.
    """
    root = discover_class_root(cfg.raw_dir, cfg.min_class_folders_to_detect)
    if root is not None:
        return DatasetStatus(True, root, "existing", f"Dataset found at {root}")

    notes: list[str] = []
    if extract_zip and Path(cfg.raw_dir).is_dir():
        zips = sorted(Path(cfg.raw_dir).glob("*.zip"))
        if zips:
            LOG.info("Extracting %s into %s", zips[0].name, cfg.raw_dir)
            extract_archive(zips[0], Path(cfg.raw_dir))
            root = discover_class_root(cfg.raw_dir, cfg.min_class_folders_to_detect)
            if root is not None:
                return DatasetStatus(True, root, "extracted", f"Extracted {zips[0].name}")
            notes.append(f"Extracted {zips[0].name} but found no class folders.")
        else:
            notes.append("--extract-zip given but no .zip found in the raw directory.")

    if download:
        ok, msg = download_from_kaggle(cfg)
        notes.append(msg)
        if ok:
            root = discover_class_root(cfg.raw_dir, cfg.min_class_folders_to_detect)
            if root is not None:
                return DatasetStatus(True, root, "downloaded", f"Downloaded and verified at {root}")
            notes.append("Download finished but no class folders were found.")

    message = " ".join(notes + [manual_instructions(cfg)])
    return DatasetStatus(False, None, "missing", message)
