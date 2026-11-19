"""Training-only augmentation (Pillow + NumPy; no flips, no extra dependencies).

Approach: on-the-fly augmentation is the default (``TrainAugmentation``); the
training Dataset in a later partition calls it for the train split only. An
optional offline generator writes images plus a ``train_augmented.csv`` that
keeps the source -> augmented relationship.

Why not Albumentations: the earlier notebook's ``CoarseDropout(max_holes=...)``
was silently ignored on newer Albumentations (Phase-1 finding #11). Occlusion is
implemented here with explicit parameters and is covered by a test proving it
changes pixels. Horizontal flip is intentionally absent (handedness matters).
"""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance

from .config import AugmentationConfig, PreprocessConfig
from .records import write_csv

LOG = logging.getLogger(__name__)
AUG_FIELDS = ["aug_path", "source_path", "source_sha256", "group_id", "class_name",
              "class_index", "aug_index", "seed", "split"]


def rng_from_key(base_seed: int, key: str) -> np.random.Generator:
    """Deterministic RNG for (seed, key), e.g. key = "<image_path>|<epoch>"."""
    digest = hashlib.sha256(f"{base_seed}|{key}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


class TrainAugmentation:
    """Mild geometric/photometric augmentation suitable for hand-sign images."""

    def __init__(self, cfg: AugmentationConfig | None = None, seed: int = 0) -> None:
        self.cfg = cfg or AugmentationConfig()
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def __call__(self, image: Image.Image) -> Image.Image:
        return self.apply(image, self._rng)

    def for_sample(self, image: Image.Image, key: str) -> Image.Image:
        """Reproducible augmentation tied to ``key`` (independent of call order)."""
        return self.apply(image, rng_from_key(self.seed, key))

    def apply(self, image: Image.Image, rng: np.random.Generator) -> Image.Image:
        c = self.cfg
        img = image.convert("RGB")
        img = self._affine(img, rng)
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(*c.brightness_range))
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(*c.contrast_range))
        img = ImageEnhance.Color(img).enhance(rng.uniform(*c.saturation_range))
        if c.channel_gain > 0:
            gains = 1.0 + rng.uniform(-c.channel_gain, c.channel_gain, size=3)
            arr = np.asarray(img, dtype=np.float32) * gains
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        if rng.random() < c.occlusion_prob:
            img = self._occlude(img, rng)
        return img

    def _affine(self, img: Image.Image, rng: np.random.Generator) -> Image.Image:
        c = self.cfg
        w, h = img.size
        theta = math.radians(rng.uniform(-c.rotation_deg, c.rotation_deg))
        s = rng.uniform(*c.scale_range)
        tx = rng.uniform(-c.translate_frac, c.translate_frac) * w
        ty = rng.uniform(-c.translate_frac, c.translate_frac) * h
        cx, cy = w / 2.0, h / 2.0
        cos, sin = math.cos(theta), math.sin(theta)
        a, b = cos / s, sin / s
        d, e = -sin / s, cos / s
        cc = cx - (cos * (cx + tx) + sin * (cy + ty)) / s
        f = cy - (-sin * (cx + tx) + cos * (cy + ty)) / s
        fill = tuple(int(v) for v in np.asarray(img).reshape(-1, 3).mean(axis=0))
        return img.transform((w, h), Image.Transform.AFFINE, (a, b, cc, d, e, f),
                             resample=Image.Resampling.BILINEAR, fillcolor=fill)

    def _occlude(self, img: Image.Image, rng: np.random.Generator) -> Image.Image:
        """Random rectangular occlusion(s); replacement for the broken CoarseDropout call."""
        c = self.cfg
        arr = np.array(img)
        h, w = arr.shape[:2]
        for _ in range(int(rng.integers(1, c.occlusion_max_holes + 1))):
            area = rng.uniform(*c.occlusion_area_frac) * h * w
            aspect = rng.uniform(0.5, 2.0)
            bh = int(min(h, max(1, round(math.sqrt(area / aspect)))))
            bw = int(min(w, max(1, round(math.sqrt(area * aspect)))))
            y = int(rng.integers(0, h - bh + 1))
            x = int(rng.integers(0, w - bw + 1))
            arr[y:y + bh, x:x + bw] = rng.integers(0, 256, size=3, dtype=np.uint8)
        return Image.fromarray(arr)

    def signature(self) -> dict[str, Any]:
        """Parameter record stored in metadata."""
        from dataclasses import asdict
        return {"seed": self.seed, "horizontal_flip": False, **asdict(self.cfg)}


def generate_offline_augmentations(
    cfg: PreprocessConfig, class_root: Path, train_rows: list[dict[str, Any]],
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Write augmented copies of TRAIN images only; return manifest rows.

    Existing output files are kept unless ``overwrite`` is set (rerun-safe).
    """
    if any(r.get("split") != "train" for r in train_rows):
        raise ValueError("Offline augmentation received non-train rows; refusing to proceed")
    aug = TrainAugmentation(cfg.augmentation, seed=cfg.seed)
    out_root = cfg.augmented_dir / "train"
    out_rows: list[dict[str, Any]] = []
    copies = cfg.augmentation.offline_copies_per_image
    for row in train_rows:
        src = Path(class_root) / row["image_path"]
        stem = Path(row["image_path"]).stem
        with Image.open(src) as im:
            im.load()
            source = im.copy()
        for k in range(copies):
            rel = f"{row['class_name']}/{row['group_id']}_{stem}__aug{k}.jpg"
            dest = out_root / rel
            key = f"{row['image_path']}|aug{k}"
            if overwrite or not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                aug.for_sample(source, key).save(
                    dest, format="JPEG", quality=cfg.augmentation.offline_jpeg_quality)
            out_rows.append({
                "aug_path": f"train/{rel}", "source_path": row["image_path"],
                "source_sha256": row["sha256"], "group_id": row["group_id"],
                "class_name": row["class_name"], "class_index": row["class_index"],
                "aug_index": k, "seed": cfg.seed, "split": "train"})
    out_rows.sort(key=lambda r: r["aug_path"])
    write_csv(cfg.augmented_manifest_path, AUG_FIELDS, out_rows)
    LOG.info("Offline augmentation: %d images for %d train sources", len(out_rows), len(train_rows))
    return out_rows
