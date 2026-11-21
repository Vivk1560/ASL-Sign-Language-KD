"""On-the-fly image transforms.

Training uses a random rotation done with ``scipy.ndimage.rotate``; validation/test use a purely
deterministic resize + normalise. Nothing here ever writes an augmented image to disk.

Pipeline per training sample (all in memory, one image at a time):
    PIL image -> resize (H, W) -> uint8 HWC array -> random SciPy rotation -> float tensor -> normalise
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from .config import IMAGENET_MEAN, IMAGENET_STD


def rotate_image_array(image: np.ndarray, angle_deg: float, order: int = 1,
                       fill: str | float = "mean") -> np.ndarray:
    """Rotate an HWC (or HW) uint8 image by ``angle_deg`` with ``scipy.ndimage.rotate``.

    * ``axes=(1, 0)`` selects the spatial plane (width, height) so the channel axis of an HWC array
      is never rotated/mixed. Positive angles rotate counter-clockwise, negative clockwise.
    * ``reshape=False`` keeps the output the same size as the input (no enlarged canvas, and the
      parts that rotate outside the frame are cropped by design).
    * ``order=1`` (bilinear) avoids ringing; ``mode='constant'`` fills empty corners with ``cval``.
    * ``fill='mean'`` uses each channel's mean (less jarring than black); a number uses that constant.
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"expected HW or HWC array, got shape {image.shape}")
    if angle_deg == 0.0:
        return image.copy()
    if image.ndim == 2:
        cval = float(image.mean()) if fill == "mean" else float(fill)
        out = ndimage.rotate(image, angle_deg, axes=(1, 0), reshape=False, order=order,
                             mode="constant", cval=cval)
    else:
        channels = []
        for c in range(image.shape[2]):
            plane = image[:, :, c]
            cval = float(plane.mean()) if fill == "mean" else float(fill)
            channels.append(ndimage.rotate(plane, angle_deg, axes=(1, 0), reshape=False,
                                           order=order, mode="constant", cval=cval))
        out = np.stack(channels, axis=2)
    if image.dtype == np.uint8:
        out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out


class RandomScipyRotation:
    """Sample an angle uniformly from [-max_degrees, +max_degrees] on every call."""

    def __init__(self, max_degrees: float = 15.0, seed: int = 0, order: int = 1,
                 fill: str | float = "mean") -> None:
        if max_degrees < 0:
            raise ValueError("max_degrees must be >= 0")
        self.max_degrees = float(max_degrees)
        self.seed = seed
        self.order = order
        self.fill = fill
        self._rng: np.random.Generator | None = None
        self._rng_key: int | None = None
        self.last_angle: float = 0.0

    def _get_rng(self) -> np.random.Generator:
        # torch.initial_seed() differs per DataLoader worker *and* per epoch (non-persistent workers),
        # so workers never share angle sequences and epochs do not repeat the same angles.
        key = torch.initial_seed()
        if self._rng is None or self._rng_key != key:
            self._rng = np.random.default_rng(np.random.SeedSequence([self.seed, key % (2**32)]))
            self._rng_key = key
        return self._rng

    def sample_angle(self) -> float:
        if self.max_degrees == 0:
            return 0.0
        return float(self._get_rng().uniform(-self.max_degrees, self.max_degrees))

    def __call__(self, image: np.ndarray) -> np.ndarray:
        self.last_angle = self.sample_angle()
        return rotate_image_array(image, self.last_angle, order=self.order, fill=self.fill)


def array_to_normalized_tensor(image: np.ndarray) -> torch.Tensor:
    """uint8 HWC -> float32 CHW, scaled to [0, 1] and ImageNet-normalised."""
    tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def _resize_to_array(image: Image.Image, image_size: int) -> np.ndarray:
    rgb = image.convert("RGB")
    if rgb.size != (image_size, image_size):
        rgb = rgb.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.asarray(rgb, dtype=np.uint8)


class EvalTransform:
    """Deterministic transform for validation / internal-test / inference."""

    is_random = False

    def __init__(self, image_size: int = 224) -> None:
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return array_to_normalized_tensor(_resize_to_array(image, self.image_size))


class TrainTransform:
    """Training transform: resize -> random SciPy rotation -> tensor -> normalise."""

    is_random = True

    def __init__(self, image_size: int = 224, rotation_deg: float = 15.0, seed: int = 0) -> None:
        self.image_size = image_size
        self.rotation = RandomScipyRotation(rotation_deg, seed=seed)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        array = _resize_to_array(image, self.image_size)
        array = self.rotation(array)
        return array_to_normalized_tensor(array)
