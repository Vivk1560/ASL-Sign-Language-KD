"""MobileNetV2 student (torchvision) with a 29-class head."""

from __future__ import annotations

import logging

from torch import nn
from torchvision import models

from teacher_model.model import DEFAULT_NUM_CLASSES, count_parameters  # noqa: F401  (re-exported)

LOG = logging.getLogger(__name__)


def build_student(num_classes: int = DEFAULT_NUM_CLASSES, pretrained: bool = True,
                  dropout: float = 0.2) -> nn.Module:
    """torchvision MobileNetV2 whose final ``Linear(1280, 1000)`` is replaced by ``Linear(1280, num_classes)``.

    ``pretrained=True`` loads ImageNet weights (internet needed on first use, cached afterwards). A failed
    download raises ``RuntimeError`` -- there is never a silent fallback to random weights. Output logits
    have shape ``[batch_size, num_classes]`` (raw, un-softmaxed), the same convention as the teacher.
    """
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    try:
        model = models.mobilenet_v2(weights=weights, dropout=dropout)
    except Exception as exc:  # network / cache problems
        raise RuntimeError(
            "Could not obtain ImageNet MobileNetV2 weights (internet required on first use). "
            "Retry online or run with --no-pretrained. Original error: " + repr(exc)) from exc
    LOG.info("MobileNetV2 built (ImageNet pretrained weights: %s)", "LOADED" if pretrained else "NOT used")
    model.classifier[-1] = nn.Linear(model.last_channel, num_classes)
    return model
