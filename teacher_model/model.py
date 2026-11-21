"""ResNet50 teacher (torchvision) with a 29-class head."""

from __future__ import annotations

import logging

import torch
from torch import nn
from torchvision import models

LOG = logging.getLogger(__name__)
DEFAULT_NUM_CLASSES = 29


def build_teacher(num_classes: int = DEFAULT_NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Standard torchvision ResNet50 whose final ``fc`` is replaced by ``Linear(2048, num_classes)``.

    ``pretrained=True`` downloads ImageNet weights (needs internet on first use, cached afterwards in
    the torch hub cache). If the download fails a ``RuntimeError`` is raised -- the code never silently
    falls back to random weights. Use ``pretrained=False`` (``--no-pretrained``) to train from scratch.
    Output logits have shape ``[batch_size, num_classes]`` (raw, un-softmaxed) which is what knowledge
    distillation needs.
    """
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    weights = None
    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
    try:
        model = models.resnet50(weights=weights)
    except Exception as exc:  # network / cache problems
        raise RuntimeError(
            "Could not obtain ImageNet ResNet50 weights (internet required on first use). "
            "Retry online or run with --no-pretrained. Original error: " + repr(exc)) from exc
    LOG.info("ResNet50 built (ImageNet pretrained weights: %s)", "LOADED" if pretrained else "NOT used")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


@torch.no_grad()
def predict_logits(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Eval-mode forward pass helper (e.g. for later distillation)."""
    was_training = model.training
    model.eval()
    out = model(batch)
    model.train(was_training)
    return out
