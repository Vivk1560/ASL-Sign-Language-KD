"""Knowledge Distillation objective (Hinton et al., 2015).

Mathematical formulation
------------------------
Let ``z_s`` and ``z_t`` be the raw student / teacher logits of one sample, ``T > 0`` the distillation
temperature and ``alpha`` in ``[0, 1]`` the weight of the supervised term::

    L_CE = CrossEntropy(softmax(z_s), y)                         (hard labels)
    L_KD = T^2 * KL( softmax(z_t / T)  ||  softmax(z_s / T) )    (soft labels)
         = T^2 * sum_c p_t(c) * [log p_t(c) - log p_s(c)]        averaged over the batch
    L    = alpha * L_CE + (1 - alpha) * L_KD

Implementation notes
    * ``log_softmax`` is used for the student and ``softmax`` for the teacher, as required by
      ``torch.nn.functional.kl_div`` (input = log-probabilities, target = probabilities).
    * ``reduction="batchmean"`` divides the summed KL by the batch size, which is the mathematically
      correct KL divergence (``reduction="mean"`` would also divide by the number of classes).
    * The ``T^2`` factor keeps the gradient magnitude of the soft-label term comparable to the
      hard-label term when ``T`` is changed.
    * Teacher logits are detached: no gradient can ever flow into the teacher.
    * Everything is computed in float32, even when the forward pass ran under mixed precision.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_TEMPERATURE = 4.0
DEFAULT_ALPHA = 0.5


def validate_kd_hyperparameters(temperature: float, alpha: float) -> None:
    """Raise ``ValueError`` unless ``temperature`` is finite and > 0 and ``alpha`` is finite in [0, 1]."""
    for name, value in (("temperature", temperature), ("alpha", alpha)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a real number, got {value!r}")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature!r}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")


def kd_divergence(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                  temperature: float) -> torch.Tensor:
    """``T^2 * KL(softmax(z_t/T) || softmax(z_s/T))``, batch-mean reduced. Returns a scalar tensor."""
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError("student and teacher logits must both have shape [batch, classes]; got "
                         f"{tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}")
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError(f"temperature must be finite and > 0, got {temperature!r}")
    log_p_student = F.log_softmax(student_logits.float() / temperature, dim=1)
    p_teacher = F.softmax(teacher_logits.detach().float() / temperature, dim=1)
    return F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (temperature ** 2)


class DistillationOutput(NamedTuple):
    total: torch.Tensor   # alpha * ce + (1 - alpha) * kd  (the value to back-propagate)
    ce: torch.Tensor      # supervised cross-entropy of the student
    kd: torch.Tensor      # temperature-scaled teacher->student KL divergence


class DistillationLoss(nn.Module):
    """``L = alpha * CE(student, y) + (1 - alpha) * T^2 * KL(teacher_T || student_T)``."""

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE, alpha: float = DEFAULT_ALPHA,
                 label_smoothing: float = 0.0) -> None:
        super().__init__()
        validate_kd_hyperparameters(temperature, alpha)
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing!r}")
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.label_smoothing = float(label_smoothing)

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                targets: torch.Tensor) -> DistillationOutput:
        if targets.ndim != 1 or targets.shape[0] != student_logits.shape[0]:
            raise ValueError(f"targets must have shape [batch]; got {tuple(targets.shape)} for "
                             f"logits {tuple(student_logits.shape)}")
        ce = F.cross_entropy(student_logits.float(), targets, label_smoothing=self.label_smoothing)
        kd = kd_divergence(student_logits, teacher_logits, self.temperature)
        return DistillationOutput(self.alpha * ce + (1.0 - self.alpha) * kd, ce, kd)
