"""Classification metrics (scikit-learn) and confusion-matrix plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]) -> dict[str, Any]:
    """Accuracy, macro/weighted precision-recall-F1, per-class scores, report and confusion matrix."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = list(range(len(class_names)))
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    p_c, r_c, f_c, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "num_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_macro), "recall_macro": float(r_macro), "f1_macro": float(f_macro),
        "precision_weighted": float(p_w), "recall_weighted": float(r_w), "f1_weighted": float(f_w),
        "per_class": {name: {"precision": float(p_c[i]), "recall": float(r_c[i]),
                             "f1": float(f_c[i]), "support": int(support[i])}
                      for i, name in enumerate(class_names)},
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=list(class_names), zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def save_confusion_matrix_png(matrix: Sequence[Sequence[int]], class_names: Sequence[str], path: Path,
                              title: str = "Teacher ResNet50 - internal test confusion matrix") -> Path:
    """Save a row-normalised confusion matrix image (headless-safe backend)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(matrix, dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n)); ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticks(range(n)); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Row-normalised fraction")
    if n <= 30:
        for i in range(n):
            for j in range(n):
                if cm[i, j] > 0:
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6,
                            color="white" if norm[i, j] > 0.5 else "black")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
