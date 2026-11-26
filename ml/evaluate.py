"""
ml/evaluate.py
─────────────────────────────────────────────────────────────────────────────
SmartWaste AI — Evaluation and model-comparison utilities

Provides:
  - test-set inference
  - confusion matrix generation
  - per-class classification metrics
  - loss/accuracy history plots
  - model comparison plots

This module evaluates existing checkpoints; it does not train models.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import cfg
from ml.metrics import classification_metrics


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader,
    criterion: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """Run test-set inference and return loss, predictions, targets, and metrics."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    predictions: List[int] = []
    targets: List[int] = []
    total_loss = 0.0
    total_samples = 0

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        if criterion is not None:
            total_loss += criterion(logits, labels).item() * labels.size(0)
        total_samples += labels.size(0)

        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(labels.cpu().tolist())

    metrics = classification_metrics(predictions, targets, cfg.NUM_CLASSES)
    metrics["loss"] = total_loss / total_samples if criterion is not None and total_samples else None
    metrics["predictions"] = predictions
    metrics["targets"] = targets
    return metrics


def plot_confusion_matrix(
    targets: Iterable[int],
    predictions: Iterable[int],
    class_names: Optional[Sequence[str]] = None,
    output_path: Optional[Path] = None,
    normalize: bool = True,
) -> Path:
    """Create a labeled confusion matrix and save it as PNG."""
    class_names = list(cfg.CLASS_NAMES if class_names is None else class_names)
    num_classes = len(class_names)

    targets = np.asarray(list(targets), dtype=np.int64)
    predictions = np.asarray(list(predictions), dtype=np.int64)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for target, prediction in zip(targets, predictions):
        if 0 <= target < num_classes and 0 <= prediction < num_classes:
            matrix[target, prediction] += 1

    display_matrix = matrix.astype(np.float64)
    if normalize:
        row_totals = display_matrix.sum(axis=1, keepdims=True)
        display_matrix = np.divide(
            display_matrix,
            row_totals,
            out=np.zeros_like(display_matrix),
            where=row_totals != 0,
        )

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(display_matrix, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0 if normalize else None)
    ax.figure.colorbar(image, ax=ax)

    tick_labels = [name.capitalize() for name in class_names]
    ax.set_xticks(np.arange(num_classes), labels=tick_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(num_classes), labels=tick_labels)

    threshold = display_matrix.max() / 2.0 if display_matrix.size else 0.0
    for row in range(num_classes):
        for column in range(num_classes):
            value = display_matrix[row, column]
            label = f"{value:.2f}" if normalize else str(matrix[row, column])
            ax.text(column, row, label, ha="center", va="center",
                    color="white" if value > threshold else "black", fontsize=9)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    fig.tight_layout()

    output_path = Path(output_path) if output_path else cfg.paths.confusion_matrix_dir / "confusion_matrix.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_classification_report(
    metrics: Dict[str, object],
    output_path: Optional[Path] = None,
) -> Path:
    """Save accuracy and per-class precision/recall/F1 as JSON."""
    per_class = metrics.get("per_class", {})
    named_metrics: Dict[str, object] = {}

    if isinstance(per_class, dict):
        for index_text, values in per_class.items():
            index = int(index_text)
            name = cfg.CLASS_NAMES[index] if 0 <= index < cfg.NUM_CLASSES else index_text
            named_metrics[name] = values

    report = {
        "accuracy": metrics.get("accuracy"),
        "precision_macro": metrics.get("precision_macro"),
        "recall_macro": metrics.get("recall_macro"),
        "f1_macro": metrics.get("f1_macro"),
        "loss": metrics.get("loss"),
        "per_class": named_metrics,
    }

    output_path = Path(output_path) if output_path else cfg.paths.reports_dir / "evaluation_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def plot_training_history(
    history: Dict[str, Sequence[float]],
    output_path: Optional[Path] = None,
) -> Path:
    """Plot training/validation loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history.get("train_loss", []), label="Train loss", marker="o")
    axes[0].plot(history.get("val_loss", []), label="Validation loss", marker="o")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(history.get("train_acc", []), label="Train accuracy", marker="o")
    axes[1].plot(history.get("val_acc", []), label="Validation accuracy", marker="o")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    output_path = Path(output_path) if output_path else cfg.paths.graphs_dir / "training_history.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_model_comparison(
    comparison: Dict[str, Dict[str, float]],
    output_path: Optional[Path] = None,
) -> Path:
    """Plot accuracy/F1 comparison for named evaluated models."""
    names = list(comparison)
    accuracy = [comparison[name].get("accuracy", 0.0) for name in names]
    f1 = [comparison[name].get("f1_macro", 0.0) for name in names]

    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, accuracy, width, label="Accuracy", color="#3498DB")
    ax.bar(x + width / 2, f1, width, label="Macro F1", color="#E67E22")

    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_path = Path(output_path) if output_path else cfg.paths.graphs_dir / "model_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
