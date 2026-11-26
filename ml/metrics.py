"""
ml/metrics.py
─────────────────────────────────────────────────────────────────────────────
SmartWaste AI — Training and evaluation metrics.

Provides accuracy and per-class precision, recall, and F1 calculations
without duplicating class/configuration values.
"""

from typing import Dict, Iterable, List, Sequence

import torch


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Return top-1 classification accuracy for a batch."""
    if logits.shape[0] == 0:
        return 0.0
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()


def classification_metrics(
    predictions: Iterable[int] | torch.Tensor,
    targets: Iterable[int] | torch.Tensor,
    num_classes: int,
) -> Dict[str, object]:
    """Calculate accuracy, per-class metrics, and macro averages."""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().tolist()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().tolist()

    predictions = list(predictions)
    targets = list(targets)
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")

    per_class: Dict[str, Dict[str, float]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1_scores: List[float] = []

    for class_index in range(num_classes):
        tp = sum(pred == class_index and target == class_index for pred, target in zip(predictions, targets))
        fp = sum(pred == class_index and target != class_index for pred, target in zip(predictions, targets))
        fn = sum(pred != class_index and target == class_index for pred, target in zip(predictions, targets))

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        precisions.append(float(precision))
        recalls.append(float(recall))
        f1_scores.append(float(f1))
        per_class[str(class_index)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": float(sum(target == class_index for target in targets)),
        }

    correct = sum(pred == target for pred, target in zip(predictions, targets))
    accuracy = correct / len(targets) if targets else 0.0

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(sum(precisions) / num_classes),
        "recall_macro": float(sum(recalls) / num_classes),
        "f1_macro": float(sum(f1_scores) / num_classes),
        "per_class": per_class,
    }
