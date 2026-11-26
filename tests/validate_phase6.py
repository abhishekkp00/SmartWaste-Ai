"""
tests/validate_phase6.py
─────────────────────────────────────────────────────────────────────────────
Phase 6 evaluation validation using a deterministic synthetic test DataLoader.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import cfg
from ml.custom_cnn import build_custom_cnn
from ml.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    save_classification_report,
    plot_training_history,
    plot_model_comparison,
)

RESULTS: List[Tuple[str, bool, str]] = []


def record(name: str, passed: bool, message: str = "") -> None:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  {name}" + (f" — {message}" if message else ""))
    RESULTS.append((name, passed, message))


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n {title}\n{'─' * 60}")


class SyntheticTestLoader:
    def __iter__(self):
        yield torch.randint(0, 255, (8, 3, 224, 224), dtype=torch.uint8).float() / 255.0, torch.tensor([0, 1, 2, 3, 4, 5, 0, 1])


def main() -> int:
    section("1. Test-Set Evaluation")
    try:
        model = build_custom_cnn()
        metrics = evaluate_model(model, SyntheticTestLoader(), criterion=torch.nn.CrossEntropyLoss(), device=torch.device("cpu"))

        record("Accuracy key", "accuracy" in metrics)
        record("Macro precision key", "precision_macro" in metrics)
        record("Macro recall key", "recall_macro" in metrics)
        record("Macro F1 key", "f1_macro" in metrics)
        record("Prediction count", len(metrics["predictions"]) == 8,
               f"got {len(metrics['predictions'])}")
        record("Target count", len(metrics["targets"]) == 8,
               f"got {len(metrics['targets'])}")
        record("Evaluation loss finite", metrics["loss"] is not None and torch.isfinite(torch.tensor(metrics["loss"])).item())
    except Exception as exc:
        record("Test-set evaluation", False, str(exc))
        return print_summary()

    section("2. Confusion Matrix")
    try:
        targets = [0, 0, 1, 1, 2, 3, 4, 5]
        predictions = [0, 1, 1, 1, 2, 3, 4, 5]
        matrix_path = plot_confusion_matrix(
            targets,
            predictions,
            output_path=cfg.paths.confusion_matrix_dir / "phase6_validation_confusion_matrix.png",
        )
        record("Confusion matrix saved", matrix_path.exists(), str(matrix_path))

        raw_matrix_path = plot_confusion_matrix(
            targets,
            predictions,
            normalize=False,
            output_path=cfg.paths.confusion_matrix_dir / "phase6_validation_counts.png",
        )
        record("Unnormalized confusion matrix saved", raw_matrix_path.exists(), str(raw_matrix_path))
    except Exception as exc:
        record("Confusion matrix", False, str(exc))

    section("3. Classification Report")
    try:
        report_path = save_classification_report(
            metrics,
            cfg.paths.reports_dir / "phase6_validation_report.json",
        )
        record("Classification report saved", report_path.exists(), str(report_path))

        import json
        report = json.loads(report_path.read_text(encoding="utf-8"))
        record("Report contains six classes", len(report["per_class"]) == cfg.NUM_CLASSES,
               f"got {len(report['per_class'])}")
    except Exception as exc:
        record("Classification report", False, str(exc))

    section("4. Training and Comparison Plots")
    try:
        history = {
            "train_loss": [1.2, 0.9, 0.7],
            "val_loss": [1.3, 1.0, 0.8],
            "train_acc": [0.35, 0.55, 0.68],
            "val_acc": [0.32, 0.50, 0.62],
        }
        history_path = plot_training_history(history)
        record("Training history plot saved", history_path.exists(), str(history_path))

        comparison = {
            "CustomCNN": {"accuracy": 0.75, "f1_macro": 0.72},
            "ResNet18 Stage A": {"accuracy": 0.82, "f1_macro": 0.79},
            "ResNet18 Stage B": {"accuracy": 0.85, "f1_macro": 0.83},
        }
        comparison_path = plot_model_comparison(comparison)
        record("Model comparison plot saved", comparison_path.exists(), str(comparison_path))
    except Exception as exc:
        record("Evaluation plots", False, str(exc))

    return print_summary()


def print_summary() -> int:
    passed = sum(ok for _, ok, _ in RESULTS)
    failed = len(RESULTS) - passed

    print(f"\n{'═' * 60}\n Phase 6 Validation Summary\n{'═' * 60}")
    print(f"  Total checks : {len(RESULTS)}")
    print(f"  ✅ Passed    : {passed}")
    print(f"  ❌ Failed    : {failed}")
    print("═" * 60)

    if failed:
        print("\n Failed checks:")
        for name, ok, message in RESULTS:
            if not ok:
                print(f"  ✗ {name}: {message}")
        return 1

    print("\n ✅ ALL PHASE 6 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
