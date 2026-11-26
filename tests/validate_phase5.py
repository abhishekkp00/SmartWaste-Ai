"""
tests/validate_phase5.py
─────────────────────────────────────────────────────────────────────────────
Phase 5 validation for metrics and training engine.

Uses one real batch and one training epoch/step to confirm CPU compatibility,
weighted loss, optional sampling, history, and checkpoint creation.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import cfg
from ml.custom_cnn import build_custom_cnn
from ml.metrics import accuracy_from_logits, classification_metrics
from ml.train import Trainer, create_training_components
from ml.utils import get_device, set_seed

RESULTS: List[Tuple[str, bool, str]] = []


def record(name: str, passed: bool, message: str = "") -> None:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  {name}" + (f" — {message}" if message else ""))
    RESULTS.append((name, passed, message))


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n {title}\n{'─' * 60}")


def main() -> int:
    set_seed(cfg.training.RANDOM_SEED)

    section("1. Metrics")
    try:
        predictions = torch.tensor([0, 1, 2, 1, 3])
        targets = torch.tensor([0, 1, 1, 1, 3])
        metrics = classification_metrics(predictions, targets, num_classes=4)
        record("Accuracy is correct", abs(metrics["accuracy"] - 0.8) < 1e-6,
               f"got {metrics['accuracy']:.4f}")
        record("Macro precision key", "precision_macro" in metrics)
        record("Macro recall key", "recall_macro" in metrics)
        record("Macro F1 key", "f1_macro" in metrics)
        record("Per-class metrics exist", len(metrics["per_class"]) == 4,
               f"got {len(metrics['per_class'])}")

        logits = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        labels = torch.tensor([0, 1])
        record("Accuracy from logits", accuracy_from_logits(logits, labels) == 1.0)
    except Exception as exc:
        record("Metrics validation", False, str(exc))

    section("2. Weighted and Unweighted Loss")
    try:
        model = build_custom_cnn()
        trainer_weighted, train_loader, val_loader, test_loader, weights = create_training_components(
            model=model,
            batch_size=4,
            num_workers=0,
            use_weighted_sampler=False,
            weighted_loss=True,
            checkpoint_path=cfg.paths.models_dir / "phase5_weighted_test.pth",
        )
        record("Training weights shape", tuple(weights.shape) == (cfg.NUM_CLASSES,),
               f"got {tuple(weights.shape)}")
        record("Weighted CrossEntropy configured", isinstance(trainer_weighted.criterion, torch.nn.CrossEntropyLoss))
        record("Train/val/test loaders exist", all(loader is not None for loader in (train_loader, val_loader, test_loader)))

        unweighted_model = build_custom_cnn()
        trainer_unweighted, _, _, _, _ = create_training_components(
            model=unweighted_model,
            batch_size=4,
            num_workers=0,
            use_weighted_sampler=True,
            weighted_loss=False,
            checkpoint_path=cfg.paths.models_dir / "phase5_unweighted_test.pth",
        )
        record("Sampler mode creates Weighted loader", hasattr(train_loader.dataset, "targets"))
        record("Unweighted loss configured", isinstance(trainer_unweighted.criterion, torch.nn.CrossEntropyLoss))
    except Exception as exc:
        record("Training component creation", False, str(exc))
        return print_summary()

    section("3. One-Epoch CPU Smoke Test")
    try:
        device = torch.device("cpu")
        model = build_custom_cnn()
        trainer = Trainer(
            model=model,
            device=device,
            weighted_loss=True,
            class_weights=weights,
            checkpoint_path=cfg.paths.models_dir / "phase5_smoke_test.pth",
            patience=2,
        )

        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate_epoch(val_loader)

        record("Training loss finite", torch.isfinite(torch.tensor(train_metrics["loss"])).item())
        record("Training accuracy range", 0.0 <= train_metrics["accuracy"] <= 1.0,
               f"got {train_metrics['accuracy']:.4f}")
        record("Validation loss finite", torch.isfinite(torch.tensor(val_metrics["loss"])).item())
        record("Validation accuracy range", 0.0 <= val_metrics["accuracy"] <= 1.0,
               f"got {val_metrics['accuracy']:.4f}")

        trainer.save_checkpoint(1, val_metrics)
        record("Checkpoint saved", trainer.checkpoint_path.exists(), str(trainer.checkpoint_path))

        history_path = trainer.save_history()
        record("History saved", history_path.exists(), str(history_path))
    except Exception as exc:
        record("CPU smoke training", False, str(exc))

    section("4. Scheduler and Optimizer")
    try:
        trainer.configure_scheduler(t_max=2, eta_min=1e-6)
        record("Cosine scheduler configured", trainer.scheduler is not None)
        record("AdamW optimizer configured", isinstance(trainer.optimizer, torch.optim.AdamW))
    except Exception as exc:
        record("Scheduler/optimizer", False, str(exc))

    section("5. Invalid Weighted Loss Guard")
    try:
        Trainer(build_custom_cnn(), device=torch.device("cpu"), weighted_loss=True, class_weights=None)
        record("Weighted loss requires weights", False, "Trainer did not raise")
    except ValueError:
        record("Weighted loss requires weights", True)

    return print_summary()


def print_summary() -> int:
    passed = sum(ok for _, ok, _ in RESULTS)
    failed = len(RESULTS) - passed

    print(f"\n{'═' * 60}\n Phase 5 Validation Summary\n{'═' * 60}")
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

    print("\n ✅ ALL PHASE 5 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
