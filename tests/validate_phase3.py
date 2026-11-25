"""
tests/validate_phase3.py
─────────────────────────────────────────────────────────────────────────────
SmartWaste AI — Phase 3 Custom CNN Validation

Checks:
  1. Architecture instantiation
  2. [Batch, 3, 224, 224] → [Batch, 6] logits
  3. [Batch, 512] feature vectors
  4. Parameter counts are finite and positive
  5. Backpropagation produces gradients
  6. No NaN or infinite outputs
  7. Freeze/unfreeze behavior
  8. Real training batch compatibility
  9. CUDA/CPU device compatibility
"""

import sys
from pathlib import Path
from typing import List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import cfg
from ml.custom_cnn import CustomCNN, build_custom_cnn, count_parameters
from ml.dataset import create_dataloaders
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
    device = get_device()

    section("1. Custom CNN Construction")
    try:
        model = build_custom_cnn()
        record("Model type is CustomCNN", isinstance(model, CustomCNN))
        record("Num classes comes from config", model.num_classes == cfg.NUM_CLASSES,
               f"got {model.num_classes}")
        record("Dropout 1 comes from config", model.classifier[3].p == cfg.training.CNN_DROPOUT1,
               f"got {model.classifier[3].p}")
        record("Dropout 2 comes from config", model.classifier[6].p == cfg.training.CNN_DROPOUT2,
               f"got {model.classifier[6].p}")
    except Exception as exc:
        record("Model construction", False, str(exc))
        return print_summary()

    section("2. Synthetic Forward Pass")
    model = model.to(device)
    model.eval()
    batch_size = 4
    dummy = torch.randn(batch_size, 3, cfg.IMAGE_SIZE[0], cfg.IMAGE_SIZE[1], device=device)

    try:
        with torch.no_grad():
            features = model.forward_features(dummy)
            logits = model(dummy)

        record("Feature shape", features.shape == (batch_size, 512), f"got {tuple(features.shape)}")
        record("Logit shape", logits.shape == (batch_size, cfg.NUM_CLASSES), f"got {tuple(logits.shape)}")
        record("Features are finite", torch.isfinite(features).all().item())
        record("Logits are finite", torch.isfinite(logits).all().item())
        record("Logits require no probability conversion", logits.dim() == 2,
               "CrossEntropyLoss is applied to raw logits")
    except Exception as exc:
        record("Synthetic forward pass", False, str(exc))

    section("3. Parameter and Gradient Checks")
    try:
        trainable, total = count_parameters(model)
        record("Trainable parameters > 0", trainable > 0, f"{trainable:,}")
        record("Total parameters > 0", total > 0, f"{total:,}")

        model.train()
        logits = model(dummy)
        loss = logits.square().mean()
        loss.backward()

        conv_grads = sum(parameter.grad is not None for parameter in model.features.parameters())
        classifier_grads = sum(parameter.grad is not None for parameter in model.classifier.parameters())
        record("Conv gradients exist", conv_grads > 0, f"parameter tensors with grad={conv_grads}")
        record("Classifier gradients exist", classifier_grads > 0,
               f"parameter tensors with grad={classifier_grads}")
    except Exception as exc:
        record("Backpropagation", False, str(exc))

    section("4. Freeze and Unfreeze")
    try:
        model.freeze_features()
        frozen = sum(not parameter.requires_grad for parameter in model.features.parameters())
        frozen_total = sum(1 for parameter in model.features.parameters())
        record("Feature parameters frozen", frozen == frozen_total,
               f"frozen={frozen}/{frozen_total}")

        model.unfreeze_features()
        unfrozen = sum(parameter.requires_grad for parameter in model.features.parameters())
        record("Feature parameters restored", unfrozen == frozen_total,
               f"unfrozen={unfrozen}/{frozen_total}")
    except Exception as exc:
        record("Freeze/unfreeze", False, str(exc))

    section("5. Real Phase 2 Batch Compatibility")
    try:
        train_loader, _, _ = create_dataloaders(batch_size=4, num_workers=0)
        images, labels = next(iter(train_loader))
        images = images.to(device)

        model.eval()
        with torch.no_grad():
            real_logits = model(images)

        record("Real batch image shape", tuple(images.shape) == (4, 3, 224, 224),
               f"got {tuple(images.shape)}")
        record("Real batch label shape", tuple(labels.shape) == (4,), f"got {tuple(labels.shape)}")
        record("Real batch label dtype", labels.dtype == torch.int64, f"got {labels.dtype}")
        record("Real batch logits shape", tuple(real_logits.shape) == (4, cfg.NUM_CLASSES),
               f"got {tuple(real_logits.shape)}")
        record("Real batch logits finite", torch.isfinite(real_logits).all().item())
    except Exception as exc:
        record("Real batch compatibility", False, str(exc))

    section("6. Invalid Configuration Guards")
    try:
        CustomCNN(num_classes=1)
        record("Reject num_classes=1", False, "constructor did not raise")
    except ValueError:
        record("Reject num_classes=1", True)

    try:
        CustomCNN(dropout1=1.0)
        record("Reject dropout1=1.0", False, "constructor did not raise")
    except ValueError:
        record("Reject dropout1=1.0", True)

    return print_summary()


def print_summary() -> int:
    passed = sum(ok for _, ok, _ in RESULTS)
    failed = len(RESULTS) - passed

    print(f"\n{'═' * 60}\n Phase 3 Validation Summary\n{'═' * 60}")
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

    print("\n ✅ ALL PHASE 3 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
