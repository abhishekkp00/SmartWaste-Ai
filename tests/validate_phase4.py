"""
tests/validate_phase4.py
─────────────────────────────────────────────────────────────────────────────
Phase 4 validation for ResNet18 architecture and Grad-CAM.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import cfg
from ml.dataset import create_dataloaders
from ml.gradcam import GradCAM, overlay_heatmap, save_heatmap
from ml.resnet_model import (
    ResNet18Classifier,
    build_resnet18_stage_a,
    build_resnet18_stage_b,
    count_parameters,
)
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

    section("1. Stage A Construction")
    try:
        stage_a = build_resnet18_stage_a(pretrained=False)
        stage_a = stage_a.to(device)
        record("Stage A model type", isinstance(stage_a, ResNet18Classifier))

        trainable, total = count_parameters(stage_a)
        record("Stage A has trainable head", trainable > 0, f"trainable={trainable:,}")
        record("Stage A freezes backbone", trainable < total,
               f"trainable={trainable:,}, total={total:,}")

        fc_trainable = sum(parameter.requires_grad for parameter in stage_a.backbone.fc.parameters())
        fc_total = sum(1 for parameter in stage_a.backbone.fc.parameters())
        record("Stage A classifier head trainable", fc_trainable == fc_total,
               f"{fc_trainable}/{fc_total}")
    except Exception as exc:
        record("Stage A construction", False, str(exc))
        return print_summary()

    section("2. Stage B Construction")
    try:
        stage_b = build_resnet18_stage_b(pretrained=False)
        stage_b = stage_b.to(device)

        trainable_b, total_b = count_parameters(stage_b)
        record("Stage B has trainable parameters", trainable_b > 0,
               f"trainable={trainable_b:,}")

        layer1_frozen = all(not parameter.requires_grad for parameter in stage_b.backbone.layer1.parameters())
        layer2_frozen = all(not parameter.requires_grad for parameter in stage_b.backbone.layer2.parameters())
        layer3_trainable = any(parameter.requires_grad for parameter in stage_b.backbone.layer3.parameters())
        layer4_trainable = any(parameter.requires_grad for parameter in stage_b.backbone.layer4.parameters())
        fc_trainable = any(parameter.requires_grad for parameter in stage_b.backbone.fc.parameters())

        record("Stage B freezes layer1", layer1_frozen)
        record("Stage B freezes layer2", layer2_frozen)
        record("Stage B trains layer3", layer3_trainable)
        record("Stage B trains layer4", layer4_trainable)
        record("Stage B trains classifier head", fc_trainable)
    except Exception as exc:
        record("Stage B construction", False, str(exc))

    section("3. Synthetic Forward Pass")
    try:
        dummy = torch.randn(4, 3, cfg.IMAGE_SIZE[0], cfg.IMAGE_SIZE[1], device=device)
        stage_a.eval()
        with torch.no_grad():
            logits = stage_a(dummy)

        record("Logit shape", tuple(logits.shape) == (4, cfg.NUM_CLASSES),
               f"got {tuple(logits.shape)}")
        record("Logits are finite", torch.isfinite(logits).all().item())
        record("Logits are dtype float32", logits.dtype == torch.float32,
               f"got {logits.dtype}")
    except Exception as exc:
        record("Synthetic forward pass", False, str(exc))

    section("4. Real DataLoader Batch")
    try:
        train_loader, _, _ = create_dataloaders(batch_size=4, num_workers=0)
        images, labels = next(iter(train_loader))
        images = images.to(device)

        stage_a.eval()
        with torch.no_grad():
            real_logits = stage_a(images)

        record("Real batch image shape", tuple(images.shape) == (4, 3, 224, 224),
               f"got {tuple(images.shape)}")
        record("Real batch label shape", tuple(labels.shape) == (4,),
               f"got {tuple(labels.shape)}")
        record("Real batch logits shape", tuple(real_logits.shape) == (4, cfg.NUM_CLASSES),
               f"got {tuple(real_logits.shape)}")
        record("Real batch logits finite", torch.isfinite(real_logits).all().item())
    except Exception as exc:
        record("Real DataLoader compatibility", False, str(exc))

    section("5. Grad-CAM")
    try:
        images, labels = next(iter(train_loader))
        images = images[:1].to(device)
        grad_model = build_resnet18_stage_a(pretrained=False).to(device)

        with GradCAM(grad_model) as cam:
            heatmap, predicted = cam.generate(images)

        record("Heatmap shape matches input", heatmap.shape == (224, 224),
               f"got {heatmap.shape}")
        record("Heatmap normalized", heatmap.min() >= 0.0 and heatmap.max() <= 1.0,
               f"range=[{heatmap.min():.3f}, {heatmap.max():.3f}]")
        record("Predicted class valid", 0 <= predicted < cfg.NUM_CLASSES,
               f"got {predicted}")

        rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        overlay = overlay_heatmap(rgb, heatmap)
        record("Overlay shape", overlay.shape == (224, 224, 3), f"got {overlay.shape}")

        heatmap_path = cfg.paths.eda_dir / "phase4_gradcam_heatmap.png"
        saved = save_heatmap(heatmap, heatmap_path, title=f"Predicted class {predicted}")
        record("Heatmap output saved", saved.exists(), str(saved))
    except Exception as exc:
        record("Grad-CAM generation", False, str(exc))

    section("6. Configuration Guards")
    try:
        ResNet18Classifier(num_classes=1, pretrained=False)
        record("Reject num_classes=1", False, "constructor did not raise")
    except ValueError:
        record("Reject num_classes=1", True)

    return print_summary()


def print_summary() -> int:
    passed = sum(ok for _, ok, _ in RESULTS)
    failed = len(RESULTS) - passed

    print(f"\n{'═' * 60}\n Phase 4 Validation Summary\n{'═' * 60}")
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

    print("\n ✅ ALL PHASE 4 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
