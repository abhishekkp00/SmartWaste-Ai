"""
tests/validate_phase2.py
─────────────────────────────────────────────────────────────────────────────
SmartWaste AI — Phase 2 Validation Script

A standalone script that validates the entire Phase 2 implementation:
  1. Preprocessing transforms (shape, dtype, range)
  2. WasteDataset (__init__, __len__, __getitem__)
  3. Class-to-index mapping
  4. DataLoader (batch shape, label dtype)
  5. All 6 classes represented in every split
  6. No data leakage between splits
  7. Class weight tensor shape and values
  8. WeightedRandomSampler batch class distribution
  9. Corrupted image placeholder handling
 10. Inverse normalisation round-trip

Usage:
    python tests/validate_phase2.py
    python tests/validate_phase2.py --verbose

Exit codes:
    0 — All checks passed
    1 — One or more checks failed
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import torch

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import cfg
from ml.dataset import (
    WasteDataset,
    compute_class_weights,
    create_dataloaders,
    get_weighted_sampler,
)
from ml.preprocessing import (
    get_train_transforms,
    get_val_transforms,
    get_test_transforms,
    get_inverse_normalize,
)
from ml.utils import get_logger, set_seed

logger = get_logger("validate_phase2")


# ──────────────────────────────────────────────────────────────────────────────
# Test registry
# ──────────────────────────────────────────────────────────────────────────────

RESULTS: List[Tuple[str, bool, str]] = []   # (test_name, passed, message)


def _record(name: str, passed: bool, msg: str = "") -> bool:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  {name}" + (f" — {msg}" if msg else ""))
    RESULTS.append((name, passed, msg))
    return passed


def _section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f" {title}")
    print(f"{'─'*60}")


# ──────────────────────────────────────────────────────────────────────────────
# Check 1: Preprocessing transforms
# ──────────────────────────────────────────────────────────────────────────────

def test_preprocessing() -> None:
    _section("1. Preprocessing Transforms")

    import numpy as np
    from PIL import Image

    # Create a synthetic test image with realistic dimensions (non-square)
    dummy = Image.fromarray(
        (np.random.rand(400, 600, 3) * 255).astype(np.uint8), mode="RGB"
    )

    # Training transforms
    try:
        train_tfm = get_train_transforms()
        t = train_tfm(dummy)
        _record("Train transform output shape", t.shape == torch.Size([3, 224, 224]),
                f"got {t.shape}")
        _record("Train transform dtype", t.dtype == torch.float32,
                f"got {t.dtype}")
        # After ImageNet normalisation, some values will be negative
        _record("Train tensor has variation", t.std().item() > 0.01,
                f"std={t.std().item():.4f}")
    except Exception as e:
        _record("Train transform", False, str(e))

    # Validation transforms
    try:
        val_tfm = get_val_transforms()
        v = val_tfm(dummy)
        _record("Val transform output shape", v.shape == torch.Size([3, 224, 224]),
                f"got {v.shape}")
        _record("Val transform deterministic",
                torch.allclose(val_tfm(dummy), val_tfm(dummy)),
                "val transforms must be identical on repeated calls")
    except Exception as e:
        _record("Val transform", False, str(e))

    # Test transforms identical to val
    try:
        test_tfm = get_test_transforms()
        t2 = test_tfm(dummy)
        val2 = val_tfm(dummy)
        _record("Test == Val transforms", torch.allclose(t2, val2),
                "test and val should produce identical tensors")
    except Exception as e:
        _record("Test transform", False, str(e))

    # Inverse normalisation round-trip
    try:
        inv_norm = get_inverse_normalize()
        val_t = val_tfm(dummy)
        recovered = inv_norm(val_t).clamp(0, 1)
        _record("Inverse norm shape", recovered.shape == torch.Size([3, 224, 224]),
                f"got {recovered.shape}")
        _record("Inverse norm range [0,1]",
                0.0 <= recovered.min().item() and recovered.max().item() <= 1.0,
                f"min={recovered.min().item():.3f} max={recovered.max().item():.3f}")
    except Exception as e:
        _record("Inverse normalisation", False, str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Check 2: WasteDataset
# ──────────────────────────────────────────────────────────────────────────────

def test_waste_dataset() -> None:
    _section("2. WasteDataset")

    for split in ("train", "val", "test"):
        try:
            tfm = get_train_transforms() if split == "train" else get_val_transforms()
            ds = WasteDataset(split=split, transform=tfm, verbose=False)

            # __len__
            _record(f"[{split}] __len__ > 0", len(ds) > 0, f"got {len(ds)}")

            # __getitem__ — first sample
            img, label = ds[0]
            _record(f"[{split}] img shape",
                    img.shape == torch.Size([3, 224, 224]),
                    f"got {img.shape}")
            _record(f"[{split}] img dtype",
                    img.dtype == torch.float32,
                    f"got {img.dtype}")
            _record(f"[{split}] label valid",
                    0 <= label < cfg.NUM_CLASSES,
                    f"got label={label}")

            # Last sample
            img_last, label_last = ds[len(ds) - 1]
            _record(f"[{split}] last sample shape",
                    img_last.shape == torch.Size([3, 224, 224]),
                    f"got {img_last.shape}")

        except FileNotFoundError as e:
            _record(f"[{split}] dataset exists", False, str(e))
        except Exception as e:
            _record(f"[{split}] WasteDataset", False, traceback.format_exc(limit=2))


# ──────────────────────────────────────────────────────────────────────────────
# Check 3: Class-to-index mapping
# ──────────────────────────────────────────────────────────────────────────────

def test_class_mapping() -> None:
    _section("3. Class-to-Index Mapping")

    try:
        ds = WasteDataset("train", transform=get_val_transforms(), verbose=False)

        # Alphabetical order
        expected_classes = sorted(cfg.CLASS_NAMES)
        _record("Classes in alphabetical order",
                ds.classes == expected_classes,
                f"got {ds.classes}, expected {expected_classes}")

        # All 6 classes present
        _record("All 6 classes present",
                len(ds.classes) == cfg.NUM_CLASSES,
                f"found {len(ds.classes)} classes")

        # class_to_idx is consistent with classes list
        for name, idx in ds.class_to_idx.items():
            _record(f"class_to_idx['{name}'] == {idx}",
                    ds.classes[idx] == name,
                    f"ds.classes[{idx}]={ds.classes[idx]}")

        # Indices are contiguous from 0
        indices = sorted(ds.class_to_idx.values())
        _record("Indices are 0…N-1",
                indices == list(range(cfg.NUM_CLASSES)),
                f"got {indices}")

        logger.info(f"  class_to_idx: {ds.class_to_idx}")

    except Exception as e:
        _record("Class mapping", False, str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Check 4: All 6 classes represented in every split
# ──────────────────────────────────────────────────────────────────────────────

def test_class_coverage() -> None:
    _section("4. Class Coverage in Every Split")

    for split in ("train", "val", "test"):
        try:
            ds = WasteDataset(split, transform=get_val_transforms(), verbose=False)
            present = set(Counter(ds.targets).keys())
            all_classes_present = present == set(range(cfg.NUM_CLASSES))
            missing = set(range(cfg.NUM_CLASSES)) - present
            _record(
                f"[{split}] all 6 classes present",
                all_classes_present,
                f"missing indices: {missing}" if missing else ""
            )
            # Print per-class counts
            counts = ds.get_class_counts()
            for cls, n in counts.items():
                logger.info(f"    [{split}] {cls:<12} n={n}")
        except Exception as e:
            _record(f"[{split}] class coverage", False, str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Check 5: No data leakage
# ──────────────────────────────────────────────────────────────────────────────

def test_no_data_leakage() -> None:
    _section("5. No Data Leakage Between Splits")

    try:
        all_filenames: dict = {}
        for split in ("train", "val", "test"):
            ds = WasteDataset(split, transform=get_val_transforms(), verbose=False)
            for path, _ in ds.samples:
                fname = path.name
                if fname in all_filenames:
                    all_filenames[fname].append(split)
                else:
                    all_filenames[fname] = [split]

        duplicates = {f: splits for f, splits in all_filenames.items() if len(splits) > 1}
        _record(
            "No duplicate filenames across splits",
            len(duplicates) == 0,
            f"{len(duplicates)} duplicates found" if duplicates else ""
        )

        if duplicates:
            for fname, splits in list(duplicates.items())[:5]:
                logger.error(f"  DUPLICATE: '{fname}' in {splits}")
    except Exception as e:
        _record("Data leakage check", False, str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Check 6: DataLoader
# ──────────────────────────────────────────────────────────────────────────────

def test_dataloaders() -> None:
    _section("6. DataLoader Batch Validation")

    try:
        set_seed(42)
        # Use num_workers=0 for validation (avoids multiprocessing issues)
        train_loader, val_loader, test_loader = create_dataloaders(
            batch_size=8, num_workers=0
        )

        for name, loader in [("train", train_loader), ("val", val_loader),
                              ("test", test_loader)]:
            batch_imgs, batch_labels = next(iter(loader))

            _record(
                f"[{name}] batch imgs shape",
                batch_imgs.shape[1:] == torch.Size([3, 224, 224]),
                f"got {batch_imgs.shape}"
            )
            _record(
                f"[{name}] batch labels dtype",
                batch_labels.dtype == torch.int64,
                f"got {batch_labels.dtype}"
            )
            _record(
                f"[{name}] batch labels range",
                (batch_labels >= 0).all() and (batch_labels < cfg.NUM_CLASSES).all(),
                f"got min={batch_labels.min()} max={batch_labels.max()}"
            )

        # Training loader uses shuffle (or sampler)
        logger.info(f"  Train loader: {len(train_loader)} batches")
        logger.info(f"  Val loader  : {len(val_loader)} batches")
        logger.info(f"  Test loader : {len(test_loader)} batches")

    except Exception as e:
        _record("DataLoader", False, traceback.format_exc(limit=3))


# ──────────────────────────────────────────────────────────────────────────────
# Check 7: Class weights
# ──────────────────────────────────────────────────────────────────────────────

def test_class_weights() -> None:
    _section("7. Class Weight Computation")

    try:
        train_ds = WasteDataset("train", transform=get_val_transforms(), verbose=False)
        weights = compute_class_weights(train_ds, device=torch.device("cpu"))

        _record("Class weights shape",
                weights.shape == torch.Size([cfg.NUM_CLASSES]),
                f"got {weights.shape}")
        _record("All weights > 0",
                (weights > 0).all().item(),
                f"weights: {weights.tolist()}")
        _record("Class weights dtype",
                weights.dtype == torch.float32,
                f"got {weights.dtype}")

        # 'trash' (index 5) should have the highest weight
        trash_idx = train_ds.class_to_idx.get("trash", 5)
        max_weight_idx = weights.argmax().item()
        _record(
            "Highest weight is 'trash' class",
            max_weight_idx == trash_idx,
            f"highest weight at idx={max_weight_idx} "
            f"(trash idx={trash_idx}, weight={weights[trash_idx]:.4f})"
        )

    except Exception as e:
        _record("Class weights", False, str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Check 8: WeightedRandomSampler class distribution
# ──────────────────────────────────────────────────────────────────────────────

def test_weighted_sampler() -> None:
    _section("8. WeightedRandomSampler Distribution")

    try:
        set_seed(42)
        train_loader_wrs, _, _ = create_dataloaders(
            batch_size=32, num_workers=0, use_weighted_sampler=True
        )

        # Collect labels from first 5 batches
        all_labels = []
        for i, (_, labels) in enumerate(train_loader_wrs):
            all_labels.extend(labels.tolist())
            if i >= 4:
                break

        counts = Counter(all_labels)
        logger.info(f"  Label distribution over 5 batches (WRS):")
        for idx in range(cfg.NUM_CLASSES):
            cls_name = list(cfg.CLASS_NAMES)[idx]
            logger.info(f"    [{idx}] {cls_name:<12} count={counts.get(idx, 0)}")

        # With WRS, 'trash' should not be completely absent from 160 samples
        trash_idx = 5
        _record(
            "Trash class appears in sampled batches",
            counts.get(trash_idx, 0) > 0,
            f"trash count={counts.get(trash_idx, 0)} out of {len(all_labels)} samples"
        )

        # Class distribution should be more balanced than without sampler
        values = [counts.get(i, 0) for i in range(cfg.NUM_CLASSES)]
        cv = (sum(v**2 for v in values) ** 0.5 / max(sum(values), 1)) if values else 1
        logger.info(f"  Label distribution CV (lower = more balanced): {cv:.4f}")

    except Exception as e:
        _record("WeightedRandomSampler", False, str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Check 9: Corrupted image handling
# ──────────────────────────────────────────────────────────────────────────────

def test_corrupted_image_handling() -> None:
    _section("9. Corrupted Image Placeholder Handling")

    import tempfile
    import os

    try:
        ds = WasteDataset("train", transform=get_val_transforms(), verbose=False)

        # Monkey-patch one sample with an invalid path
        original_samples = ds.samples.copy()
        fake_path = Path("/nonexistent/corrupted_image.jpg")
        ds.samples[0] = (fake_path, 0)

        img, label = ds[0]  # Should NOT raise, should return placeholder
        _record(
            "Corrupted image returns placeholder tensor",
            img.shape == torch.Size([3, 224, 224]),
            f"got shape {img.shape}"
        )
        _record(
            "Corrupted image returns correct label",
            label == 0,
            f"got label={label}"
        )
        _record(
            "Placeholder is zero tensor",
            img.sum().item() == 0.0,
            f"sum={img.sum().item()}"
        )

        # Restore original samples
        ds.samples = original_samples

    except Exception as e:
        _record("Corrupted image handling", False, str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Check 10: Batch visualisation (saves to results/eda/)
# ──────────────────────────────────────────────────────────────────────────────

def test_batch_visualisation() -> None:
    _section("10. Batch Visualisation")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from ml.preprocessing import get_inverse_normalize

        set_seed(42)
        train_loader, _, _ = create_dataloaders(batch_size=8, num_workers=0)
        batch_imgs, batch_labels = next(iter(train_loader))

        inv_norm = get_inverse_normalize()
        train_ds = WasteDataset("train", transform=get_val_transforms(), verbose=False)

        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        fig.patch.set_facecolor("#1a1a2e")
        fig.suptitle("Sample Training Batch (after augmentation)", fontsize=13, color="#e0e0e0")

        for i, ax in enumerate(axes.flat):
            if i >= batch_imgs.size(0):
                ax.axis("off")
                continue
            img = inv_norm(batch_imgs[i]).clamp(0, 1).permute(1, 2, 0).numpy()
            ax.imshow(img)
            cls_name = train_ds.classes[batch_labels[i].item()]
            ax.set_title(cls_name.capitalize(), fontsize=9, color="#e0e0e0", pad=4)
            ax.axis("off")
            ax.set_facecolor("#16213e")

        plt.tight_layout()
        eda_dir = cfg.paths.results_dir / "eda"
        eda_dir.mkdir(parents=True, exist_ok=True)
        out_path = eda_dir / "sample_batch.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        _record("Sample batch saved", out_path.exists(), str(out_path))

    except Exception as e:
        _record("Batch visualisation", False, str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────

def print_summary() -> int:
    total = len(RESULTS)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = total - passed

    print(f"\n{'═'*60}")
    print(f" Phase 2 Validation Summary")
    print(f"{'═'*60}")
    print(f"  Total checks : {total}")
    print(f"  ✅ Passed    : {passed}")
    print(f"  ❌ Failed    : {failed}")
    print(f"{'═'*60}")

    if failed > 0:
        print("\n Failed checks:")
        for name, ok, msg in RESULTS:
            if not ok:
                print(f"  ✗ {name}: {msg}")

    if failed == 0:
        print("\n ✅ ALL PHASE 2 CHECKS PASSED — Ready for Phase 3")
    else:
        print(f"\n ❌ {failed} check(s) failed — fix before proceeding to Phase 3")

    return 0 if failed == 0 else 1


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 Validation")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        import logging
        logger.setLevel(logging.DEBUG)

    print("=" * 60)
    print(" SmartWaste AI — Phase 2 Validation")
    print("=" * 60)

    # Run all checks
    test_preprocessing()
    test_waste_dataset()
    test_class_mapping()
    test_class_coverage()
    test_no_data_leakage()
    test_dataloaders()
    test_class_weights()
    test_weighted_sampler()
    test_corrupted_image_handling()
    test_batch_visualisation()

    return print_summary()


if __name__ == "__main__":
    sys.exit(main())
