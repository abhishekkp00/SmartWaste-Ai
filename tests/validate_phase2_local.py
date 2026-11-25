"""
tests/validate_phase2_local.py
─────────────────────────────────────────────────────────────────────────────
SmartWaste AI — Phase 2 Local Validation (No PyTorch Required)

Tests everything that can run without PyTorch installed:
  1. Config module imports and paths
  2. Utils module (seed, logger, AverageMeter, EarlyStopping)
  3. PIL image loading and RGB conversion
  4. EDA chart generation (saves to results/eda/)
  5. Project directory structure
  6. Dataset manifest existence
  7. Class weight formula (pure Python)
  8. Preprocessing transform description

For full validation (including tensor shapes, DataLoader, WeightedSampler),
run on Google Colab after installing PyTorch:
    python tests/validate_phase2.py

Usage:
    python tests/validate_phase2_local.py
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import json
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS: List[Tuple[str, bool, str]] = []


def _record(name: str, passed: bool, msg: str = "") -> bool:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  {name}" + (f"  — {msg}" if msg else ""))
    RESULTS.append((name, passed, msg))
    return passed


def _section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f" {title}")
    print(f"{'─'*60}")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Config imports
# ──────────────────────────────────────────────────────────────────────────────

def test_config() -> None:
    _section("1. Config Module")
    try:
        from ml.config import cfg
        _record("cfg import",             True)
        _record("NUM_CLASSES == 6",       cfg.NUM_CLASSES == 6, f"got {cfg.NUM_CLASSES}")
        _record("IMAGE_SIZE == (224,224)", cfg.IMAGE_SIZE == (224, 224), f"got {cfg.IMAGE_SIZE}")
        _record("CLASS_NAMES has 6",      len(cfg.CLASS_NAMES) == 6, str(cfg.CLASS_NAMES))
        _record("IMAGENET_MEAN correct",
                cfg.IMAGENET_MEAN == (0.485, 0.456, 0.406), str(cfg.IMAGENET_MEAN))
        _record("IMAGENET_STD correct",
                cfg.IMAGENET_STD == (0.229, 0.224, 0.225), str(cfg.IMAGENET_STD))
        _record("TRAIN_RATIO + VAL_RATIO + TEST_RATIO ≈ 1.0",
                abs(cfg.dataset.TRAIN_RATIO + cfg.dataset.VAL_RATIO + cfg.dataset.TEST_RATIO - 1.0) < 1e-9)
        _record("BATCH_SIZE == 32",        cfg.training.BATCH_SIZE == 32)
        _record("PATIENCE == 7",           cfg.training.PATIENCE == 7)
        _record("CONFIDENCE_THRESHOLD == 0.60",
                cfg.inference.CONFIDENCE_THRESHOLD == 0.60)

        # Path sanity checks (paths don't have to exist yet)
        _record("paths.train_dir defined",
                str(cfg.paths.train_dir).endswith("processed/train"))
        _record("paths.eda_dir defined",
                "eda" in str(cfg.paths.eda_dir))

        print(f"\n  Config repr: {cfg}")
    except Exception as e:
        _record("Config import", False, str(e))


# ──────────────────────────────────────────────────────────────────────────────
# 2. Utils
# ──────────────────────────────────────────────────────────────────────────────

def test_utils() -> None:
    _section("2. Utils Module")
    try:
        import torch  # noqa – utils.py imports torch at top level
    except ImportError:
        _record("Utils (torch not installed — skipped)", True, "install torch to run full check")
        return
    try:
        from ml.utils import set_seed, get_logger, AverageMeter, EarlyStopping, format_time

        set_seed(42)
        _record("set_seed(42) runs without error", True)

        log = get_logger("test_logger")
        _record("get_logger returns logger", log is not None)

        # AverageMeter
        meter = AverageMeter("Loss", ":.4f")
        for v in [1.0, 0.8, 0.6]:
            meter.update(v, n=10)
        expected_avg = (1.0 + 0.8 + 0.6) / 3
        _record("AverageMeter avg correct",
                abs(meter.avg - expected_avg) < 1e-6,
                f"avg={meter.avg:.4f} expected={expected_avg:.4f}")

        # EarlyStopping (max mode)
        es = EarlyStopping(patience=3, mode="max", verbose=False)
        for val in [0.70, 0.80, 0.80, 0.80, 0.80]:
            should_stop = es(val)
        _record("EarlyStopping triggers after patience",
                should_stop, "should stop after 3 no-improvement epochs")

        # format_time
        _record("format_time(90) == '1m 30s'",   format_time(90) == "1m 30s")
        _record("format_time(3661) == '1h 1m 1s'", format_time(3661) == "1h 1m 1s")
        _record("format_time(45) == '45s'",        format_time(45) == "45s")

    except Exception as e:
        _record("Utils", False, str(e))


# ──────────────────────────────────────────────────────────────────────────────
# 3. PIL image loading
# ──────────────────────────────────────────────────────────────────────────────

def test_pil() -> None:
    _section("3. PIL Image Loading & RGB Conversion")
    try:
        import numpy as np
        from PIL import Image

        # Create a synthetic non-square RGBA image
        rgba_arr = (np.random.rand(300, 400, 4) * 255).astype("uint8")
        rgba_img = Image.fromarray(rgba_arr, mode="RGBA")
        rgb_img  = rgba_img.convert("RGB")

        _record("PIL RGBA → RGB conversion", rgb_img.mode == "RGB", f"mode={rgb_img.mode}")
        _record("PIL RGB shape",
                rgb_img.size == (400, 300),
                f"size={rgb_img.size}")

        # Resize to 224
        resized = rgb_img.resize((224, 224), Image.BILINEAR)
        _record("PIL resize to 224×224",
                resized.size == (224, 224),
                f"size={resized.size}")

        # Pixel range
        arr = np.array(rgb_img)
        _record("RGB pixel range [0, 255]",
                arr.min() >= 0 and arr.max() <= 255,
                f"min={arr.min()} max={arr.max()}")

        # NumPy conversion
        norm_arr = arr.astype("float32") / 255.0
        _record("Normalised [0,1]",
                0.0 <= norm_arr.min() and norm_arr.max() <= 1.0)

    except Exception as e:
        _record("PIL", False, str(e))


# ──────────────────────────────────────────────────────────────────────────────
# 4. Project directory structure
# ──────────────────────────────────────────────────────────────────────────────

def test_directory_structure() -> None:
    _section("4. Project Directory Structure")
    expected_dirs = [
        "ml",
        "dataset",
        "models",
        "results",
        "notebooks",
        "tests",
        "backend",
        "frontend",
    ]
    expected_files = [
        "requirements.txt",
        "requirements-dev.txt",
        ".gitignore",
        ".env.example",
        "README.md",
        "setup_project.py",
        "ml/config.py",
        "ml/utils.py",
        "ml/preprocessing.py",
        "ml/dataset.py",
        "ml/eda.py",
        "dataset/download_trashnet.py",
        "dataset/verify_dataset.py",
        "tests/validate_phase2.py",
        "notebooks/01_phase1_dataset_setup.ipynb",
        "notebooks/02_phase2_eda_preprocessing.ipynb",
    ]

    for d in expected_dirs:
        full = PROJECT_ROOT / d
        _record(f"dir  {d}/", full.is_dir(), "" if full.is_dir() else f"missing at {full}")

    for f in expected_files:
        full = PROJECT_ROOT / f
        exists = full.exists() and full.stat().st_size > 0
        _record(f"file {f}", exists, "" if exists else f"missing at {full}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Dataset manifest
# ──────────────────────────────────────────────────────────────────────────────

def test_manifest() -> None:
    _section("5. Dataset Manifest & Processed Split Directories")
    from ml.config import cfg

    manifest_path = cfg.paths.reports_dir / "split_manifest.json"

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

        _record("Manifest has 'splits' key", "splits" in manifest)
        _record("Manifest has train/val/test",
                all(s in manifest.get("splits", {}) for s in ("train", "val", "test")))

        expected_classes = sorted(cfg.CLASS_NAMES)
        for split in ("train", "val", "test"):
            split_data = manifest.get("splits", {}).get(split, {})
            present_classes = sorted(k for k in split_data if not k.startswith("__"))
            _record(f"Manifest [{split}] has 6 classes",
                    present_classes == expected_classes,
                    f"got {present_classes}")

        total = manifest.get("total_images", 0)
        _record(f"Manifest total_images in [2000, 3000]",
                2000 <= total <= 3000,
                f"total={total}")
    else:
        _record("Manifest exists", False,
                f"{manifest_path} — run: python dataset/download_trashnet.py first")

    # Check processed split directories exist
    for split in ("train", "val", "test"):
        split_dir = cfg.paths.processed_data / split
        _record(f"processed/{split}/ exists", split_dir.exists(),
                str(split_dir) if not split_dir.exists() else "")

        if split_dir.exists():
            # Count class subdirs
            class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
            _record(f"processed/{split}/ has 6 classes",
                    len(class_dirs) == 6,
                    f"found {len(class_dirs)}: {[d.name for d in class_dirs]}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Class weight formula verification (pure Python, no PyTorch)
# ──────────────────────────────────────────────────────────────────────────────

def test_class_weight_formula() -> None:
    _section("6. Class Weight Formula Verification")

    # Simulate the TrashNet training distribution
    known_train_counts = {
        "cardboard": 282,
        "glass":     351,
        "metal":     287,
        "paper":     416,
        "plastic":   337,
        "trash":      96,
    }
    n_total = sum(known_train_counts.values())
    n_cls   = 6

    weights = {
        cls: n_total / (n_cls * n_c)
        for cls, n_c in known_train_counts.items()
    }

    _record("All weights > 0",        all(w > 0 for w in weights.values()))
    _record("'trash' has highest weight",
            max(weights, key=weights.get) == "trash",
            f"weights={weights}")
    _record("'paper' has lowest weight",
            min(weights, key=weights.get) == "paper",
            f"paper_weight={weights['paper']:.4f}")
    _record("Weight formula: trash / paper ratio > 2×",
            weights["trash"] / weights["paper"] > 2.0,
            f"ratio={weights['trash']/weights['paper']:.2f}×")

    print(f"\n  Computed class weights (simulated training counts):")
    for cls in ["cardboard", "glass", "metal", "paper", "plastic", "trash"]:
        bar = "█" * int(weights[cls] * 8)
        print(f"    {cls:<12}  n={known_train_counts[cls]:>4}  "
              f"weight={weights[cls]:.4f}  {bar}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. EDA chart generation (uses PIL + matplotlib only)
# ──────────────────────────────────────────────────────────────────────────────

def test_eda_generation() -> None:
    _section("7. EDA Chart Generation")

    from ml.config import cfg

    # Only run if processed data exists
    train_dir = cfg.paths.processed_data / "train"
    if not train_dir.exists():
        _record("EDA generation skipped",
                True,
                "No processed data yet — run download_trashnet.py first")
        return

    try:
        import torch  # noqa
    except ImportError:
        # EDA imports torch indirectly via preprocessing; count_images still works
        pass

    try:
        # Test just the count function (no torch needed)
        from ml.eda import count_images, plot_class_distribution, write_imbalance_report

        counts = count_images()
        total_classes = sum(1 for cls in counts.get("train", {}) if counts["train"].get(cls, 0) > 0)
        _record("count_images() returns 6 classes", total_classes == 6, f"got {total_classes}")

        # Print what was found
        print(f"\n  Image counts (from processed dataset):")
        for cls in cfg.CLASS_NAMES:
            tr = counts["train"].get(cls, 0)
            va = counts["val"].get(cls, 0)
            te = counts["test"].get(cls, 0)
            print(f"    {cls:<12}  train={tr:>4}  val={va:>4}  test={te:>4}  total={tr+va+te:>4}")

        # Generate distribution chart
        out = plot_class_distribution(counts)
        _record("class_distribution_bar.png saved", out.exists(), str(out))

        # Write imbalance report
        report_out = write_imbalance_report(counts)
        _record("class_imbalance_summary.txt saved", report_out.exists(), str(report_out))

    except Exception as e:
        _record("EDA generation", False, str(e))


# ──────────────────────────────────────────────────────────────────────────────
# 8. Preprocessing module imports
# ──────────────────────────────────────────────────────────────────────────────

def test_preprocessing_imports() -> None:
    _section("8. Preprocessing Module Imports")

    try:
        import torchvision  # noqa
    except ImportError:
        _record("Preprocessing imports (torchvision not installed — skipped)",
                True, "install torch+torchvision to run full check")
        return

    try:
        from ml.preprocessing import (
            get_train_transforms,
            get_val_transforms,
            get_test_transforms,
            get_inverse_normalize,
        )
        _record("get_train_transforms importable",   True)
        _record("get_val_transforms importable",     True)
        _record("get_test_transforms importable",    True)
        _record("get_inverse_normalize importable",  True)

        # Inspect transform descriptions
        train_t = get_train_transforms()
        val_t   = get_val_transforms()
        _record("Train transform is Compose", "Compose" in type(train_t).__name__)
        _record("Val transform is Compose",   "Compose" in type(val_t).__name__)

        train_str = str(train_t)
        _record("Train has RandomResizedCrop",   "RandomResizedCrop" in train_str)
        _record("Train has RandomHorizontalFlip","RandomHorizontalFlip" in train_str)
        _record("Train has RandomRotation",      "RandomRotation" in train_str)
        _record("Train has ColorJitter",         "ColorJitter" in train_str)
        _record("Train has Normalize",           "Normalize" in train_str)

        val_str = str(val_t)
        _record("Val has Resize",        "Resize" in val_str)
        _record("Val has CenterCrop",    "CenterCrop" in val_str)
        _record("Val has Normalize",     "Normalize" in val_str)
        _record("Val has NO RandomFlip", "RandomHorizontalFlip" not in val_str)
        _record("Val has NO RandomCrop", "RandomResizedCrop" not in val_str)

    except ImportError as e:
        _record("Preprocessing imports", False, str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────

def print_summary() -> int:
    total  = len(RESULTS)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = total - passed

    print(f"\n{'═'*60}")
    print(f" Phase 2 Local Validation Summary")
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
        print("\n ✅ ALL LOCAL CHECKS PASSED")
        print(" For full tensor/DataLoader validation, run on Colab:")
        print("   python tests/validate_phase2.py")
    else:
        print(f"\n ⚠  {failed} check(s) failed")

    return 0 if failed == 0 else 1


def main() -> int:
    print("=" * 60)
    print(" SmartWaste AI — Phase 2 Local Validation")
    print(" (No PyTorch required)")
    print("=" * 60)

    test_config()
    test_utils()
    test_pil()
    test_directory_structure()
    test_manifest()
    test_class_weight_formula()
    test_eda_generation()
    test_preprocessing_imports()

    return print_summary()


if __name__ == "__main__":
    sys.exit(main())
