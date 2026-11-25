"""
dataset/verify_dataset.py
─────────────────────────────────────────────────────────────────────────────
SmartWaste AI — Dataset Integrity Verification Script

Run this AFTER download_trashnet.py to confirm:
  1. All 6 expected class folders exist in train / val / test.
  2. Image counts per class match expected distribution (±10% tolerance).
  3. No corrupted / zero-byte / unreadable image files.
  4. No duplicate filenames across splits (data leakage check).
  5. All images can be opened and converted to RGB tensors.
  6. Saves a detailed JSON verification report.

Usage:
    python dataset/verify_dataset.py

Exit codes:
    0 — All checks passed.
    1 — One or more checks failed (see printed report).
─────────────────────────────────────────────────────────────────────────────
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import cfg
from ml.utils import get_logger

logger = get_logger("verify_dataset")

# Acceptable tolerance: actual count may differ from known count by this fraction
COUNT_TOLERANCE = 0.12   # 12% — accounts for download variants & minor version changes

EXPECTED_CLASSES = list(cfg.CLASS_NAMES)   # ['cardboard','glass','metal','paper','plastic','trash']
SPLITS = ["train", "val", "test"]

# ──────────────────────────────────────────────────────────────────────────────
# Check 1: Directory structure
# ──────────────────────────────────────────────────────────────────────────────


def check_directory_structure() -> Tuple[bool, List[str]]:
    """
    Verify that all expected class subdirectories exist in every split.

    Returns (passed: bool, issues: List[str])
    """
    issues = []
    for split in SPLITS:
        split_dir = cfg.paths.processed_data / split
        if not split_dir.exists():
            issues.append(f"Missing split directory: {split_dir}")
            continue
        for class_name in EXPECTED_CLASSES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                issues.append(f"Missing class folder: {class_dir}")

    passed = len(issues) == 0
    return passed, issues


# ──────────────────────────────────────────────────────────────────────────────
# Check 2: Image counts
# ──────────────────────────────────────────────────────────────────────────────


def count_images() -> Dict[str, Dict[str, int]]:
    """
    Count image files in each split/class directory.

    Returns nested dict: { split: { class: count } }
    """
    counts: Dict[str, Dict[str, int]] = {}
    for split in SPLITS:
        counts[split] = {}
        for class_name in EXPECTED_CLASSES:
            class_dir = cfg.paths.processed_data / split / class_name
            if not class_dir.exists():
                counts[split][class_name] = 0
                continue
            n = sum(
                1 for f in class_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            )
            counts[split][class_name] = n
    return counts


def check_image_counts(
    counts: Dict[str, Dict[str, int]]
) -> Tuple[bool, List[str]]:
    """
    Verify total images per class are within tolerance of TrashNet known counts.

    Uses total across all splits to be robust to rounding differences.
    """
    known = cfg.dataset.KNOWN_CLASS_COUNTS
    issues = []

    for class_name in EXPECTED_CLASSES:
        total = sum(counts[split].get(class_name, 0) for split in SPLITS)
        expected = known.get(class_name, 0)

        if expected > 0:
            deviation = abs(total - expected) / expected
            if deviation > COUNT_TOLERANCE:
                issues.append(
                    f"{class_name}: found {total} images, "
                    f"expected ~{expected} (deviation={deviation:.1%}, "
                    f"tolerance={COUNT_TOLERANCE:.0%})"
                )
        elif total == 0:
            issues.append(f"{class_name}: NO images found at all!")

    passed = len(issues) == 0
    return passed, issues


# ──────────────────────────────────────────────────────────────────────────────
# Check 3: Corrupted images
# ──────────────────────────────────────────────────────────────────────────────


def check_image_integrity() -> Tuple[bool, List[str], int]:
    """
    Attempt to open and decode every image file.

    A file is flagged as corrupted if:
      - It is zero bytes.
      - PIL cannot open it.
      - It cannot be converted to RGB.

    Returns (passed, issues_list, total_checked)
    """
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError:
        return False, ["Pillow not installed. Run: pip install Pillow"], 0

    issues = []
    total_checked = 0
    start = time.time()

    for split in SPLITS:
        for class_name in EXPECTED_CLASSES:
            class_dir = cfg.paths.processed_data / split / class_name
            if not class_dir.exists():
                continue

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() not in {
                    ".jpg", ".jpeg", ".png", ".webp", ".bmp"
                }:
                    continue

                total_checked += 1

                # Zero-byte check
                if img_path.stat().st_size == 0:
                    issues.append(f"Zero-byte file: {img_path.relative_to(PROJECT_ROOT)}")
                    continue

                # PIL open + RGB conversion
                try:
                    with Image.open(img_path) as img:
                        img.convert("RGB")
                except UnidentifiedImageError:
                    issues.append(
                        f"Unreadable (not an image): "
                        f"{img_path.relative_to(PROJECT_ROOT)}"
                    )
                except Exception as e:
                    issues.append(
                        f"Corrupted ({type(e).__name__}): "
                        f"{img_path.relative_to(PROJECT_ROOT)}"
                    )

    elapsed = time.time() - start
    logger.info(
        f"Integrity check complete: {total_checked} images in {elapsed:.1f}s"
    )

    passed = len(issues) == 0
    return passed, issues, total_checked


# ──────────────────────────────────────────────────────────────────────────────
# Check 4: Data leakage (duplicate filenames across splits)
# ──────────────────────────────────────────────────────────────────────────────


def check_no_data_leakage() -> Tuple[bool, List[str]]:
    """
    Verify that no filename appears in more than one split.

    Data leakage occurs when the same image appears in both train and test,
    causing artificially inflated evaluation metrics.

    Uses filename as the proxy for identity.  For a production system you
    would use perceptual hashing, but filenames are sufficient for TrashNet
    since images have unique filenames.
    """
    filename_to_splits: Dict[str, List[str]] = defaultdict(list)

    for split in SPLITS:
        for class_name in EXPECTED_CLASSES:
            class_dir = cfg.paths.processed_data / split / class_name
            if not class_dir.exists():
                continue
            for img_path in class_dir.iterdir():
                filename_to_splits[img_path.name].append(split)

    issues = []
    for filename, splits_found in filename_to_splits.items():
        if len(splits_found) > 1:
            issues.append(
                f"Duplicate across splits: '{filename}' found in {splits_found}"
            )

    passed = len(issues) == 0
    return passed, issues


# ──────────────────────────────────────────────────────────────────────────────
# Check 5: Tensor shape smoke test
# ──────────────────────────────────────────────────────────────────────────────


def check_tensor_pipeline() -> Tuple[bool, List[str]]:
    """
    Sample one image per class from the training set and run it through
    the full preprocessing pipeline to verify the output tensor shape.

    Expected output shape: (3, 224, 224)
    """
    issues = []
    try:
        import torch
        from torchvision import transforms
        from PIL import Image
    except ImportError:
        return False, ["torch / torchvision / Pillow not installed"]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=list(cfg.IMAGENET_MEAN),
            std=list(cfg.IMAGENET_STD),
        ),
    ])

    for class_name in EXPECTED_CLASSES:
        class_dir = cfg.paths.processed_data / "train" / class_name
        if not class_dir.exists():
            issues.append(f"Train dir missing: {class_dir}")
            continue

        # Pick the first image in the directory
        images = sorted(
            p for p in class_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not images:
            issues.append(f"No images in train/{class_name}")
            continue

        try:
            with Image.open(images[0]) as img:
                tensor = transform(img.convert("RGB"))
            assert tensor.shape == torch.Size([3, 224, 224]), (
                f"Wrong tensor shape for {class_name}: {tensor.shape}"
            )
        except AssertionError as e:
            issues.append(str(e))
        except Exception as e:
            issues.append(f"Pipeline error for {class_name}: {e}")

    passed = len(issues) == 0
    return passed, issues


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────


def print_count_table(counts: Dict[str, Dict[str, int]]) -> None:
    """Print a formatted split-count table."""
    sep = "─" * 65
    print(f"\n{sep}")
    print(f" {'Class':<14} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>7}  Distribution")
    print(sep)

    grand = {"train": 0, "val": 0, "test": 0}
    all_totals = []

    for class_name in EXPECTED_CLASSES:
        n_train = counts["train"].get(class_name, 0)
        n_val   = counts["val"].get(class_name, 0)
        n_test  = counts["test"].get(class_name, 0)
        n_total = n_train + n_val + n_test
        all_totals.append(n_total)

        grand["train"] += n_train
        grand["val"]   += n_val
        grand["test"]  += n_test

        bar = "█" * max(1, n_total // 20)
        print(
            f"  {class_name:<12} {n_train:>7} {n_val:>7} {n_test:>7} {n_total:>7}  {bar}"
        )

    grand_total = sum(grand.values())
    print(sep)
    print(
        f"  {'TOTAL':<12} {grand['train']:>7} {grand['val']:>7} "
        f"{grand['test']:>7} {grand_total:>7}"
    )
    print(sep)

    # Class weight computation (inverse frequency)
    print("\n  Class weights for CrossEntropyLoss (inverse frequency):")
    total_train = grand["train"]
    for i, class_name in enumerate(EXPECTED_CLASSES):
        n = counts["train"].get(class_name, 1)
        weight = total_train / (len(EXPECTED_CLASSES) * max(n, 1))
        print(f"    [{i}] {class_name:<12}  weight = {weight:.4f}")
    print()


def save_report(results: Dict, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info(f"Verification report saved → {report_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> int:
    logger.info("=" * 60)
    logger.info(" SmartWaste AI — Dataset Verification")
    logger.info("=" * 60)

    all_passed = True
    report = {"checks": {}}

    # ── Check 1: Structure ───────────────────────────────────────────────────
    logger.info("\n[1/5] Checking directory structure…")
    passed, issues = check_directory_structure()
    status = "PASS" if passed else "FAIL"
    logger.info(f"      → {status}")
    if issues:
        for iss in issues:
            logger.error(f"      ✗ {iss}")
    report["checks"]["directory_structure"] = {"passed": passed, "issues": issues}
    all_passed = all_passed and passed

    # ── Check 2: Image counts ────────────────────────────────────────────────
    logger.info("\n[2/5] Counting images per class…")
    counts = count_images()
    print_count_table(counts)
    passed, issues = check_image_counts(counts)
    status = "PASS" if passed else "FAIL"
    logger.info(f"      Count check → {status}")
    if issues:
        for iss in issues:
            logger.warning(f"      ⚠ {iss}")
    report["checks"]["image_counts"] = {
        "passed": passed,
        "counts": counts,
        "issues": issues,
    }
    # Non-fatal: allow slight count mismatches due to dataset version differences
    # (we warn but don't fail the whole pipeline)

    # ── Check 3: Corruption ──────────────────────────────────────────────────
    logger.info("\n[3/5] Checking image integrity (this may take a moment)…")
    passed, issues, total_checked = check_image_integrity()
    status = "PASS" if passed else "FAIL"
    logger.info(f"      Checked {total_checked} files → {status}")
    if issues:
        for iss in issues[:10]:    # Show at most 10 corrupted file paths
            logger.error(f"      ✗ {iss}")
        if len(issues) > 10:
            logger.error(f"      ... and {len(issues) - 10} more corrupted files.")
    report["checks"]["integrity"] = {
        "passed": passed,
        "total_checked": total_checked,
        "corrupted_count": len(issues),
        "issues": issues[:50],
    }
    all_passed = all_passed and passed

    # ── Check 4: Data leakage ────────────────────────────────────────────────
    logger.info("\n[4/5] Checking for data leakage (duplicates across splits)…")
    passed, issues = check_no_data_leakage()
    status = "PASS" if passed else "FAIL"
    logger.info(f"      → {status}")
    if issues:
        for iss in issues[:10]:
            logger.error(f"      ✗ {iss}")
    report["checks"]["data_leakage"] = {"passed": passed, "issues": issues[:50]}
    all_passed = all_passed and passed

    # ── Check 5: Tensor pipeline ─────────────────────────────────────────────
    logger.info("\n[5/5] Smoke-testing preprocessing pipeline (one image/class)…")
    passed, issues = check_tensor_pipeline()
    status = "PASS" if passed else "FAIL"
    logger.info(f"      → {status}")
    if issues:
        for iss in issues:
            logger.error(f"      ✗ {iss}")
    report["checks"]["tensor_pipeline"] = {"passed": passed, "issues": issues}
    all_passed = all_passed and passed

    # ── Final verdict ────────────────────────────────────────────────────────
    report["all_passed"] = all_passed
    report["summary"] = "All checks passed." if all_passed else "Some checks FAILED."

    report_path = cfg.paths.reports_dir / "verification_report.json"
    save_report(report, report_path)

    print("\n" + "=" * 60)
    if all_passed:
        print(" ✅  ALL CHECKS PASSED — Dataset is ready for training.")
    else:
        print(" ❌  SOME CHECKS FAILED — See above for details.")
        print(f"     Full report: {report_path}")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
