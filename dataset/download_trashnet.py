"""
dataset/download_trashnet.py
─────────────────────────────────────────────────────────────────────────────
SmartWaste AI — Reproducible TrashNet Dataset Downloader

What this script does:
  1. Downloads the TrashNet dataset from the official Stanford/Gary Thung
     Google Drive link (no Kaggle account required).
  2. Extracts the ZIP archive.
  3. Reorganises images into  dataset/processed/{train,val,test}/{class}/
     using a stratified 70/15/15 split (seed=42).
  4. Prints a summary of image counts per split per class.
  5. Saves the split manifest to  results/reports/split_manifest.json

Usage:
    # Standard (auto-downloads):
    python dataset/download_trashnet.py

    # If you already have the ZIP at a custom path:
    python dataset/download_trashnet.py --zip /path/to/dataset-resized.zip

    # Google Colab (run as a cell):
    # !python dataset/download_trashnet.py

License note:
    TrashNet is released under the MIT License by Gary Thung & Mindy Yang.
    https://github.com/garythung/trashnet
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Add project root to sys.path so ml.config is importable ──────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import cfg  # noqa: E402
from ml.utils import get_logger, set_seed  # noqa: E402

logger = get_logger("download_trashnet")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Primary download source: Google Drive (official TrashNet resized dataset)
# File: dataset-resized.zip  (~150 MB)
GDRIVE_FILE_ID = "0B3P9oO5A3ReSWDJoTE1mam1HMGM"

# Fallback: HuggingFace mirror (more reliable, same content)
HF_DOWNLOAD_URL = (
    "https://huggingface.co/datasets/garythung/trashnet/resolve/main/"
    "dataset-resized.zip"
)

# Expected class names (alphabetical order = ImageFolder class indices)
EXPECTED_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Minimum images per class required to pass verification
MIN_IMAGES_PER_CLASS = 100

# ──────────────────────────────────────────────────────────────────────────────
# Download helpers
# ──────────────────────────────────────────────────────────────────────────────


def _gdrive_download(file_id: str, dest: Path) -> bool:
    """
    Attempt to download a file from Google Drive using gdown.

    Returns True on success, False if gdown is not installed or fails.
    """
    try:
        import gdown  # type: ignore
    except ImportError:
        logger.warning(
            "gdown is not installed. Install it with: pip install gdown"
        )
        return False

    url = f"https://drive.google.com/uc?id={file_id}"
    logger.info(f"Downloading from Google Drive: {url}")
    try:
        gdown.download(url, str(dest), quiet=False, fuzzy=True)
        return dest.exists() and dest.stat().st_size > 1_000_000
    except Exception as e:
        logger.error(f"Google Drive download failed: {e}")
        return False


def _http_download(url: str, dest: Path) -> bool:
    """
    Download a file over HTTP using the requests library with a progress bar.

    Returns True on success, False on failure.
    """
    try:
        import requests  # type: ignore
        from tqdm import tqdm  # type: ignore
    except ImportError:
        logger.warning(
            "requests and tqdm are needed. Run: pip install requests tqdm"
        )
        return False

    logger.info(f"Downloading from: {url}")
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        total_bytes = int(response.headers.get("content-length", 0))

        with open(dest, "wb") as fh:
            with tqdm(
                total=total_bytes,
                unit="B",
                unit_scale=True,
                desc=dest.name,
                ncols=80,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    fh.write(chunk)
                    pbar.update(len(chunk))

        return dest.exists() and dest.stat().st_size > 1_000_000
    except Exception as e:
        logger.error(f"HTTP download failed: {e}")
        return False


def download_dataset(zip_dest: Path) -> Path:
    """
    Download the TrashNet ZIP archive if it does not already exist locally.

    Tries in order:
      1. gdown (Google Drive) — fastest, official source
      2. HuggingFace HTTP mirror — reliable fallback

    Returns the path to the downloaded ZIP file.
    Raises RuntimeError if all download methods fail.
    """
    if zip_dest.exists() and zip_dest.stat().st_size > 1_000_000:
        logger.info(f"ZIP already present at {zip_dest}, skipping download.")
        return zip_dest

    zip_dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Starting dataset download…")

    # Try gdown first
    if _gdrive_download(GDRIVE_FILE_ID, zip_dest):
        logger.info("Downloaded via Google Drive.")
        return zip_dest

    # Fallback to HuggingFace
    logger.info("Trying HuggingFace mirror…")
    if _http_download(HF_DOWNLOAD_URL, zip_dest):
        logger.info("Downloaded via HuggingFace.")
        return zip_dest

    raise RuntimeError(
        "\n"
        "Automatic download failed. Please download TrashNet manually:\n\n"
        "  Option A — Google Drive:\n"
        "    https://drive.google.com/file/d/0B3P9oO5A3ReSWDJoTE1mam1HMGM\n\n"
        "  Option B — Kaggle (same data, different host):\n"
        "    https://www.kaggle.com/datasets/fedesoriano/garbage-classification-dataset\n\n"
        "  After downloading, run:\n"
        f"    python dataset/download_trashnet.py --zip /path/to/dataset-resized.zip\n"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Extraction helper
# ──────────────────────────────────────────────────────────────────────────────


def extract_zip(zip_path: Path, extract_to: Path) -> Path:
    """
    Extract the ZIP archive and return the path to the top-level data folder.

    TrashNet ZIP structure:
        dataset-resized/
            cardboard/  glass/  metal/  paper/  plastic/  trash/

    Returns the path to the extracted 'dataset-resized/' folder.
    """
    logger.info(f"Extracting {zip_path.name} → {extract_to} …")
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Inspect the archive to find the data root
        members = zf.namelist()
        logger.info(f"ZIP contains {len(members)} entries.")
        zf.extractall(extract_to)

    # Find the directory that contains the class folders
    # (the ZIP may contain a wrapper directory)
    for candidate in sorted(extract_to.iterdir()):
        if candidate.is_dir():
            class_subdirs = [
                d for d in candidate.iterdir()
                if d.is_dir() and d.name in EXPECTED_CLASSES
            ]
            if class_subdirs:
                logger.info(f"Data root found: {candidate}")
                return candidate

    raise FileNotFoundError(
        f"Could not find class folders inside the extracted archive at {extract_to}.\n"
        f"Expected subdirectories: {EXPECTED_CLASSES}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Stratified split
# ──────────────────────────────────────────────────────────────────────────────


def stratified_split(
    data_root: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Dict[str, List[Path]]]:
    """
    Collect all images per class and perform a stratified 70/15/15 split.

    Why stratified?
      With only 137 'trash' images, a random split could place almost all
      trash images in one split, making evaluation meaningless.  Stratified
      splitting guarantees each split has the same class proportion.

    Args:
        data_root:   Path to the extracted dataset root (contains class folders).
        train_ratio: Fraction of images for training (default 0.70).
        val_ratio:   Fraction for validation (default 0.15).
                     The remaining fraction goes to test.
        seed:        Random seed for reproducibility.

    Returns:
        Nested dict: { 'train': {'cardboard': [Path, ...], ...}, 'val': {...}, 'test': {...} }
    """
    set_seed(seed)
    test_ratio = round(1.0 - train_ratio - val_ratio, 10)
    logger.info(
        f"Split ratios → train:{train_ratio:.0%}  "
        f"val:{val_ratio:.0%}  test:{test_ratio:.0%}  (seed={seed})"
    )

    splits: Dict[str, Dict[str, List[Path]]] = {
        "train": defaultdict(list),
        "val":   defaultdict(list),
        "test":  defaultdict(list),
    }

    for class_name in EXPECTED_CLASSES:
        class_dir = data_root / class_name
        if not class_dir.exists():
            logger.warning(f"Class directory not found: {class_dir}")
            continue

        # Gather all image files, sorted for reproducibility
        images: List[Path] = sorted(
            p for p in class_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        )

        if not images:
            logger.warning(f"No images found in {class_dir}")
            continue

        # Shuffle with fixed seed
        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        # Remaining images go to test (avoids rounding errors)
        n_test  = n - n_train - n_val

        splits["train"][class_name] = images[:n_train]
        splits["val"][class_name]   = images[n_train : n_train + n_val]
        splits["test"][class_name]  = images[n_train + n_val :]

        logger.info(
            f"  {class_name:<12}  total={n:>4}  "
            f"train={n_train:>3}  val={n_val:>3}  test={n_test:>3}"
        )

    return splits


# ──────────────────────────────────────────────────────────────────────────────
# File copy / symlink into processed split dirs
# ──────────────────────────────────────────────────────────────────────────────


def copy_splits_to_processed(
    splits: Dict[str, Dict[str, List[Path]]],
    processed_root: Path,
    overwrite: bool = False,
) -> None:
    """
    Copy images from the raw extracted directory into the processed split
    directories expected by torchvision.ImageFolder:

        processed/
          train/
            cardboard/  glass/  metal/  paper/  plastic/  trash/
          val/
            cardboard/  ...
          test/
            cardboard/  ...

    Args:
        splits:         Nested dict from stratified_split().
        processed_root: Path to dataset/processed/.
        overwrite:      If False (default), skip copying if destination exists.
    """
    total_copied = 0
    total_skipped = 0

    for split_name, class_dict in splits.items():
        for class_name, image_paths in class_dict.items():
            dest_dir = processed_root / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for src_path in image_paths:
                dest_path = dest_dir / src_path.name
                if dest_path.exists() and not overwrite:
                    total_skipped += 1
                    continue
                shutil.copy2(src_path, dest_path)
                total_copied += 1

    logger.info(
        f"File copy complete: {total_copied} copied, {total_skipped} already existed."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Manifest (audit trail)
# ──────────────────────────────────────────────────────────────────────────────


def save_manifest(
    splits: Dict[str, Dict[str, List[Path]]],
    manifest_path: Path,
) -> None:
    """
    Save a JSON manifest recording which files are in each split.

    This provides a full audit trail — the manifest can be re-read to
    reproduce the exact same split on any machine, even if images are
    re-downloaded.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Build JSON-serialisable structure
    manifest: Dict = {
        "description": "SmartWaste AI — TrashNet dataset split manifest",
        "split_ratios": {
            "train": cfg.dataset.TRAIN_RATIO,
            "val":   cfg.dataset.VAL_RATIO,
            "test":  cfg.dataset.TEST_RATIO,
        },
        "seed": cfg.dataset.SPLIT_SEED,
        "classes": EXPECTED_CLASSES,
        "splits": {},
    }

    total_summary: Dict[str, int] = {}
    for split_name, class_dict in splits.items():
        manifest["splits"][split_name] = {}
        split_total = 0
        for class_name, paths in class_dict.items():
            filenames = [p.name for p in paths]
            manifest["splits"][split_name][class_name] = filenames
            split_total += len(filenames)
        manifest["splits"][split_name]["__total__"] = split_total
        total_summary[split_name] = split_total

    manifest["summary"] = total_summary
    manifest["total_images"] = sum(total_summary.values())

    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    logger.info(f"Split manifest saved → {manifest_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Summary printer
# ──────────────────────────────────────────────────────────────────────────────


def print_summary(splits: Dict[str, Dict[str, List[Path]]]) -> None:
    """Print a formatted summary table of the split distribution."""
    header = f"\n{'Class':<14} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>7}"
    separator = "─" * len(header)
    print(separator)
    print(" SmartWaste AI — TrashNet Dataset Split Summary")
    print(separator)
    print(header)
    print(separator)

    grand_train = grand_val = grand_test = 0
    for class_name in EXPECTED_CLASSES:
        n_train = len(splits["train"].get(class_name, []))
        n_val   = len(splits["val"].get(class_name, []))
        n_test  = len(splits["test"].get(class_name, []))
        n_total = n_train + n_val + n_test

        grand_train += n_train
        grand_val   += n_val
        grand_test  += n_test

        bar_len = max(1, int(n_total / 20))   # rough visual bar
        bar = "█" * bar_len

        print(f"  {class_name:<12} {n_train:>6} {n_val:>6} {n_test:>6} {n_total:>7}  {bar}")

    grand_total = grand_train + grand_val + grand_test
    print(separator)
    print(f"  {'TOTAL':<12} {grand_train:>6} {grand_val:>6} {grand_test:>6} {grand_total:>7}")
    print(separator)
    print()

    # Warn about class imbalance
    class_totals = {
        cls: len(splits["train"].get(cls, []))
            + len(splits["val"].get(cls, []))
            + len(splits["test"].get(cls, []))
        for cls in EXPECTED_CLASSES
    }
    min_cls = min(class_totals, key=class_totals.get)
    max_cls = max(class_totals, key=class_totals.get)
    imbalance_ratio = class_totals[max_cls] / max(class_totals[min_cls], 1)

    if imbalance_ratio > 2.0:
        print(
            f"  ⚠  Class imbalance detected!\n"
            f"     Largest class : {max_cls} ({class_totals[max_cls]} images)\n"
            f"     Smallest class: {min_cls} ({class_totals[min_cls]} images)\n"
            f"     Imbalance ratio: {imbalance_ratio:.1f}×\n"
            f"     → Will apply CrossEntropyLoss(weight=...) and "
            f"WeightedRandomSampler during training.\n"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────


def main(custom_zip: Optional[Path] = None, overwrite: bool = False) -> None:
    logger.info("=" * 60)
    logger.info(" SmartWaste AI — TrashNet Dataset Downloader")
    logger.info("=" * 60)

    zip_dest = cfg.paths.raw_data / "dataset-resized.zip"
    extract_dest = cfg.paths.raw_data / "extracted"

    # Step 1: Download
    if custom_zip:
        zip_path = Path(custom_zip)
        if not zip_path.exists():
            raise FileNotFoundError(f"Custom ZIP not found: {custom_zip}")
        logger.info(f"Using provided ZIP: {zip_path}")
    else:
        zip_path = download_dataset(zip_dest)

    # Step 2: Extract
    data_root = extract_zip(zip_path, extract_dest)

    # Step 3: Stratified split
    logger.info("\nComputing stratified train/val/test split…")
    splits = stratified_split(
        data_root=data_root,
        train_ratio=cfg.dataset.TRAIN_RATIO,
        val_ratio=cfg.dataset.VAL_RATIO,
        seed=cfg.dataset.SPLIT_SEED,
    )

    # Step 4: Copy into processed/
    logger.info("\nCopying images to dataset/processed/ …")
    copy_splits_to_processed(
        splits=splits,
        processed_root=cfg.paths.processed_data,
        overwrite=overwrite,
    )

    # Step 5: Save audit manifest
    manifest_path = cfg.paths.reports_dir / "split_manifest.json"
    save_manifest(splits, manifest_path)

    # Step 6: Print summary
    print_summary(splits)

    logger.info("Dataset preparation complete.")
    logger.info(f"Processed data  → {cfg.paths.processed_data}")
    logger.info(f"Split manifest  → {manifest_path}")
    logger.info("\nNext step: python dataset/verify_dataset.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and prepare the TrashNet dataset for SmartWaste AI."
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=None,
        help="Path to an already-downloaded dataset-resized.zip file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-copy files even if destination already exists.",
    )
    args = parser.parse_args()
    main(custom_zip=args.zip, overwrite=args.overwrite)
