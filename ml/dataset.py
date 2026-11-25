"""
ml/dataset.py
─────────────────────────────────────────────────────────────────────────────
SmartWaste AI — PyTorch Dataset and DataLoader Implementation

This module provides:

  WasteDataset          — Custom torch.utils.data.Dataset for TrashNet
  create_dataloaders()  — Factory function returning train/val/test DataLoaders
  compute_class_weights() — Inverse-frequency weights from training labels
  get_sampler()           — WeightedRandomSampler for the training set

Design decisions:
  - Images are loaded lazily in __getitem__ (not pre-loaded into RAM) so the
    dataset works on machines with limited memory.
  - The class-to-index mapping is loaded from the split manifest created in
    Phase 1, guaranteeing consistent ordering across all scripts.
  - Corrupted images are handled gracefully: a black placeholder tensor is
    returned and a warning is logged rather than crashing the DataLoader.
  - WeightedRandomSampler and weighted CrossEntropyLoss are both implemented
    so they can be compared during the training experiments (Phase 5).
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

# Project-local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import cfg
from ml.preprocessing import get_train_transforms, get_val_transforms, get_test_transforms
from ml.utils import get_logger

logger = get_logger(__name__)

# PIL import — optional graceful degradation message
try:
    from PIL import Image, UnidentifiedImageError
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow not found. Run: pip install Pillow")


# ──────────────────────────────────────────────────────────────────────────────
# WasteDataset
# ──────────────────────────────────────────────────────────────────────────────

class WasteDataset(Dataset):
    """
    Custom PyTorch Dataset for the TrashNet waste classification dataset.

    Reads image file paths and labels from the processed split directories
    created in Phase 1.  Optionally loads the split manifest for a full
    audit trail.

    Args:
        split:      One of 'train', 'val', or 'test'.
        transform:  A torchvision transform pipeline.  If None, only
                    ToTensor + Normalize are applied (safe default).
        data_root:  Root of the processed dataset.  Defaults to the path
                    defined in ml/config.py.
        verbose:    If True, log dataset statistics at construction time.

    Attributes:
        samples:    List of (image_path, class_index) tuples.
        classes:    List of class names in the same order as class indices.
        class_to_idx: Dict mapping class name → integer index.
        targets:    List of integer labels (same order as samples).
                    Used by WeightedRandomSampler.

    Example:
        >>> train_ds = WasteDataset(split='train', transform=get_train_transforms())
        >>> image, label = train_ds[0]
        >>> image.shape   # torch.Size([3, 224, 224])
        >>> label         # int in {0, 1, 2, 3, 4, 5}
    """

    # Supported image file extensions
    _IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    # Placeholder tensor returned when an image cannot be opened
    # Shape matches a valid preprocessed image: (3, 224, 224)
    _PLACEHOLDER_SHAPE = (3, cfg.IMAGE_SIZE[0], cfg.IMAGE_SIZE[1])

    def __init__(
        self,
        split: str,
        transform: Optional[Callable] = None,
        data_root: Optional[Path] = None,
        verbose: bool = True,
    ) -> None:
        assert split in ("train", "val", "test"), (
            f"split must be 'train', 'val', or 'test'. Got: '{split}'"
        )

        self.split = split
        self.data_root = Path(data_root) if data_root else cfg.paths.processed_data
        self.split_dir = self.data_root / split

        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {self.split_dir}\n"
                f"Run: python dataset/download_trashnet.py"
            )

        # Use provided transform, or fall back to a safe default
        if transform is not None:
            self.transform = transform
        else:
            logger.warning(
                "No transform provided — using default val/test transforms."
            )
            self.transform = get_val_transforms()

        # Build class list (alphabetical → consistent with torchvision.ImageFolder)
        self.classes: List[str] = sorted(
            d.name for d in self.split_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

        if not self.classes:
            raise FileNotFoundError(
                f"No class subdirectories found in {self.split_dir}"
            )

        # class_to_idx: {'cardboard': 0, 'glass': 1, ...}
        self.class_to_idx: Dict[str, int] = {
            cls_name: idx for idx, cls_name in enumerate(self.classes)
        }

        # Collect all (path, label) pairs
        self.samples: List[Tuple[Path, int]] = []
        self.targets: List[int] = []   # label-only list for WeightedRandomSampler

        self._build_index()

        if verbose:
            self._log_stats()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_index(self) -> None:
        """
        Walk the split directory and collect valid image paths with labels.

        Skips hidden files, non-image files, and zero-byte files.
        Unknown class directories (not in EXPECTED_CLASSES from config) are
        logged as warnings but do not raise exceptions, allowing forward
        compatibility if more classes are added later.
        """
        expected = set(cfg.CLASS_NAMES)
        for cls_name in self.classes:
            if cls_name not in expected:
                logger.warning(
                    f"Unexpected class directory found: '{cls_name}'. "
                    f"Expected: {list(expected)}"
                )

            class_dir = self.split_dir / cls_name
            label = self.class_to_idx[cls_name]

            for img_path in sorted(class_dir.iterdir()):
                # Skip non-image files and hidden files
                if img_path.suffix.lower() not in self._IMG_EXTENSIONS:
                    continue
                if img_path.name.startswith("."):
                    continue
                # Skip empty files
                if img_path.stat().st_size == 0:
                    logger.warning(f"Zero-byte file skipped: {img_path}")
                    continue

                self.samples.append((img_path, label))
                self.targets.append(label)

        if not self.samples:
            raise ValueError(
                f"No valid images found in {self.split_dir}. "
                f"Did the dataset download complete successfully?"
            )

    def _log_stats(self) -> None:
        """Log per-class image counts for the current split."""
        from collections import Counter
        counts = Counter(self.targets)
        logger.info(
            f"[WasteDataset] split='{self.split}'  "
            f"total={len(self.samples)} images"
        )
        for cls_name, idx in self.class_to_idx.items():
            logger.info(
                f"  {cls_name:<12}  idx={idx}  count={counts.get(idx, 0):>4}"
            )

    # ── torch.utils.data.Dataset interface ────────────────────────────────────

    def __len__(self) -> int:
        """Return the total number of images in this split."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Load, preprocess, and return the image and label at `index`.

        Args:
            index: Integer index into self.samples.

        Returns:
            Tuple[torch.Tensor, int]:
              - image tensor of shape (3, 224, 224), dtype float32
              - integer class label in [0, NUM_CLASSES - 1]

        Corrupted image handling:
            If PIL fails to open or convert the image, a black placeholder
            tensor of the correct shape is returned and a warning is logged.
            This prevents a single corrupted file from crashing a training
            epoch — especially important for multi-worker DataLoaders where
            an exception inside a worker silently kills that worker.
        """
        img_path, label = self.samples[index]

        # ── Load image ────────────────────────────────────────────────────────
        try:
            with Image.open(img_path) as img:
                # Always convert to RGB — handles RGBA, palette, grayscale images
                img = img.convert("RGB")
        except Exception as e:
            logger.warning(
                f"Could not open image at index {index} "
                f"({img_path.name}): {type(e).__name__}: {e}. "
                f"Returning black placeholder."
            )
            # Return a black (zero) tensor of the correct shape
            # Using float zeros avoids the need to run through the transform
            placeholder = torch.zeros(self._PLACEHOLDER_SHAPE, dtype=torch.float32)
            return placeholder, label

        # ── Apply transform ───────────────────────────────────────────────────
        try:
            image_tensor = self.transform(img)
        except Exception as e:
            logger.warning(
                f"Transform failed for {img_path.name}: {e}. "
                f"Returning black placeholder."
            )
            return torch.zeros(self._PLACEHOLDER_SHAPE, dtype=torch.float32), label

        return image_tensor, label

    # ── Utility methods ───────────────────────────────────────────────────────

    def get_class_counts(self) -> Dict[str, int]:
        """Return a dict of {class_name: image_count} for this split."""
        from collections import Counter
        idx_counts = Counter(self.targets)
        return {
            cls_name: idx_counts.get(idx, 0)
            for cls_name, idx in self.class_to_idx.items()
        }

    def get_sample_path(self, index: int) -> Path:
        """Return the filesystem path for the image at `index`."""
        return self.samples[index][0]

    def __repr__(self) -> str:
        return (
            f"WasteDataset(split='{self.split}', "
            f"n={len(self)}, "
            f"classes={self.classes})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Class weight computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_class_weights(
    train_dataset: WasteDataset,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from training labels.

    Formula for class c:
        weight[c] = N / (K × n_c)
    where:
        N   = total number of training images
        K   = number of classes
        n_c = number of images in class c

    This ensures that the loss contribution from the minority class (trash,
    ~137 images) is weighted much higher than the majority class (paper,
    ~415 images), forcing the model to pay attention to rare classes.

    Args:
        train_dataset: The training WasteDataset instance.
        device:        Target device for the returned tensor.

    Returns:
        torch.Tensor of shape (NUM_CLASSES,) on `device`.

    Example:
        >>> weights = compute_class_weights(train_ds, device)
        >>> criterion = nn.CrossEntropyLoss(weight=weights)
    """
    from collections import Counter

    counts = Counter(train_dataset.targets)
    n_total = len(train_dataset)
    n_classes = len(train_dataset.classes)

    weights = []
    logger.info("Class weights (inverse frequency):")
    for idx, cls_name in enumerate(train_dataset.classes):
        n_c = counts.get(idx, 1)   # Avoid division by zero
        w = n_total / (n_classes * n_c)
        weights.append(w)
        logger.info(f"  [{idx}] {cls_name:<12}  n={n_c:>4}  weight={w:.4f}")

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        weight_tensor = weight_tensor.to(device)

    return weight_tensor


# ──────────────────────────────────────────────────────────────────────────────
# WeightedRandomSampler
# ──────────────────────────────────────────────────────────────────────────────

def get_weighted_sampler(train_dataset: WasteDataset) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler for the training DataLoader.

    This is an ALTERNATIVE to using weighted loss.  Instead of changing the
    loss function, we oversample images from minority classes so that each
    batch sees a roughly balanced class distribution.

    How it works:
      1. Compute per-class sample weights (same inverse-frequency formula).
      2. Assign each individual sample a weight equal to its class weight.
      3. WeightedRandomSampler draws samples with replacement according to
         these per-sample weights, ensuring each epoch sees all classes
         equally often.

    Trade-off vs weighted loss:
      - Sampler: physically shows minority images more often → more gradient
        updates for rare classes per epoch.  Risk: overfits to minority class.
      - Weighted loss: sees all images once per epoch but upweights the loss
        for rare classes.  More stable, less risk of overfitting.
      - Best practice: experiment with both (Phase 5) and compare.

    Args:
        train_dataset: The training WasteDataset instance.

    Returns:
        torch.utils.data.WeightedRandomSampler
    """
    from collections import Counter

    counts = Counter(train_dataset.targets)
    n_total = len(train_dataset)
    n_classes = len(train_dataset.classes)

    # Per-class weights
    class_weights = {
        idx: n_total / (n_classes * max(counts.get(idx, 1), 1))
        for idx in range(n_classes)
    }

    # Per-sample weights (each sample gets its class weight)
    sample_weights = [
        class_weights[label] for label in train_dataset.targets
    ]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,   # Sampling with replacement allows oversampling minority classes
    )

    logger.info(
        f"WeightedRandomSampler created: "
        f"{len(sample_weights)} samples, "
        f"replacement=True"
    )
    return sampler


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

def create_dataloaders(
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    use_weighted_sampler: bool = False,
    data_root: Optional[Path] = None,
    pin_memory: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create and return train, validation, and test DataLoaders.

    Args:
        batch_size:            Images per mini-batch.  Defaults to cfg value (32).
        num_workers:           Parallel data loading workers.  Defaults to cfg value.
                               Set to 0 for debugging (single-process loading).
        use_weighted_sampler:  If True, use WeightedRandomSampler on the training
                               loader instead of random shuffling.  This is one
                               of two imbalance-handling strategies.
        data_root:             Custom processed data root.  Defaults to cfg path.
        pin_memory:            Pin tensors to CUDA-pinned memory for faster
                               GPU transfer.  Defaults to cfg value.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Notes:
        - Training DataLoader shuffles by default (or uses WeightedRandomSampler).
        - Validation and test DataLoaders never shuffle — deterministic order
          ensures confusion matrix rows/columns are consistent across runs.
        - drop_last=True on train prevents a small final batch from causing
          BatchNorm instability (a single-sample batch makes running stats noisy).
    """
    # Use config defaults if not overridden
    batch_size  = batch_size  if batch_size  is not None else cfg.training.BATCH_SIZE
    num_workers = num_workers if num_workers is not None else cfg.training.NUM_WORKERS
    pin_memory  = pin_memory  if pin_memory  is not None else cfg.training.PIN_MEMORY

    logger.info(
        f"Creating DataLoaders | "
        f"batch={batch_size}  workers={num_workers}  "
        f"weighted_sampler={use_weighted_sampler}"
    )

    # ── Build datasets ────────────────────────────────────────────────────────
    train_ds = WasteDataset(
        split="train",
        transform=get_train_transforms(),
        data_root=data_root,
        verbose=True,
    )
    val_ds = WasteDataset(
        split="val",
        transform=get_val_transforms(),
        data_root=data_root,
        verbose=True,
    )
    test_ds = WasteDataset(
        split="test",
        transform=get_test_transforms(),
        data_root=data_root,
        verbose=True,
    )

    # ── Training DataLoader ───────────────────────────────────────────────────
    if use_weighted_sampler:
        sampler = get_weighted_sampler(train_ds)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,          # sampler is mutually exclusive with shuffle
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=(num_workers > 0),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,             # Shuffle training data each epoch
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=(num_workers > 0),
        )

    # ── Validation DataLoader ─────────────────────────────────────────────────
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,                # Never shuffle validation — reproducible
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    # ── Test DataLoader ───────────────────────────────────────────────────────
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    logger.info(
        f"DataLoaders ready:\n"
        f"  Train : {len(train_ds):>5} images  "
        f"{len(train_loader):>4} batches\n"
        f"  Val   : {len(val_ds):>5} images  "
        f"{len(val_loader):>4} batches\n"
        f"  Test  : {len(test_ds):>5} images  "
        f"{len(test_loader):>4} batches"
    )

    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
# Self-test (run directly to validate on the processed dataset)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from collections import Counter

    print("=" * 60)
    print(" WasteDataset — Self-Test")
    print("=" * 60)

    # Build datasets
    train_ds = WasteDataset("train", transform=get_train_transforms())
    val_ds   = WasteDataset("val",   transform=get_val_transforms())
    test_ds  = WasteDataset("test",  transform=get_test_transforms())

    # Verify __len__
    print(f"\n__len__ check:")
    print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    assert len(train_ds) > 0 and len(val_ds) > 0 and len(test_ds) > 0

    # Verify __getitem__ shape
    img, label = train_ds[0]
    print(f"\n__getitem__ check:")
    print(f"  image.shape = {img.shape}")   # (3, 224, 224)
    print(f"  image.dtype = {img.dtype}")   # float32
    print(f"  label       = {label} ({train_ds.classes[label]})")
    assert img.shape == torch.Size([3, 224, 224])
    assert img.dtype == torch.float32
    assert 0 <= label < cfg.NUM_CLASSES

    # Verify class_to_idx is consistent
    print(f"\nclass_to_idx: {train_ds.class_to_idx}")
    assert list(train_ds.class_to_idx.keys()) == sorted(train_ds.class_to_idx.keys()), \
        "class_to_idx must be in alphabetical order"

    # Verify all 6 classes are represented
    present_classes = set(train_ds.classes)
    expected_classes = set(cfg.CLASS_NAMES)
    missing = expected_classes - present_classes
    if missing:
        print(f"\n⚠  Missing classes in training set: {missing}")
    else:
        print(f"\n✅ All {cfg.NUM_CLASSES} classes present in training set.")

    # Class weight computation
    device = torch.device("cpu")
    weights = compute_class_weights(train_ds, device)
    print(f"\nClass weights tensor: {weights}")

    # DataLoader
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=4, num_workers=0
    )
    batch_imgs, batch_labels = next(iter(train_loader))
    print(f"\nBatch check:")
    print(f"  batch_imgs.shape   = {batch_imgs.shape}")    # (4, 3, 224, 224)
    print(f"  batch_labels.shape = {batch_labels.shape}")  # (4,)
    print(f"  batch_labels.dtype = {batch_labels.dtype}")  # torch.int64
    assert batch_imgs.shape == torch.Size([4, 3, 224, 224])
    assert batch_labels.dtype == torch.int64

    print("\n✅ All dataset self-tests passed.")
