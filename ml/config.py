"""
ml/config.py
─────────────────────────────────────────────────────────────────────────────
SmartWaste AI — Central Configuration Module

All hyperparameters, paths, and constants are defined here.
Training scripts, evaluation scripts, and the backend all import from this
file so that every change is made in exactly one place.

Usage:
    from ml.config import cfg

    print(cfg.NUM_CLASSES)       # 6
    print(cfg.IMAGE_SIZE)        # (224, 224)
    print(cfg.TRAIN_DIR)         # Path object
─────────────────────────────────────────────────────────────────────────────
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Root paths  (all relative to the project root so the config works on Colab,
# local machines, and inside Docker containers without editing.)
# ──────────────────────────────────────────────────────────────────────────────

# Resolve the project root as the parent of this file's directory (ml/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class _Paths:
    """All filesystem paths used in the project."""

    # Project root
    root: Path = _PROJECT_ROOT

    # Raw downloaded dataset
    raw_data: Path = _PROJECT_ROOT / "dataset" / "raw"

    # Processed split directories
    processed_data: Path = _PROJECT_ROOT / "dataset" / "processed"
    train_dir: Path = _PROJECT_ROOT / "dataset" / "processed" / "train"
    val_dir: Path = _PROJECT_ROOT / "dataset" / "processed" / "val"
    test_dir: Path = _PROJECT_ROOT / "dataset" / "processed" / "test"

    # Saved model checkpoints
    models_dir: Path = _PROJECT_ROOT / "models"
    custom_cnn_checkpoint: Path = _PROJECT_ROOT / "models" / "custom_cnn_best.pth"
    resnet18_checkpoint: Path = _PROJECT_ROOT / "models" / "resnet18_best.pth"

    # Training results
    results_dir: Path = _PROJECT_ROOT / "results"
    graphs_dir: Path = _PROJECT_ROOT / "results" / "graphs"
    confusion_matrix_dir: Path = _PROJECT_ROOT / "results" / "confusion_matrix"
    reports_dir: Path = _PROJECT_ROOT / "results" / "reports"
    eda_dir: Path = _PROJECT_ROOT / "results" / "eda"      # Phase 2 EDA outputs

    # Notebooks
    notebooks_dir: Path = _PROJECT_ROOT / "notebooks"


# ──────────────────────────────────────────────────────────────────────────────
# Dataset constants
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Dataset:
    """
    TrashNet dataset constants.

    Class mapping matches torchvision ImageFolder alphabetical ordering:
    cardboard=0, glass=1, metal=2, paper=3, plastic=4, trash=5

    Known class imbalance (from original TrashNet, ~2,527 total images):
        cardboard: 403    glass: 501    metal: 410
        paper:     594    plastic: 482  trash: 137   ← severely underrepresented

    We handle this via:
      1. CrossEntropyLoss with computed class weights
      2. WeightedRandomSampler on the training set
      (Both approaches are run separately and compared.)
    """

    # Ordered alphabetically to match ImageFolder class_to_idx
    CLASS_NAMES: Tuple[str, ...] = (
        "cardboard",
        "glass",
        "metal",
        "paper",
        "plastic",
        "trash",
    )

    NUM_CLASSES: int = 6

    # Recycling-friendly display names for the UI
    CLASS_DISPLAY_NAMES: Dict[str, str] = field(
        default_factory=lambda: {
            "cardboard": "Cardboard",
            "glass": "Glass",
            "metal": "Metal",
            "paper": "Paper",
            "plastic": "Plastic",
            "trash": "Trash / General Waste",
        }
    )

    # Approximate known distribution in TrashNet (used for sanity check)
    KNOWN_CLASS_COUNTS: Dict[str, int] = field(
        default_factory=lambda: {
            "cardboard": 403,
            "glass": 501,
            "metal": 410,
            "paper": 594,
            "plastic": 482,
            "trash": 137,
        }
    )

    # Dataset split ratios
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    # Reproducibility seed for all splits
    SPLIT_SEED: int = 42

    # TrashNet Google Drive file ID (direct dataset download, no Kaggle API needed)
    # Source: https://github.com/garythung/trashnet
    GDRIVE_FILE_ID: str = "0B3P9oO5A3ReSWDJoTE1mam1HMGM"

    # Fallback: Kaggle dataset slug
    KAGGLE_DATASET: str = "fedesoriano/garbage-classification-dataset"


# ──────────────────────────────────────────────────────────────────────────────
# Image preprocessing constants
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Preprocessing:
    """
    Image preprocessing and augmentation constants.

    ImageNet normalisation statistics are used because both our models
    (CustomCNN for feature comparison and ResNet18) benefit from the same
    input distribution that the pretrained backbone expects.

    Why 224×224?
      – Standard input size for ResNet, VGG, MobileNet etc.
      – Large enough to retain fine-grained texture details in waste images.
      – Small enough to fit multiple images per GPU batch.
    """

    # Target spatial dimensions after resizing
    IMAGE_SIZE: Tuple[int, int] = (224, 224)

    # Intermediate resize for centre-crop pipeline (val/test)
    RESIZE_SIZE: int = 256

    # ImageNet channel means and standard deviations
    # Applied identically to training, validation, and test images.
    IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Training augmentation parameters
    # RandomResizedCrop: randomly crops then scales to IMAGE_SIZE.
    #   scale=(0.7, 1.0) → at least 70% of original area is kept, preventing
    #   extreme cropping that destroys waste object structure.
    CROP_SCALE: Tuple[float, float] = (0.70, 1.0)
    CROP_RATIO: Tuple[float, float] = (0.75, 1.33)   # aspect ratio range

    # RandomHorizontalFlip probability
    H_FLIP_PROB: float = 0.5

    # RandomRotation degrees (symmetric around 0)
    ROTATION_DEGREES: int = 15

    # ColorJitter parameters
    # Moderate jitter to simulate different lighting conditions on waste bins.
    BRIGHTNESS: float = 0.2
    CONTRAST: float = 0.2
    SATURATION: float = 0.1
    HUE: float = 0.05

    # RandomGrayscale probability (adds robustness to colour-less images)
    GRAYSCALE_PROB: float = 0.05


# ──────────────────────────────────────────────────────────────────────────────
# Training hyperparameters
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Training:
    """
    Hyperparameters for the full training pipeline.

    Separate config blocks are provided for the Custom CNN and ResNet18
    so each can be tuned independently without touching the other.
    """

    # ── Shared ──────────────────────────────────────────────────────────────
    BATCH_SIZE: int = 32          # Fits comfortably on Colab T4 (16 GB VRAM)
    NUM_WORKERS: int = 2          # DataLoader worker processes (4 on local GPU)
    PIN_MEMORY: bool = True       # Faster CPU→GPU transfer on CUDA

    RANDOM_SEED: int = 42

    # Early stopping: stop if val_loss does not improve for PATIENCE epochs
    PATIENCE: int = 7

    # ── Custom CNN ───────────────────────────────────────────────────────────
    CNN_LR: float = 1e-3
    CNN_WEIGHT_DECAY: float = 1e-4
    CNN_EPOCHS: int = 30
    CNN_LR_T_MAX: int = 30        # CosineAnnealingLR period (= max epochs)
    CNN_LR_ETA_MIN: float = 1e-6  # Minimum LR at the end of cosine cycle
    CNN_DROPOUT1: float = 0.5     # After first FC layer
    CNN_DROPOUT2: float = 0.3     # After second FC layer

    # ── ResNet18 Stage A (head-only, backbone frozen) ────────────────────────
    RESNET_HEAD_LR: float = 1e-3
    RESNET_HEAD_WEIGHT_DECAY: float = 1e-4
    RESNET_HEAD_EPOCHS: int = 15

    # ── ResNet18 Stage B (fine-tuning layer3 + layer4 + head) ───────────────
    RESNET_FINETUNE_HEAD_LR: float = 1e-3
    RESNET_FINETUNE_BACKBONE_LR: float = 1e-4   # 10× smaller for backbone
    RESNET_FINETUNE_WEIGHT_DECAY: float = 1e-4
    RESNET_FINETUNE_EPOCHS: int = 15

    # ── Checkpoint naming ────────────────────────────────────────────────────
    # Metric used to decide "best" checkpoint
    CHECKPOINT_METRIC: str = "val_acc"   # "val_acc" | "val_loss"


# ──────────────────────────────────────────────────────────────────────────────
# Inference / Prediction constants
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Inference:
    """
    Runtime inference parameters shared by the prediction engine and backend.
    """

    # Predictions below this probability trigger an uncertainty warning.
    # Rationale: waste images taken in poor lighting or unusual angles often
    # lead to low-confidence outputs that should not be acted upon blindly.
    CONFIDENCE_THRESHOLD: float = 0.60

    # Number of top-k predictions to return
    TOP_K: int = 3

    # Supported image MIME types
    SUPPORTED_FORMATS: Tuple[str, ...] = (
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "image/bmp",
    )

    # Maximum upload size in bytes (10 MB)
    MAX_IMAGE_BYTES: int = 10 * 1024 * 1024


# ──────────────────────────────────────────────────────────────────────────────
# Public singleton — import this everywhere
# ──────────────────────────────────────────────────────────────────────────────

class Config:
    """
    Single config object exposed to all modules.

    Usage:
        from ml.config import cfg

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = cfg.paths.resnet18_checkpoint
        classes = cfg.dataset.CLASS_NAMES
    """

    paths: _Paths = _Paths()
    dataset: _Dataset = _Dataset()
    preprocessing: _Preprocessing = _Preprocessing()
    training: _Training = _Training()
    inference: _Inference = _Inference()

    # Convenience aliases (most commonly used)
    @property
    def NUM_CLASSES(self) -> int:
        return self.dataset.NUM_CLASSES

    @property
    def CLASS_NAMES(self) -> Tuple[str, ...]:
        return self.dataset.CLASS_NAMES

    @property
    def IMAGE_SIZE(self) -> Tuple[int, int]:
        return self.preprocessing.IMAGE_SIZE

    @property
    def IMAGENET_MEAN(self) -> Tuple[float, float, float]:
        return self.preprocessing.IMAGENET_MEAN

    @property
    def IMAGENET_STD(self) -> Tuple[float, float, float]:
        return self.preprocessing.IMAGENET_STD

    def ensure_dirs(self) -> None:
        """Create all output directories if they do not exist."""
        for attr_name in vars(self.paths):
            path: Path = getattr(self.paths, attr_name)
            if path.suffix == "":          # Directory (no file extension)
                path.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return (
            f"SmartWaste AI Config | "
            f"classes={self.NUM_CLASSES} | "
            f"image_size={self.IMAGE_SIZE} | "
            f"batch={self.training.BATCH_SIZE}"
        )


# Module-level singleton
cfg = Config()


# ── Self-test when run directly ───────────────────────────────────────────────
if __name__ == "__main__":
    print(cfg)
    print(f"\nClass names : {cfg.CLASS_NAMES}")
    print(f"Model paths :")
    print(f"  CNN     → {cfg.paths.custom_cnn_checkpoint}")
    print(f"  ResNet  → {cfg.paths.resnet18_checkpoint}")
    print(f"\nTraining hyperparameters:")
    print(f"  CNN max epochs  : {cfg.training.CNN_EPOCHS}")
    print(f"  ResNet Stage A  : {cfg.training.RESNET_HEAD_EPOCHS} epochs")
    print(f"  ResNet Stage B  : {cfg.training.RESNET_FINETUNE_EPOCHS} epochs")
    print(f"  Batch size      : {cfg.training.BATCH_SIZE}")
    print(f"  Early stopping  : patience={cfg.training.PATIENCE}")
    print(f"\nInference:")
    print(f"  Confidence threshold : {cfg.inference.CONFIDENCE_THRESHOLD}")
    print(f"  Top-K predictions    : {cfg.inference.TOP_K}")

    # Ensure all output directories exist
    cfg.ensure_dirs()
    print("\nAll output directories created successfully.")
