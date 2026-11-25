"""
ml/preprocessing.py
─────────────────────────────────────────────────────────────────────────────
SmartWaste AI — Image Preprocessing & Augmentation Pipelines

This module defines the transform pipelines applied to images before they
are fed into the neural network.  Three pipelines are provided:

  get_train_transforms()   — augmented pipeline for training images
  get_val_transforms()     — deterministic pipeline for validation images
  get_test_transforms()    — identical to val (deterministic, no augmentation)

Design decisions explained inline.  All constants are imported from
ml/config.py so there is a single source of truth.

Usage:
    from ml.preprocessing import get_train_transforms, get_val_transforms

    train_tfm = get_train_transforms()
    val_tfm   = get_val_transforms()
─────────────────────────────────────────────────────────────────────────────
"""

from torchvision import transforms

# Import all preprocessing constants from the central config — never hard-code
from ml.config import cfg

# ──────────────────────────────────────────────────────────────────────────────
# Convenience aliases (keeps the transform code readable)
# ──────────────────────────────────────────────────────────────────────────────
_MEAN = list(cfg.IMAGENET_MEAN)   # [0.485, 0.456, 0.406]
_STD  = list(cfg.IMAGENET_STD)    # [0.229, 0.224, 0.225]
_SIZE = cfg.IMAGE_SIZE[0]         # 224  (square)
_RSZ  = cfg.preprocessing.RESIZE_SIZE  # 256


# ──────────────────────────────────────────────────────────────────────────────
# Training pipeline
# ──────────────────────────────────────────────────────────────────────────────

def get_train_transforms() -> transforms.Compose:
    """
    Return the augmented transform pipeline used during training.

    Every transformation is explained below:

    1. RandomResizedCrop(224, scale=(0.70, 1.0))
       ─────────────────────────────────────────
       Crops a random region of the image (between 70% and 100% of its area)
       and then resizes it to 224×224.  This forces the model to learn from
       different spatial scales of the same object.  The minimum 70% scale
       ensures that at least 70% of the original waste object is visible —
       preventing extreme crops that show only background.

    2. RandomHorizontalFlip(p=0.5)
       ────────────────────────────
       Mirrors the image left-to-right with 50% probability.  Waste objects
       (bottles, cans, paper) have no meaningful left-right orientation, so
       this doubles effective dataset size without introducing label noise.
       Note: We do NOT apply RandomVerticalFlip because images taken in the
       real world rarely have upside-down waste.

    3. RandomRotation(degrees=15)
       ───────────────────────────
       Rotates the image by a random angle in [-15°, +15°].  Real waste bins
       and camera angles are rarely perfectly aligned.  Moderate rotation
       builds rotational invariance without distorting the object so much
       that the class is unrecognisable.

    4. ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)
       ──────────────────────────────────────────────────────────────────────
       Randomly perturbs the colour properties.  Waste images are captured
       under various lighting conditions (outdoor bins, indoor sorting
       facilities, phone cameras).  Moderate jitter makes the model robust
       to these real-world lighting variations.
       Values are kept small (≤0.2) to avoid producing unrealistic colours.

    5. RandomGrayscale(p=0.05)
       ──────────────────────────
       Converts the image to grayscale with 5% probability.  Adds robustness
       for black-and-white images or single-channel camera captures.

    6. ToTensor()
       ───────────
       Converts a PIL Image (H×W×C, uint8, range [0, 255]) to a PyTorch
       tensor (C×H×W, float32, range [0.0, 1.0]).
       This is required for all PyTorch models.

    7. Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       ──────────────────────────────────────────────────────────────────
       Applies channel-wise z-score normalisation using the statistics
       computed from the ImageNet-1K training set.
       Formula: output[c] = (input[c] - mean[c]) / std[c]

       Why ImageNet stats?
         ResNet18 was pretrained on ImageNet with these exact statistics.
         Using the same normalisation ensures that our input distribution
         matches what the pretrained backbone expects.  We use the same
         stats for the Custom CNN too, so both models receive identically
         normalised inputs — enabling a fair comparison.

    Returns:
        torchvision.transforms.Compose
    """
    p = cfg.preprocessing

    return transforms.Compose([
        # 1. Random spatial crop → resized to 224
        transforms.RandomResizedCrop(
            size=_SIZE,
            scale=p.CROP_SCALE,       # (0.70, 1.0)
            ratio=p.CROP_RATIO,       # (0.75, 1.33)  — aspect ratio range
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),

        # 2. Random left-right mirror
        transforms.RandomHorizontalFlip(p=p.H_FLIP_PROB),  # p=0.5

        # 3. Slight rotation
        transforms.RandomRotation(
            degrees=p.ROTATION_DEGREES,  # 15
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),

        # 4. Colour perturbation
        transforms.ColorJitter(
            brightness=p.BRIGHTNESS,    # 0.2
            contrast=p.CONTRAST,        # 0.2
            saturation=p.SATURATION,    # 0.1
            hue=p.HUE,                  # 0.05
        ),

        # 5. Occasional grayscale (5%)
        transforms.RandomGrayscale(p=p.GRAYSCALE_PROB),

        # 6. PIL → float32 tensor [0, 1]
        transforms.ToTensor(),

        # 7. ImageNet normalisation
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Validation / Test pipeline
# ──────────────────────────────────────────────────────────────────────────────

def get_val_transforms() -> transforms.Compose:
    """
    Return the deterministic transform pipeline for validation and test sets.

    No randomness is used here.  Every image produces exactly the same
    output tensor on every call, which ensures reproducible evaluation.

    1. Resize(256)
       ───────────
       Scales the shorter edge to 256 pixels while preserving the aspect
       ratio.  This is larger than the final crop (224) to avoid any
       border artefacts after CenterCrop.

    2. CenterCrop(224)
       ─────────────────
       Crops the central 224×224 region.  Waste objects are typically
       centred in real photos (especially in TrashNet, where images were
       captured against a neutral background).  Using the centre gives a
       consistent, representative view without randomness.

    3. ToTensor()
       Identical to training pipeline.

    4. Normalize(...)
       Identical statistics to training pipeline.
       IMPORTANT: validation images MUST be normalised with the SAME mean
       and std as training images.  Inconsistency here is a silent bug that
       degrades evaluation accuracy without obvious error messages.

    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose([
        # 1. Maintain aspect ratio, resize shorter edge to 256
        transforms.Resize(_RSZ, interpolation=transforms.InterpolationMode.BILINEAR),

        # 2. Take central 224×224 crop
        transforms.CenterCrop(_SIZE),

        # 3. PIL → float32 tensor
        transforms.ToTensor(),

        # 4. ImageNet normalisation (identical to training)
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def get_test_transforms() -> transforms.Compose:
    """
    Return the test-time transform pipeline.

    Identical to validation transforms.  Defined as a separate function
    so that the calling code is self-documenting and so that test-time
    augmentation (TTA) can be added here independently if needed later.
    """
    return get_val_transforms()


# ──────────────────────────────────────────────────────────────────────────────
# Inverse normalisation (for visualisation)
# ──────────────────────────────────────────────────────────────────────────────

def get_inverse_normalize() -> transforms.Normalize:
    """
    Return a transform that reverses ImageNet normalisation.

    Used to convert a normalised tensor back to a displayable image.
    Formula: output[c] = input[c] * std[c] + mean[c]

    Usage:
        inv_norm = get_inverse_normalize()
        displayable = inv_norm(normalised_tensor).clamp(0, 1)
    """
    inv_mean = [-m / s for m, s in zip(_MEAN, _STD)]
    inv_std  = [1.0 / s for s in _STD]
    return transforms.Normalize(mean=inv_mean, std=inv_std)


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch
    from PIL import Image
    import numpy as np

    # Create a synthetic 400×300 RGB image (simulates a real photo)
    dummy_img = Image.fromarray(
        (np.random.rand(300, 400, 3) * 255).astype(np.uint8), mode="RGB"
    )
    print(f"Input image size : {dummy_img.size}  (W × H)")

    # Test training transform
    train_tfm = get_train_transforms()
    train_tensor = train_tfm(dummy_img)
    print(f"Train tensor shape : {train_tensor.shape}")   # torch.Size([3, 224, 224])
    print(f"Train tensor dtype : {train_tensor.dtype}")   # torch.float32
    assert train_tensor.shape == torch.Size([3, 224, 224])

    # Test val transform
    val_tfm = get_val_transforms()
    val_tensor = val_tfm(dummy_img)
    print(f"Val tensor shape   : {val_tensor.shape}")     # torch.Size([3, 224, 224])
    assert val_tensor.shape == torch.Size([3, 224, 224])

    # Test inverse normalisation
    inv_norm = get_inverse_normalize()
    recovered = inv_norm(val_tensor).clamp(0.0, 1.0)
    assert recovered.shape == torch.Size([3, 224, 224])
    print(f"Inverse norm range : [{recovered.min():.3f}, {recovered.max():.3f}]")

    print("\n✅ preprocessing.py — all self-tests passed.")
