"""
ml/eda.py
─────────────────────────────────────────────────────────────────────────────
SmartWaste AI — Exploratory Data Analysis (EDA) Module

Performs a thorough analysis of the TrashNet dataset and generates charts
saved in results/eda/:

  1. class_distribution_bar.png    — image counts per class (all splits)
  2. class_distribution_pie.png    — proportional distribution
  3. sample_grid.png               — 4 representative images per class
  4. image_dimensions.png          — scatter plot of W vs H
  5. channel_stats.png             — mean per-channel pixel values per class
  6. class_imbalance_summary.txt   — text report
  7. augmentation_preview.png      — same image before/after train transforms

Usage:
    python ml/eda.py          (runs all analyses)
    python ml/eda.py --quick  (runs only fast checks, skips dimension scan)
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (works in Colab & headless)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import cfg
from ml.utils import get_logger

logger = get_logger("eda")

# ── Output directory for all EDA artefacts ────────────────────────────────────
EDA_DIR = cfg.paths.results_dir / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# ── Class display colours (one per class, consistent across all plots) ────────
CLASS_COLORS = {
    "cardboard": "#E67E22",
    "glass":     "#3498DB",
    "metal":     "#95A5A6",
    "paper":     "#F1C40F",
    "plastic":   "#E74C3C",
    "trash":     "#7F8C8D",
}
CLASSES   = list(cfg.CLASS_NAMES)   # ['cardboard','glass','metal','paper','plastic','trash']
SPLITS    = ["train", "val", "test"]
SPLIT_COLORS = {"train": "#2ECC71", "val": "#3498DB", "test": "#E74C3C"}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _image_paths(split: str, class_name: str) -> List[Path]:
    """Return sorted list of image paths for a split/class combination."""
    d = cfg.paths.processed_data / split / class_name
    if not d.exists():
        return []
    return sorted(
        p for p in d.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )


def _load_pil(path: Path):
    """Load a PIL Image in RGB mode; return None if corrupted."""
    try:
        from PIL import Image
        with Image.open(path) as img:
            return img.convert("RGB").copy()
    except Exception as e:
        logger.warning(f"Cannot open {path.name}: {e}")
        return None


def _apply_style() -> None:
    """Apply a consistent matplotlib style for all plots."""
    plt.rcParams.update({
        "figure.facecolor":  "#1a1a2e",
        "axes.facecolor":    "#16213e",
        "axes.edgecolor":    "#0f3460",
        "axes.labelcolor":   "#e0e0e0",
        "axes.titlecolor":   "#e0e0e0",
        "xtick.color":       "#a0a0a0",
        "ytick.color":       "#a0a0a0",
        "text.color":        "#e0e0e0",
        "grid.color":        "#0f3460",
        "grid.linewidth":    0.6,
        "font.family":       "DejaVu Sans",
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
    })


# ──────────────────────────────────────────────────────────────────────────────
# 1. Count images per class / split
# ──────────────────────────────────────────────────────────────────────────────

def count_images() -> Dict[str, Dict[str, int]]:
    """
    Count images in each split/class directory.

    Returns:
        {split: {class_name: count}}
    """
    counts: Dict[str, Dict[str, int]] = {}
    for split in SPLITS:
        counts[split] = {}
        for cls in CLASSES:
            counts[split][cls] = len(_image_paths(split, cls))
    return counts


# ──────────────────────────────────────────────────────────────────────────────
# 2. Bar chart: image counts per class
# ──────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(counts: Dict[str, Dict[str, int]]) -> Path:
    """
    Stacked bar chart showing train/val/test image counts per class.

    Observations documented:
      - Severe class imbalance with 'trash' having the fewest images.
      - 'paper' has the highest count (~594 total).
      - Imbalance ratio paper:trash ≈ 4.3×.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("TrashNet — Class Distribution", fontsize=16, y=1.01, color="#e0e0e0")

    x = np.arange(len(CLASSES))

    # ── Left: Stacked bar by split ──────────────────────────────────────────
    ax1 = axes[0]
    bottoms = np.zeros(len(CLASSES))
    for split in SPLITS:
        vals = np.array([counts[split].get(cls, 0) for cls in CLASSES])
        bars = ax1.bar(
            x, vals, bottom=bottoms, label=split.capitalize(),
            color=SPLIT_COLORS[split], alpha=0.88, width=0.65, edgecolor="none"
        )
        # Add count labels on segments > 20 images
        for bar, v in zip(bars, vals):
            if v > 20:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(int(v)), ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold"
                )
        bottoms += vals

    ax1.set_title("Image Counts by Split")
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.capitalize() for c in CLASSES], rotation=20, ha="right")
    ax1.set_ylabel("Number of Images")
    ax1.legend(facecolor="#1a1a2e", edgecolor="#0f3460")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_axisbelow(True)

    # ── Right: Total count with class-colour bars ───────────────────────────
    ax2 = axes[1]
    totals = [
        sum(counts[s].get(cls, 0) for s in SPLITS)
        for cls in CLASSES
    ]
    colors = [CLASS_COLORS[cls] for cls in CLASSES]
    bars2 = ax2.bar(x, totals, color=colors, alpha=0.9, width=0.65, edgecolor="none")

    for bar, total in zip(bars2, totals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(total), ha="center", va="bottom",
            fontsize=9, color="#e0e0e0", fontweight="bold"
        )

    grand_total = sum(totals)
    for bar, total in zip(bars2, totals):
        pct = 100 * total / grand_total
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{pct:.1f}%", ha="center", va="center",
            fontsize=8, color="white"
        )

    ax2.set_title("Total Images per Class")
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.capitalize() for c in CLASSES], rotation=20, ha="right")
    ax2.set_ylabel("Total Images")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    out_path = EDA_DIR / "class_distribution_bar.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved → {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# 3. Sample image grid: 4 images per class
# ──────────────────────────────────────────────────────────────────────────────

def plot_sample_grid(samples_per_class: int = 4) -> Path:
    """
    Display representative images from each class.

    Picks the first N images from the training split for each class.
    Images are displayed in a grid: rows=classes, cols=samples.
    """
    import random
    random.seed(cfg.dataset.SPLIT_SEED)

    n_classes = len(CLASSES)
    fig, axes = plt.subplots(
        n_classes, samples_per_class,
        figsize=(samples_per_class * 2.8, n_classes * 2.8)
    )
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("TrashNet — Sample Images per Class", fontsize=15, color="#e0e0e0", y=1.01)

    for row, cls_name in enumerate(CLASSES):
        paths = _image_paths("train", cls_name)
        # Randomly sample from first 50 images for variety
        sample_pool = paths[:min(50, len(paths))]
        samples = random.sample(sample_pool, min(samples_per_class, len(sample_pool)))

        for col in range(samples_per_class):
            ax = axes[row][col]
            ax.set_facecolor("#16213e")
            ax.axis("off")

            if col < len(samples):
                img = _load_pil(samples[col])
                if img is not None:
                    ax.imshow(np.array(img))

            # Class label on the left of the first column
            if col == 0:
                ax.set_ylabel(
                    cls_name.upper(),
                    rotation=0, labelpad=55, va="center",
                    color=CLASS_COLORS[cls_name],
                    fontsize=10, fontweight="bold"
                )

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    out_path = EDA_DIR / "sample_grid.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved → {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# 4. Image dimension analysis
# ──────────────────────────────────────────────────────────────────────────────

def plot_image_dimensions(max_per_class: int = 100) -> Path:
    """
    Scatter plot of original image widths vs heights, coloured by class.

    Limited to `max_per_class` images per class to keep runtime manageable.
    Also reports the most common image dimensions.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("TrashNet — Original Image Dimensions", fontsize=15, color="#e0e0e0")

    all_sizes: Dict[str, List[Tuple[int, int]]] = {cls: [] for cls in CLASSES}
    all_widths, all_heights = [], []

    for cls in CLASSES:
        paths = _image_paths("train", cls)[:max_per_class]
        for p in paths:
            img = _load_pil(p)
            if img is not None:
                w, h = img.size
                all_sizes[cls].append((w, h))
                all_widths.append(w)
                all_heights.append(h)

    # ── Scatter: W vs H per class ───────────────────────────────────────────
    ax1 = axes[0]
    for cls in CLASSES:
        ws = [s[0] for s in all_sizes[cls]]
        hs = [s[1] for s in all_sizes[cls]]
        ax1.scatter(ws, hs, label=cls.capitalize(), color=CLASS_COLORS[cls],
                    alpha=0.6, s=20, edgecolors="none")

    ax1.set_title("Width vs Height per Class")
    ax1.set_xlabel("Width (px)")
    ax1.set_ylabel("Height (px)")
    ax1.legend(facecolor="#1a1a2e", edgecolor="#0f3460", fontsize=8)
    ax1.grid(alpha=0.3)

    # ── Histogram of widths ─────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.hist(all_widths, bins=30, color="#3498DB", alpha=0.75, label="Width")
    ax2.hist(all_heights, bins=30, color="#E74C3C", alpha=0.75, label="Height")
    ax2.set_title("Distribution of Image Dimensions")
    ax2.set_xlabel("Pixels")
    ax2.set_ylabel("Count")
    ax2.legend(facecolor="#1a1a2e", edgecolor="#0f3460")
    ax2.grid(alpha=0.3)
    ax2.axvline(x=512, color="#F1C40F", linestyle="--", linewidth=1.2, label="TrashNet standard (512)")
    ax2.axvline(x=224, color="#E67E22", linestyle="--", linewidth=1.2, label="Model input (224)")

    plt.tight_layout()
    out_path = EDA_DIR / "image_dimensions.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved → {out_path}")

    # Print statistics
    if all_widths:
        logger.info(f"Width  — min={min(all_widths):>4}  max={max(all_widths):>4}  "
                    f"mean={np.mean(all_widths):.0f}")
        logger.info(f"Height — min={min(all_heights):>4}  max={max(all_heights):>4}  "
                    f"mean={np.mean(all_heights):.0f}")

    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# 5. Per-channel pixel statistics per class
# ──────────────────────────────────────────────────────────────────────────────

def plot_channel_statistics(max_per_class: int = 50) -> Path:
    """
    Compute and visualise mean R, G, B pixel values per class.

    Helps identify whether classes have systematic colour differences —
    e.g., glass images tend to be bluer, paper images tend to be lighter.
    """
    _apply_style()
    channel_means: Dict[str, List[float]] = {cls: [] for cls in CLASSES}

    for cls in CLASSES:
        paths = _image_paths("train", cls)[:max_per_class]
        r_means, g_means, b_means = [], [], []
        for p in paths:
            img = _load_pil(p)
            if img is None:
                continue
            arr = np.array(img, dtype=np.float32) / 255.0
            r_means.append(arr[:, :, 0].mean())
            g_means.append(arr[:, :, 1].mean())
            b_means.append(arr[:, :, 2].mean())

        channel_means[cls] = [
            np.mean(r_means) if r_means else 0,
            np.mean(g_means) if g_means else 0,
            np.mean(b_means) if b_means else 0,
        ]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Mean RGB Channel Values per Class (Training Set)", fontsize=14, color="#e0e0e0")

    x = np.arange(len(CLASSES))
    width = 0.26

    # R, G, B bars side by side
    ax.bar(x - width, [channel_means[c][0] for c in CLASSES],
           width, label="Red",   color="#E74C3C", alpha=0.85, edgecolor="none")
    ax.bar(x,         [channel_means[c][1] for c in CLASSES],
           width, label="Green", color="#2ECC71", alpha=0.85, edgecolor="none")
    ax.bar(x + width, [channel_means[c][2] for c in CLASSES],
           width, label="Blue",  color="#3498DB", alpha=0.85, edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CLASSES])
    ax.set_ylabel("Mean Pixel Value (0–1)")
    ax.set_ylim(0, 1)
    ax.legend(facecolor="#1a1a2e", edgecolor="#0f3460")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_path = EDA_DIR / "channel_statistics.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved → {out_path}")

    # Print table
    logger.info("Channel statistics (mean pixel value [0,1]):")
    logger.info(f"  {'Class':<12} {'R':>6} {'G':>6} {'B':>6}")
    for cls in CLASSES:
        r, g, b = channel_means[cls]
        logger.info(f"  {cls:<12} {r:>6.4f} {g:>6.4f} {b:>6.4f}")

    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# 6. Augmentation preview
# ──────────────────────────────────────────────────────────────────────────────

def plot_augmentation_preview(n_augmented: int = 6) -> Path:
    """
    Show the same image before and after training augmentation.

    Uses one image from each class and displays:
      Row 1: Original (val transforms only — deterministic)
      Row 2: 6 different random augmentations of the same image

    This demonstrates that augmentations preserve class identity.
    """
    from ml.preprocessing import get_train_transforms, get_val_transforms, get_inverse_normalize
    import torch

    _apply_style()
    inv_norm = get_inverse_normalize()

    fig, axes = plt.subplots(
        len(CLASSES), n_augmented + 1,
        figsize=((n_augmented + 1) * 2.4, len(CLASSES) * 2.4)
    )
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(
        "Augmentation Preview (Original | 6 Random Augmentations)",
        fontsize=14, color="#e0e0e0", y=1.01
    )

    train_tfm = get_train_transforms()
    val_tfm   = get_val_transforms()

    for row, cls_name in enumerate(CLASSES):
        paths = _image_paths("train", cls_name)
        if not paths:
            continue
        pil_img = _load_pil(paths[0])
        if pil_img is None:
            continue

        for col in range(n_augmented + 1):
            ax = axes[row][col]
            ax.set_facecolor("#16213e")
            ax.axis("off")

            if col == 0:
                # Original (deterministic val transform)
                tensor = val_tfm(pil_img)
                img_np = inv_norm(tensor).clamp(0, 1).permute(1, 2, 0).numpy()
                ax.imshow(img_np)
                ax.set_title("Original", fontsize=7, color="#a0a0a0")
                if col == 0:
                    ax.set_ylabel(
                        cls_name.upper(),
                        rotation=0, labelpad=55, va="center",
                        color=CLASS_COLORS[cls_name], fontsize=9, fontweight="bold"
                    )
            else:
                # Random augmentation
                tensor = train_tfm(pil_img)
                img_np = inv_norm(tensor).clamp(0, 1).permute(1, 2, 0).numpy()
                ax.imshow(img_np)
                ax.set_title(f"Aug {col}", fontsize=7, color="#606090")

    plt.subplots_adjust(wspace=0.04, hspace=0.1)
    out_path = EDA_DIR / "augmentation_preview.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved → {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# 7. Text report: class imbalance summary
# ──────────────────────────────────────────────────────────────────────────────

def write_imbalance_report(counts: Dict[str, Dict[str, int]]) -> Path:
    """
    Write a text report documenting the class imbalance with mitigation notes.
    """
    out_path = EDA_DIR / "class_imbalance_summary.txt"

    totals = {cls: sum(counts[s].get(cls, 0) for s in SPLITS) for cls in CLASSES}
    grand_total = sum(totals.values())
    max_cls = max(totals, key=totals.get)
    min_cls = min(totals, key=totals.get)
    imbalance_ratio = totals[max_cls] / max(totals[min_cls], 1)

    lines = [
        "=" * 60,
        " SmartWaste AI — Class Imbalance Analysis",
        f" Dataset: TrashNet (Gary Thung & Mindy Yang, Stanford)",
        f" Total images: {grand_total}",
        "=" * 60,
        "",
        f"{'Class':<14} {'Total':>6} {'% of data':>10} {'Imbalance'}",
        "-" * 60,
    ]

    for cls in sorted(CLASSES, key=lambda c: totals[c], reverse=True):
        pct = 100.0 * totals[cls] / grand_total
        bar = "█" * int(pct / 2)
        lines.append(f"  {cls:<12} {totals[cls]:>6} {pct:>9.1f}%  {bar}")

    lines += [
        "",
        f"Most frequent class  : {max_cls}  ({totals[max_cls]} images)",
        f"Least frequent class : {min_cls}  ({totals[min_cls]} images)",
        f"Imbalance ratio      : {imbalance_ratio:.1f}×",
        "",
        "OBSERVATIONS:",
        "  1. The 'trash' class has only ~137 images (5.4% of total).",
        "     This is 4.3× less than 'paper', the largest class.",
        "  2. Without correction, the model will bias toward larger classes",
        "     and have poor recall for the 'trash' category.",
        "",
        "MITIGATION STRATEGIES (both will be evaluated in Phase 5):",
        "  A. Weighted CrossEntropyLoss:",
        "     weight[c] = N / (K × n_c)",
        "     The 'trash' class gets ~4× higher loss weight.",
        "     Advantage: Sees all data equally; more stable training.",
        "",
        "  B. WeightedRandomSampler:",
        "     Oversamples 'trash' images so each batch has equal class counts.",
        "     Advantage: More gradient updates per epoch for minority class.",
        "     Risk: Overfitting to the 137 'trash' images.",
        "",
        "RECOMMENDATION for initial training:",
        "  Use weighted loss (Strategy A) as the primary approach.",
        "  Run a separate experiment with WeightedRandomSampler (Strategy B).",
        "  Compare per-class F1-scores, particularly for 'trash'.",
        "",
        "=" * 60,
    ]

    # Class weights
    train_counts = {cls: counts["train"].get(cls, 0) for cls in CLASSES}
    n_train = sum(train_counts.values())
    n_cls = len(CLASSES)
    lines.append("\nCLASS WEIGHTS (for training CrossEntropyLoss):")
    for i, cls in enumerate(CLASSES):
        n_c = max(train_counts[cls], 1)
        w = n_train / (n_cls * n_c)
        lines.append(f"  [{i}] {cls:<12}  n_train={n_c:>4}  weight={w:.4f}")
    lines.append("")

    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    logger.info(f"Saved → {out_path}")
    print(text)
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(quick: bool = False) -> None:
    logger.info("=" * 60)
    logger.info(" SmartWaste AI — EDA")
    logger.info("=" * 60)

    # Count images
    logger.info("\n[1/7] Counting images per class…")
    counts = count_images()

    # Print summary table
    print(f"\n{'Class':<14}", end="")
    for s in SPLITS:
        print(f" {s.capitalize():>6}", end="")
    print(f" {'Total':>7}")
    print("─" * 50)
    for cls in CLASSES:
        total = sum(counts[s].get(cls, 0) for s in SPLITS)
        print(f"  {cls:<12}", end="")
        for s in SPLITS:
            print(f" {counts[s].get(cls, 0):>6}", end="")
        print(f" {total:>7}")

    # Charts
    logger.info("\n[2/7] Plotting class distribution…")
    plot_class_distribution(counts)

    logger.info("\n[3/7] Building sample image grid…")
    plot_sample_grid()

    if not quick:
        logger.info("\n[4/7] Analysing image dimensions…")
        plot_image_dimensions()

        logger.info("\n[5/7] Computing channel statistics…")
        plot_channel_statistics()

    logger.info("\n[6/7] Generating augmentation preview…")
    plot_augmentation_preview()

    logger.info("\n[7/7] Writing class imbalance report…")
    write_imbalance_report(counts)

    logger.info(f"\n✅ EDA complete. All outputs saved to: {EDA_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick", action="store_true",
        help="Skip slow analyses (dimension scan, channel stats)."
    )
    args = parser.parse_args()
    main(quick=args.quick)
