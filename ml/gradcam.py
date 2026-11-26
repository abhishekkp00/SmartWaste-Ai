"""
ml/gradcam.py
─────────────────────────────────────────────────────────────────────────────
SmartWaste AI — Grad-CAM implementation for ResNet18

Uses forward and backward hooks on the final residual block to produce a
class-discriminative 224×224 heatmap. This module contains no training loop.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.resnet_model import ResNet18Classifier


class GradCAM:
    """Generate Grad-CAM heatmaps for a ResNet18 classifier."""

    def __init__(
        self,
        model: ResNet18Classifier,
        target_layer: Optional[nn.Module] = None,
    ) -> None:
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or model.backbone.layer4[-1]
        self.activations: Optional[Tensor] = None
        self.gradients: Optional[Tensor] = None

        self._forward_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = None

    def _save_activation(self, module: nn.Module, inputs: Tuple[Tensor, ...], output: Tensor) -> None:
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient: Tensor) -> None:
        self.gradients = gradient

    def generate(
        self,
        input_tensor: Tensor,
        class_index: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """Generate a 2D heatmap and return it with the predicted class index."""
        if input_tensor.dim() != 4:
            raise ValueError("input_tensor must have shape [Batch, 3, H, W]")

        input_tensor = input_tensor.detach().requires_grad_(True)
        self.activations = None
        self.gradients = None
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)

        if class_index is None:
            class_index = int(logits.argmax(dim=1)[0].item())

        score = logits[:, class_index].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients")

        weights = self.gradients.detach().mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations.detach()).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()

        cam_min = float(cam.min())
        cam_max = float(cam.max())
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam, dtype=np.float32)

        return cam.astype(np.float32), class_index

    def close(self) -> None:
        self._forward_handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a normalized 2D heatmap on an RGB uint8 image using matplotlib."""
    import matplotlib.pyplot as plt

    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    cmap = plt.get_cmap("jet")
    colored = cmap(np.clip(heatmap, 0.0, 1.0))[..., :3]
    colored = (colored * 255.0).astype(np.uint8)
    overlaid = np.clip((1.0 - alpha) * image + alpha * colored, 0, 255).astype(np.uint8)
    return overlaid


def save_heatmap(
    heatmap: np.ndarray,
    output_path: Path,
    title: str = "Grad-CAM",
) -> Path:
    """Save a standalone normalized heatmap visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path
