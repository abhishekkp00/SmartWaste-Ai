"""
ml/predict.py
─────────────────────────────────────────────────────────────────────────────
SmartWaste AI — Inference and recommendation engine

Provides:
  - checkpoint loading for CustomCNN/ResNet18 variants
  - image preprocessing and single/batch prediction
  - top-k probabilities and confidence-threshold warning
  - recycling recommendation lookup by predicted class
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import torch
from PIL import Image
from torchvision.transforms import functional as TF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import cfg
from ml.custom_cnn import build_custom_cnn
from ml.preprocessing import get_test_transforms
from ml.resnet_model import build_resnet18_stage_a, build_resnet18_stage_b

ImageLike = Union[Image.Image, str, Path, torch.Tensor]

RECYCLING_RECOMMENDATIONS: Dict[str, Dict[str, str]] = {
    "cardboard": {
        "disposal_bin": "Dry recyclables (paper/cardboard)",
        "prep": "Flatten boxes and keep cardboard clean and dry.",
        "notes": "Remove plastic tape and food residue before disposal.",
    },
    "glass": {
        "disposal_bin": "Glass recycling bin",
        "prep": "Rinse containers and remove caps/lids when possible.",
        "notes": "Do not include broken mirrors or ceramics in glass recycling.",
    },
    "metal": {
        "disposal_bin": "Metal recycling bin",
        "prep": "Rinse cans and lightly crush to save space.",
        "notes": "Sharp metal objects should be wrapped safely before disposal.",
    },
    "paper": {
        "disposal_bin": "Dry paper recycling bin",
        "prep": "Keep paper dry; stack or fold to reduce volume.",
        "notes": "Greasy or heavily soiled paper should go to general waste.",
    },
    "plastic": {
        "disposal_bin": "Plastic recycling bin",
        "prep": "Rinse bottles/containers and compress where possible.",
        "notes": "Check local recycling rules for plastic type acceptance.",
    },
    "trash": {
        "disposal_bin": "General waste bin",
        "prep": "Bag securely to prevent leakage and odor.",
        "notes": "If uncertain, choose general waste to avoid recycling contamination.",
    },
}


def get_recommendation(class_name: str) -> Dict[str, str]:
    """Return recommendation metadata for a predicted class."""
    return RECYCLING_RECOMMENDATIONS.get(
        class_name,
        {
            "disposal_bin": "General waste bin",
            "prep": "Dispose safely.",
            "notes": "Class not recognized by recommendation engine.",
        },
    )


class WastePredictor:
    """Load a trained model and run top-k waste classification inference."""

    def __init__(
        self,
        model_name: str = "custom_cnn",
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        confidence_threshold: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name.lower()
        self.class_names = list(cfg.CLASS_NAMES)
        self.transform = get_test_transforms()

        self.confidence_threshold = (
            cfg.inference.CONFIDENCE_THRESHOLD
            if confidence_threshold is None
            else float(confidence_threshold)
        )
        self.confidence_threshold = min(max(self.confidence_threshold, 0.0), 1.0)

        requested_top_k = cfg.inference.TOP_K if top_k is None else int(top_k)
        self.top_k = max(1, min(requested_top_k, cfg.NUM_CLASSES))

        self.model = self._build_model(self.model_name)
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        else:
            self.model.to(self.device)
            self.model.eval()

    def _build_model(self, model_name: str):
        if model_name in {"custom_cnn", "cnn"}:
            return build_custom_cnn()
        if model_name in {"resnet18_stage_a", "resnet_stage_a", "resnet_a"}:
            return build_resnet18_stage_a(pretrained=False)
        if model_name in {"resnet18_stage_b", "resnet_stage_b", "resnet_b", "resnet18"}:
            return build_resnet18_stage_b(pretrained=False)
        raise ValueError(
            "Unsupported model_name. Use one of: "
            "custom_cnn, resnet18_stage_a, resnet18_stage_b"
        )

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load checkpoint from .pth file.

        Supported formats:
          1) Raw state_dict
          2) Dict containing 'model_state_dict'
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict):
            # Assume raw state_dict
            state_dict = checkpoint
        else:
            raise ValueError("Unsupported checkpoint format.")

        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _to_pil(image: ImageLike) -> Image.Image:
        """Convert multiple supported inputs to RGB PIL.Image."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.ndim == 4 and tensor.size(0) == 1:
                tensor = tensor.squeeze(0)
            if tensor.ndim != 3:
                raise ValueError("Tensor image must have shape [C,H,W] or [1,C,H,W].")
            if tensor.shape[0] not in {1, 3}:
                tensor = tensor.permute(2, 0, 1)
            return TF.to_pil_image(tensor).convert("RGB")
        raise TypeError("Unsupported image input type.")

    def preprocess(self, image: ImageLike) -> torch.Tensor:
        """Preprocess image to model-ready tensor [1, 3, 224, 224]."""
        pil_image = self._to_pil(image)
        tensor = self.transform(pil_image)
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image: ImageLike, top_k: Optional[int] = None) -> Dict[str, object]:
        """Run single-image inference and return structured top-k response."""
        run_top_k = self.top_k if top_k is None else max(1, min(int(top_k), cfg.NUM_CLASSES))
        batch = self.preprocess(image)
        logits = self.model(batch)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

        confidence, class_id = torch.max(probabilities, dim=0)
        confidence_value = float(confidence.item())
        class_index = int(class_id.item())
        class_name = self.class_names[class_index]

        top_probabilities, top_indices = torch.topk(probabilities, k=run_top_k)
        top_predictions = []
        for prob, index in zip(top_probabilities.tolist(), top_indices.tolist()):
            name = self.class_names[index]
            top_predictions.append(
                {
                    "class_id": int(index),
                    "class_name": name,
                    "display_name": cfg.dataset.CLASS_DISPLAY_NAMES.get(name, name.title()),
                    "confidence": float(prob),
                }
            )

        is_confident = confidence_value >= self.confidence_threshold
        warning = None
        if not is_confident:
            warning = (
                f"Low-confidence prediction ({confidence_value:.3f} < "
                f"{self.confidence_threshold:.3f}). Consider manual verification."
            )

        return {
            "model_name": self.model_name,
            "class_id": class_index,
            "class_name": class_name,
            "display_name": cfg.dataset.CLASS_DISPLAY_NAMES.get(class_name, class_name.title()),
            "confidence": confidence_value,
            "is_confident": is_confident,
            "warning": warning,
            "recommendation": get_recommendation(class_name),
            "top_k": top_predictions,
        }

    def predict_batch(self, images: Iterable[ImageLike], top_k: Optional[int] = None) -> List[Dict[str, object]]:
        """Run inference for multiple images (simple sequential implementation)."""
        return [self.predict(image, top_k=top_k) for image in images]


if __name__ == "__main__":
    demo = Image.new("RGB", (320, 240), color=(110, 160, 210))
    predictor = WastePredictor(model_name="custom_cnn", device=torch.device("cpu"))
    result = predictor.predict(demo)

    print("SmartWaste AI — Phase 7 predictor smoke test")
    print(f"Predicted class: {result['class_name']}")
    print(f"Confidence     : {result['confidence']:.4f}")
    if result["warning"]:
        print(f"Warning        : {result['warning']}")
    print(f"Recommendation : {result['recommendation']['disposal_bin']}")
