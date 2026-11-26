"""
ml/predict.py
─────────────────────────────────────────────────────────────────────────────
SmartWaste AI — Single-image and batch inference engine.

Provides WastePredictor for loading a trained checkpoint, preprocessing a PIL
image, running deterministic inference, and returning top-k predictions.
"""

import io
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch import nn
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import cfg
from ml.custom_cnn import build_custom_cnn
from ml.preprocessing import get_val_transforms


class WastePredictor:
    """Load a model checkpoint and convert waste images into predictions."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        top_k: Optional[int] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = cfg.inference.TOP_K if top_k is None else min(top_k, cfg.NUM_CLASSES)
        self.transform = get_val_transforms()

        if model is not None:
            self.model = model
        else:
            self.model = build_custom_cnn(num_classes=cfg.NUM_CLASSES)

        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _load_image(source: Union[str, Path, io.BytesIO, Image.Image]) -> Image.Image:
        if isinstance(source, Image.Image):
            image = source
        else:
            image = Image.open(source)
        return image.convert("RGB")

    @torch.no_grad()
    def predict(self, source: Union[str, Path, io.BytesIO, Image.Image]) -> Dict[str, object]:
        """Return the predicted class, confidence, uncertainty flag, and top-k list."""
        image = self._load_image(source)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        values, indices = torch.topk(probabilities, k=self.top_k)

        predicted_index = int(indices[0].item())
        predicted_confidence = float(values[0].item())
        top_predictions = [
            {
                "class_name": cfg.CLASS_NAMES[int(index.item())],
                "class_index": int(index.item()),
                "probability": float(value.item()),
            }
            for value, index in zip(values.cpu(), indices.cpu())
        ]

        threshold = cfg.inference.CONFIDENCE_THRESHOLD
        return {
            "class_name": cfg.CLASS_NAMES[predicted_index],
            "class_index": predicted_index,
            "confidence": predicted_confidence,
            "is_uncertain": predicted_confidence < threshold,
            "threshold": threshold,
            "top_k": top_predictions,
        }

    @torch.no_grad()
    def predict_batch(self, sources: List[Union[str, Path, io.BytesIO, Image.Image]]) -> List[Dict[str, object]]:
        """Run inference for a list of images and return one result per image."""
        if not sources:
            return []
        tensors = torch.stack([self.transform(self._load_image(source)) for source in sources]).to(self.device)
        logits = self.model(tensors)
        probabilities = torch.softmax(logits, dim=1)
        values, indices = torch.topk(probabilities, k=self.top_k, dim=1)

        results: List[Dict[str, object]] = []
        for row_values, row_indices in zip(values.cpu(), indices.cpu()):
            predicted_index = int(row_indices[0].item())
            confidence = float(row_values[0].item())
            top_predictions = [
                {
                    "class_name": cfg.CLASS_NAMES[int(index.item())],
                    "class_index": int(index.item()),
                    "probability": float(value.item()),
                }
                for value, index in zip(row_values, row_indices)
            ]
            results.append({
                "class_name": cfg.CLASS_NAMES[predicted_index],
                "class_index": predicted_index,
                "confidence": confidence,
                "is_uncertain": confidence < cfg.inference.CONFIDENCE_THRESHOLD,
                "threshold": cfg.inference.CONFIDENCE_THRESHOLD,
                "top_k": top_predictions,
            })
        return results
