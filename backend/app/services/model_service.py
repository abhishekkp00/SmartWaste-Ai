"""Model service for shared predictor instances."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Dict, Optional

from PIL import Image

from ml.config import cfg
from ml.predict import WastePredictor


class ModelService:
    """Manage one shared `WastePredictor` instance per model name."""

    def __init__(self) -> None:
        self._predictors: Dict[str, WastePredictor] = {}
        self._lock = Lock()

    @staticmethod
    def _default_checkpoint_for(model_name: str) -> Optional[Path]:
        if model_name == "custom_cnn" and cfg.paths.custom_cnn_checkpoint.exists():
            return cfg.paths.custom_cnn_checkpoint
        if model_name in {"resnet18_stage_b", "resnet18"} and cfg.paths.resnet18_checkpoint.exists():
            return cfg.paths.resnet18_checkpoint
        return None

    def get_predictor(self, model_name: str = "custom_cnn") -> WastePredictor:
        model_name = model_name.lower()
        with self._lock:
            if model_name not in self._predictors:
                predictor = WastePredictor(
                    model_name=model_name,
                    checkpoint_path=self._default_checkpoint_for(model_name),
                )
                self._predictors[model_name] = predictor
            return self._predictors[model_name]

    def predict(self, image: Image.Image, model_name: str = "custom_cnn", top_k: Optional[int] = None):
        predictor = self.get_predictor(model_name=model_name)
        return predictor.predict(image, top_k=top_k)

    def cache_size(self) -> int:
        return len(self._predictors)


model_service = ModelService()
