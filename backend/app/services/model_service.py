"""Model service for shared predictor instances."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Dict, Optional

from fastapi import HTTPException
from PIL import Image

from ml.config import cfg
from ml.predict import WastePredictor


class ModelService:
    """Manage one shared, checkpoint-backed WastePredictor per model name."""

    def __init__(self) -> None:
        self._predictors: Dict[str, WastePredictor] = {}
        self._lock = Lock()

    @staticmethod
    def checkpoint_status() -> Dict[str, bool]:
        """Report which trained checkpoints are present on disk."""
        return {
            "custom_cnn": cfg.paths.custom_cnn_checkpoint.is_file(),
            "resnet18_stage_b": cfg.paths.resnet18_checkpoint.is_file(),
        }

    @staticmethod
    def _default_checkpoint_for(model_name: str) -> Optional[Path]:
        if model_name == "custom_cnn":
            return cfg.paths.custom_cnn_checkpoint
        if model_name in {"resnet18_stage_b", "resnet18"}:
            return cfg.paths.resnet18_checkpoint
        return None

    def get_predictor(self, model_name: str = "custom_cnn") -> WastePredictor:
        model_name = model_name.lower()
        if model_name not in {"custom_cnn", "resnet18_stage_b", "resnet18"}:
            raise HTTPException(
                status_code=422,
                detail="Unsupported model. Choose custom_cnn or resnet18_stage_b.",
            )

        with self._lock:
            if model_name not in self._predictors:
                checkpoint = self._default_checkpoint_for(model_name)
                if checkpoint is None or not checkpoint.is_file():
                    expected = str(checkpoint) if checkpoint else "a configured model checkpoint"
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            f"No trained checkpoint is available for '{model_name}'. "
                            f"Expected: {expected}. Train the model and place the best "
                            "checkpoint in the models/ directory before enabling predictions."
                        ),
                    )
                try:
                    predictor = WastePredictor(
                        model_name=model_name,
                        checkpoint_path=checkpoint,
                    )
                except Exception as exc:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Could not load trained checkpoint for '{model_name}': {exc}",
                    ) from exc
                self._predictors[model_name] = predictor
            return self._predictors[model_name]

    def predict(self, image: Image.Image, model_name: str = "custom_cnn", top_k: Optional[int] = None):
        predictor = self.get_predictor(model_name=model_name)
        return predictor.predict(image, top_k=top_k)

    def cache_size(self) -> int:
        return len(self._predictors)


model_service = ModelService()
