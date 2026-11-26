"""Health endpoint."""

from fastapi import APIRouter

from ..schemas.prediction import HealthResponseSchema
from ..services.model_service import model_service
from ml.config import cfg

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponseSchema)
async def health_check() -> HealthResponseSchema:
    """Return API health and trained-checkpoint availability."""
    checkpoint_status = model_service.checkpoint_status()
    available_models = [name for name, available in checkpoint_status.items() if available]
    return HealthResponseSchema(
        status="ok",
        app_name="SmartWaste AI API",
        version="0.1.0",
        model_cache_size=model_service.cache_size(),
        model_ready=bool(available_models),
        available_models=available_models,
        supported_classes=list(cfg.CLASS_NAMES),
    )
