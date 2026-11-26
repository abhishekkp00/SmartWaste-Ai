"""Health endpoint."""

from fastapi import APIRouter

from ..schemas.prediction import HealthResponseSchema
from ..services.model_service import model_service
from ml.config import cfg

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponseSchema)
async def health_check() -> HealthResponseSchema:
    """Return service health and basic runtime metadata."""
    return HealthResponseSchema(
        status="ok",
        app_name="SmartWaste AI API",
        version="0.1.0",
        model_cache_size=model_service.cache_size(),
        supported_classes=list(cfg.CLASS_NAMES),
    )
