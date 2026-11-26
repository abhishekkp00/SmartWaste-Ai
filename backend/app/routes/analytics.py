"""Analytics endpoint."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.session import get_db_session
from ..schemas.prediction import AnalyticsResponseSchema
from ..services.analytics_service import build_analytics_response

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("", response_model=AnalyticsResponseSchema)
async def get_analytics(
    limit: int = Query(default=10, ge=1, le=100),
    session: AsyncSession = Depends(get_db_session),
) -> AnalyticsResponseSchema:
    """Return aggregate metrics and latest prediction records."""
    data = await build_analytics_response(session, recent_limit=limit)
    return AnalyticsResponseSchema(**data)
