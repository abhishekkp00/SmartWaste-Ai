"""CRUD helpers for prediction logs and analytics."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import PredictionLog


async def create_prediction_log(
    session: AsyncSession,
    *,
    image_name: str,
    prediction: Dict[str, Any],
) -> PredictionLog:
    """Insert one prediction log row and return the saved ORM object."""
    row = PredictionLog(
        model_name=str(prediction.get("model_name", "custom_cnn")),
        image_name=image_name,
        class_id=int(prediction["class_id"]),
        class_name=str(prediction["class_name"]),
        confidence=float(prediction["confidence"]),
        is_confident=bool(prediction["is_confident"]),
        warning=prediction.get("warning"),
        top_k_json=json.dumps(prediction.get("top_k", [])),
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


async def get_recent_predictions(session: AsyncSession, limit: int = 10) -> List[PredictionLog]:
    """Return latest predictions in descending timestamp order."""
    stmt = select(PredictionLog).order_by(PredictionLog.created_at.desc()).limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_analytics_summary(session: AsyncSession) -> Dict[str, Any]:
    """Compute aggregate analytics from prediction history."""
    total_stmt = select(func.count(PredictionLog.id))
    avg_stmt = select(func.avg(PredictionLog.confidence))
    low_conf_stmt = select(func.count(PredictionLog.id)).where(PredictionLog.is_confident.is_(False))

    total = int((await session.execute(total_stmt)).scalar_one() or 0)
    avg_confidence = float((await session.execute(avg_stmt)).scalar_one() or 0.0)
    low_confidence_count = int((await session.execute(low_conf_stmt)).scalar_one() or 0)

    class_dist_stmt = (
        select(PredictionLog.class_name, func.count(PredictionLog.id))
        .group_by(PredictionLog.class_name)
        .order_by(func.count(PredictionLog.id).desc())
    )
    class_rows = (await session.execute(class_dist_stmt)).all()
    class_distribution = {str(name): int(count) for name, count in class_rows}

    return {
        "total_predictions": total,
        "average_confidence": avg_confidence,
        "low_confidence_count": low_confidence_count,
        "low_confidence_rate": (low_confidence_count / total) if total else 0.0,
        "class_distribution": class_distribution,
    }
