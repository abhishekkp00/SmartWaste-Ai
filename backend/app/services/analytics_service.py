"""Analytics service wrappers around DB CRUD functions."""

from __future__ import annotations

import json
from typing import Dict

from sqlalchemy.ext.asyncio import AsyncSession

from ..database.crud import get_analytics_summary, get_recent_predictions


async def build_analytics_response(session: AsyncSession, recent_limit: int = 10) -> Dict[str, object]:
    """Assemble analytics summary and recent prediction items."""
    summary = await get_analytics_summary(session)
    recent = await get_recent_predictions(session, limit=recent_limit)

    recent_items = []
    for row in recent:
        try:
            _ = json.loads(row.top_k_json)
        except Exception:
            pass

        recent_items.append(
            {
                "id": row.id,
                "created_at": row.created_at,
                "image_name": row.image_name,
                "model_name": row.model_name,
                "class_name": row.class_name,
                "confidence": row.confidence,
                "is_confident": row.is_confident,
            }
        )

    return {
        "summary": summary,
        "recent_predictions": recent_items,
    }
