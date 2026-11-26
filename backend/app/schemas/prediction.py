"""Pydantic schemas for prediction APIs."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class RecommendationSchema(BaseModel):
    disposal_bin: str
    prep: str
    notes: str


class TopKPredictionSchema(BaseModel):
    class_id: int
    class_name: str
    display_name: str
    confidence: float


class PredictionResponseSchema(BaseModel):
    log_id: Optional[int] = None
    created_at: Optional[datetime] = None

    model_name: str
    class_id: int
    class_name: str
    display_name: str
    confidence: float
    is_confident: bool
    warning: Optional[str]

    recommendation: RecommendationSchema
    top_k: List[TopKPredictionSchema]


class BatchPredictionResponseSchema(BaseModel):
    total: int
    items: List[PredictionResponseSchema]


class HealthResponseSchema(BaseModel):
    status: str = "ok"
    app_name: str = "SmartWaste AI API"
    version: str = "0.1.0"
    model_cache_size: int
    supported_classes: List[str]


class AnalyticsSummarySchema(BaseModel):
    total_predictions: int
    average_confidence: float
    low_confidence_count: int
    low_confidence_rate: float
    class_distribution: Dict[str, int]


class RecentPredictionSchema(BaseModel):
    id: int
    created_at: datetime
    image_name: str
    model_name: str
    class_name: str
    confidence: float
    is_confident: bool


class AnalyticsResponseSchema(BaseModel):
    summary: AnalyticsSummarySchema
    recent_predictions: List[RecentPredictionSchema] = Field(default_factory=list)
