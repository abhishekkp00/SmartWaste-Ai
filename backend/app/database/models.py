"""SQLAlchemy ORM models for SmartWaste backend."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base declarative class."""


class PredictionLog(Base):
    """Persisted prediction history for analytics and audit."""

    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    model_name: Mapped[str] = mapped_column(String(64), default="custom_cnn")
    image_name: Mapped[str] = mapped_column(String(255))

    class_id: Mapped[int] = mapped_column(Integer)
    class_name: Mapped[str] = mapped_column(String(64), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    is_confident: Mapped[bool] = mapped_column(Boolean)
    warning: Mapped[str | None] = mapped_column(Text, nullable=True)

    top_k_json: Mapped[str] = mapped_column(Text)
