"""Database session and engine configuration for SmartWaste backend."""

from __future__ import annotations

from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATABASE_URL = f"sqlite+aiosqlite:///{(PROJECT_ROOT / 'smartwaste.db').as_posix()}"

engine = create_async_engine(DATABASE_URL, future=True, echo=False)
AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield one async SQLAlchemy session per request."""
    async with AsyncSessionLocal() as session:
        yield session
