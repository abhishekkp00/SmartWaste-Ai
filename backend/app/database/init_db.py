"""Database bootstrap helpers."""

from .models import Base
from .session import engine


async def init_db() -> None:
    """Create all tables if they do not already exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
