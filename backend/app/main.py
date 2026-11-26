"""FastAPI application entrypoint for SmartWaste backend."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .database.init_db import init_db
from .routes.analytics import router as analytics_router
from .routes.health import router as health_router
from .routes.predict import router as predict_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="SmartWaste AI API",
    description="Waste classification backend with prediction history and analytics.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health_router, prefix="/api")
app.include_router(predict_router, prefix="/api")
app.include_router(analytics_router, prefix="/api")


@app.get("/")
async def root() -> dict:
    return {"message": "SmartWaste AI API is running", "docs": "/docs", "health": "/api/health"}
