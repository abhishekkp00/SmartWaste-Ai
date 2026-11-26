"""FastAPI application entrypoint for SmartWaste backend."""

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

# Allow the local Vite development server to call the API from the browser.
cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api")
app.include_router(predict_router, prefix="/api")
app.include_router(analytics_router, prefix="/api")


@app.get("/")
async def root() -> dict:
    return {"message": "SmartWaste AI API is running", "docs": "/docs", "health": "/api/health"}
