"""Prediction routes (single and batch)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.crud import create_prediction_log
from ..database.session import get_db_session
from ..schemas.prediction import BatchPredictionResponseSchema, PredictionResponseSchema
from ..services.model_service import model_service
from ..utils.image_utils import read_and_validate_upload

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("", response_model=PredictionResponseSchema)
async def predict_single(
    file: UploadFile = File(...),
    model_name: str = Form("custom_cnn"),
    top_k: int = Form(3),
    session: AsyncSession = Depends(get_db_session),
) -> PredictionResponseSchema:
    """Classify one image and save prediction to history."""
    image = await read_and_validate_upload(file)
    prediction = model_service.predict(image=image, model_name=model_name, top_k=top_k)

    saved = await create_prediction_log(
        session,
        image_name=file.filename or "uploaded_image",
        prediction=prediction,
    )

    prediction["log_id"] = saved.id
    prediction["created_at"] = saved.created_at
    return PredictionResponseSchema(**prediction)


@router.post("/batch", response_model=BatchPredictionResponseSchema)
async def predict_batch(
    files: list[UploadFile] = File(...),
    model_name: str = Form("custom_cnn"),
    top_k: int = Form(3),
    session: AsyncSession = Depends(get_db_session),
) -> BatchPredictionResponseSchema:
    """Classify multiple images and save each prediction."""
    items = []
    for file in files:
        image = await read_and_validate_upload(file)
        prediction = model_service.predict(image=image, model_name=model_name, top_k=top_k)

        saved = await create_prediction_log(
            session,
            image_name=file.filename or "uploaded_image",
            prediction=prediction,
        )
        prediction["log_id"] = saved.id
        prediction["created_at"] = saved.created_at
        items.append(PredictionResponseSchema(**prediction))

    return BatchPredictionResponseSchema(total=len(items), items=items)
