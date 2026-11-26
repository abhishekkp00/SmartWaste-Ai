"""Image decoding and validation helpers for upload endpoints."""

from __future__ import annotations

from io import BytesIO

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from ml.config import cfg


async def read_and_validate_upload(file: UploadFile) -> Image.Image:
    """Validate content type/size and decode to RGB PIL image."""
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in cfg.inference.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type: {file.content_type}. Supported: {', '.join(cfg.inference.SUPPORTED_FORMATS)}",
        )

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(data) > cfg.inference.MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large ({len(data)} bytes). Max allowed is {cfg.inference.MAX_IMAGE_BYTES} bytes.",
        )

    try:
        image = Image.open(BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    return image
