from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import StreamingResponse
import io
import time
import logging

from services.main_bg_service import BackgroundRemovalService, PerformanceTracker
from utils.file_helper import validate_file, get_file_bytes, FileValidationError


logger = logging.getLogger(__name__)
router = APIRouter(tags=["background-removal"])


@router.post("/remove-bg", response_class=StreamingResponse)
async def remove_background(
    file: UploadFile = File(...),
    allowed_extensions: list[str] = ["png", "jpg", "jpeg", "webp"],
    max_size: int = 12582912
) -> StreamingResponse:
    """
    Remove background from an uploaded image.
    
    - **file**: Image file (PNG, JPG, WEBP) - max 12MB
    - **Returns**: PNG image with transparent background
    
    Status Codes:
    - 200: Success - returns PNG image
    - 400: Invalid file type or size
    - 413: File too large
    - 500: Processing error
    """
    try:
        await validate_file(file, allowed_extensions, max_size)
    except FileValidationError as e:
        logger.warning(f"File validation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    try:
        image_bytes = await get_file_bytes(file)
    except Exception as e:
        logger.error(f"Failed to read file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read uploaded file"
        )
    
    api_start = time.perf_counter()
    try:
        result_bytes, processing_time = await BackgroundRemovalService.remove_background(image_bytes)
    except ValueError as e:
        logger.error(f"Background removal failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process image. Please ensure the file is a valid image."
        )
    except Exception as e:
        logger.error(f"Unexpected error during background removal: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the image"
        )

    api_time = time.perf_counter() - api_start
    print(f"[API] Total Request Time: {api_time:.2f}s")

    PerformanceTracker().record(processing_time, api_time)

    return StreamingResponse(
        iter([result_bytes]),
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=output.png"}
    )


@router.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint.
    
    Returns:
    - status: "ok" if service is healthy
    - model: "rembg" indicating the AI model in use
    """
    return {
        "status": "ok",
        "model": "rembg"
    }
