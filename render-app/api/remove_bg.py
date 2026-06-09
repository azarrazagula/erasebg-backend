from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import Response
import logging

from config.settings import settings
from services.background_removal_service import BackgroundRemovalService
from utils.file_helper import validate_file, get_file_bytes, FileValidationError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["background-removal"])

@router.post("/remove-bg", response_class=Response)
async def remove_background(
    file: UploadFile = File(...)
) -> Response:
    """
    Remove background from an uploaded image.
    Delegates the heavy ML inference to the dedicated Hugging Face Space.
    """
    allowed_extensions = [ext.strip() for ext in settings.allowed_extensions.split(",")]
    max_size = settings.max_file_size

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
    
    try:
        result_bytes = await BackgroundRemovalService.remove_background(
            file_bytes=image_bytes,
            filename=file.filename or "image.png",
            content_type=file.content_type or "image/png"
        )
    except Exception as e:
        logger.error(f"Background removal failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )

    return Response(
        content=result_bytes,
        media_type="image/png",
        headers={
            "Content-Disposition": "attachment; filename=output.png",
            "Access-Control-Expose-Headers": "Content-Disposition"
        }
    )
