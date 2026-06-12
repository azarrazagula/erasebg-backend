from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import Response
import logging

from config.settings import settings                                        # File size, extensions settings
from services.background_removal_service import BackgroundRemovalService    # hf-space-க்கு delegate பண்ணும் service
from utils.file_helper import validate_file, get_file_bytes, FileValidationError  # File validation helpers

logger = logging.getLogger(__name__)
router = APIRouter(tags=["background-removal"])  # Swagger-ல "background-removal" group-ல தெரியும்

# POST /remove-bg — முக்கிய endpoint
# response_class=Response: JSON-ஐ விட binary PNG return பண்றதால் raw Response use
@router.post("/remove-bg", response_class=Response)
async def remove_background(
    file: UploadFile = File(...)    # multipart/form-data file upload, required field
) -> Response:
    """
    Remove background from an uploaded image.
    Delegates the heavy ML inference to the dedicated Hugging Face Space.
    """
    # .env-ல ALLOWED_EXTENSIONS="png,jpg,jpeg,webp" → list-ஆ split பண்ணு
    allowed_extensions = [ext.strip() for ext in settings.allowed_extensions.split(",")]
    max_size = settings.max_file_size   # 12582912 bytes = 12 MB

    # Step 1: File validate பண்ணு — extension, content-type, size check
    try:
        await validate_file(file, allowed_extensions, max_size)
    except FileValidationError as e:
        logger.warning(f"File validation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,    # 400 Bad Request return
            detail=str(e)
        )
    
    # Step 2: File-ஐ bytes-ஆ memory-ல read பண்ணு
    try:
        image_bytes = await get_file_bytes(file)
    except Exception as e:
        logger.error(f"Failed to read file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read uploaded file"
        )
    
    # Step 3: hf-space inference engine-க்கு forward பண்ணி processed image வாங்கு
    try:
        result_bytes = await BackgroundRemovalService.remove_background(
            file_bytes=image_bytes,
            filename=file.filename or "image.png",          # filename இல்லன்னா default
            content_type=file.content_type or "image/png"  # content-type இல்லன்னா default
        )
    except Exception as e:
        logger.error(f"Background removal failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )

    # Step 4: Processed PNG-ஐ browser-க்கு return பண்ணு
    return Response(
        content=result_bytes,
        media_type="image/png",     # PNG with transparent background
        headers={
            "Content-Disposition": "attachment; filename=output.png",          # Download trigger
            "Access-Control-Expose-Headers": "Content-Disposition"             # Frontend-க்கு header visible ஆகணும்
        }
    )
