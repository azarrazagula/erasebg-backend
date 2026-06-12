from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from fastapi.responses import Response
import logging

from auth.api_key import verify_api_key             # API Key dependency
from services.inference_service import InferenceService  # Full AI pipeline

logger = logging.getLogger(__name__)
router = APIRouter(tags=["inference"])

# POST /infer — hf-space main endpoint
# dependencies=[Depends(verify_api_key)] — X-API-Key header verify பண்ணியே request accept
@router.post("/infer", response_class=Response, dependencies=[Depends(verify_api_key)])
async def infer(file: UploadFile = File(...)) -> Response:
    """
    Accepts an image, runs the full AI background removal pipeline, 
    and returns a PNG image. Protected by API key.
    """
    try:
        image_bytes = await file.read()     # Image file bytes-ஆ read பண்ணு
        if not image_bytes:
            raise ValueError("Empty file uploaded")     # Empty file → reject
            
        # Full pipeline: Analyze → Route → Infer → Encode
        result_bytes = await InferenceService.process_image(image_bytes)
        
        # Transparent PNG return பண்ணு
        return Response(
            content=result_bytes,
            media_type="image/png"
        )
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}"
        )
