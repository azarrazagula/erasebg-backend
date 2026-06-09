from fastapi import APIRouter

router = APIRouter(tags=["health"])

@router.get("/health")
async def health_check() -> dict:
    """
    UptimeRobot compatible health endpoint.
    Returns 200 OK and model status.
    """
    return {
        "status": "ok",
        "service": "inference_engine",
        "models": "ready"
    }

@router.get("/")
async def root() -> dict:
    return {
        "message": "Erasebg Inference Space",
        "status": "online"
    }
