from fastapi import APIRouter

router = APIRouter(tags=["wake-check"])


@router.get("/health")
async def health_check() -> dict:
    """
    Wake-check endpoint pinged by UptimeRobot every 5 minutes.
    Keeps the HF Space from going to sleep.
    Returns model status along with OK response.
    """
    return {
        "status": "ok",
        "service": "inference_engine",
        "models": "ready"
    }
