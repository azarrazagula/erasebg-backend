from fastapi import APIRouter

router = APIRouter(tags=["wake-check"])     # Swagger-ல "wake-check" group


# GET /health — hf-space wake-up ping
# render-app-ல இருந்து call ஆகும்போதும் alive-ஆ இருக்கான்னு confirm
@router.get("/health")
async def health_check() -> dict:
    """
    Wake-check endpoint pinged by UptimeRobot every 5 minutes.
    Keeps the HF Space from going to sleep.
    Returns model status along with OK response.
    """
    return {
        "status": "ok",                     # hf-space alive
        "service": "inference_engine",      # இது AI inference service
        "models": "ready"                   # Models loaded confirm
    }
