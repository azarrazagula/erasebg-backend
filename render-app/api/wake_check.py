from fastapi import APIRouter

router = APIRouter(tags=["wake-check"])


@router.get("/health")
async def health_check() -> dict:
    """
    Wake-check endpoint pinged by UptimeRobot every 5 minutes.
    Keeps the Render free-tier server from going to sleep.
    """
    return {
        "status": "ok",
        "service": "api_gateway"
    }
