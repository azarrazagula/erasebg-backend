from fastapi import APIRouter

router = APIRouter(tags=["wake-check"])  # Swagger-ல "wake-check" group-ல தெரியும்


# GET /health — UptimeRobot every 5 min இந்த endpoint-ஐ ping பண்ணும்
# Render free tier-ல 15 min idle ஆனா server sleep ஆகும்
# இந்த ping மூலம் server எப்பவும் active-ஆ இருக்கும்
@router.get("/health")
async def health_check() -> dict:
    """
    Wake-check endpoint pinged by UptimeRobot every 5 minutes.
    Keeps the Render free-tier server from going to sleep.
    """
    return {
        "status": "ok",                 # Server alive confirm
        "service": "api_gateway"        # இது render-app-ன் health check
    }
