from fastapi import APIRouter

router = APIRouter(tags=["server-info"])    # Swagger-ல "server-info" group


# GET / — hf-space base URL visit பண்ணும்போது service alive confirm
@router.get("/")
async def root() -> dict:
    """
    Root endpoint — returns service info when someone visits the base URL.
    Useful for confirming the inference engine is running.
    """
    return {
        "message": "Erasebg Inference Space",  # இது hf-space service
        "status": "online"                       # Alive confirm
    }
