from fastapi import APIRouter

router = APIRouter(tags=["server-info"])


@router.get("/")
async def root() -> dict:
    """
    Root endpoint — returns service info when someone visits the base URL.
    Useful for confirming the inference engine is running.
    """
    return {
        "message": "Erasebg Inference Space",
        "status": "online"
    }
