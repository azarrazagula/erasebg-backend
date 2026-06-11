from fastapi import APIRouter

router = APIRouter(tags=["server-info"])


@router.get("/")
async def root() -> dict:
    """
    Root endpoint — returns API info when someone visits the base URL.
    Useful for quickly confirming the server is up and identifying the service.
    """
    return {
        "message": "Erasebg Backend API Gateway",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "online"
    }
