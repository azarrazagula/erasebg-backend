from fastapi import APIRouter

router = APIRouter(tags=["health"])

@router.get("/")
async def root() -> dict:
    """Root endpoint."""
    return {
        "message": "Erasebg Backend API Gateway",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "online"
    }

@router.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint for Render monitoring.
    """
    return {
        "status": "ok",
        "service": "api_gateway"
    }
