from fastapi import APIRouter

router = APIRouter(tags=["server-info"])    # Swagger-ல "server-info" group-ல தெரியும்


# GET / — root endpoint
# யாராவது base URL-ஐ visit பண்ணும்போது server alive-ஆ இருக்கான்னு confirm பண்ணும்
@router.get("/")
async def root() -> dict:
    """
    Root endpoint — returns API info when someone visits the base URL.
    Useful for quickly confirming the server is up and identifying the service.
    """
    return {
        "message": "Erasebg Backend API Gateway",   # Service பேர்
        "version": "1.0.0",                          # API version
        "docs": "/docs",                             # Swagger UI link
        "status": "online"                           # Server running confirm
    }
