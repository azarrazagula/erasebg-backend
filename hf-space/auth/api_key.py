from fastapi import Header, HTTPException, status
from config.settings import settings

async def verify_api_key(x_api_key: str = Header(None)):
    """Verifies the X-API-Key header matches the HF_API_KEY environment variable."""
    if not settings.hf_api_key:
        return  # No API key configured, allow access
        
    if not x_api_key or x_api_key != settings.hf_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )
