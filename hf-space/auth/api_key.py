from fastapi import Header, HTTPException, status
from config.settings import settings

async def verify_api_key(x_api_key: str = Header(None)):
    """Verifies the X-API-Key header matches the HF_API_KEY environment variable."""
    
    # HF_API_KEY configure பண்ணல-ன்னா — open access (local dev mode)
    if not settings.hf_api_key:
        return  # No API key configured, allow access
        
    # Header இல்லன்னா அல்லது key தப்பா இருந்தா → 401 Unauthorized
    # render-app மட்டும் correct key-யோட call பண்ணும், others block ஆகும்
    if not x_api_key or x_api_key != settings.hf_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )
