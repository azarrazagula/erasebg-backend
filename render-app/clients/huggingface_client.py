import httpx
import logging
import asyncio
from config.settings import settings

logger = logging.getLogger(__name__)

class HuggingFaceClient:
    def __init__(self):
        self.base_url = settings.hf_space_url.rstrip('/')
        self.timeout = settings.hf_timeout
        self.max_retries = settings.hf_max_retries
        self.api_key = settings.hf_api_key

    def _get_headers(self) -> dict:
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    async def infer(self, file_bytes: bytes, filename: str, content_type: str) -> bytes:
        """Send the image to Hugging Face Space for background removal."""
        url = f"{self.base_url}/infer"
        
        files = {
            "file": (filename, file_bytes, content_type)
        }

        # We must use a fresh client for the request, or configure a persistent one. 
        # Using async with httpx.AsyncClient is safest here to avoid resource leaks.
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(1, self.max_retries + 1):
                try:
                    logger.info(f"[HF Client] Sending request to {url} (Attempt {attempt}/{self.max_retries})")
                    response = await client.post(
                        url,
                        files=files,
                        headers=self._get_headers()
                    )
                    
                    if response.status_code == 200:
                        return response.content
                    elif response.status_code == 401:
                        logger.error("[HF Client] Authentication failed. Check HF_API_KEY.")
                        raise Exception("Authentication failed with inference server.")
                    else:
                        logger.error(f"[HF Client] Error {response.status_code}: {response.text}")
                        # If it's a client error (e.g. bad request), don't retry
                        if 400 <= response.status_code < 500:
                            raise Exception(f"Inference error: {response.json().get('detail', 'Unknown error')}")
                        
                        raise Exception(f"Server returned status {response.status_code}")

                except (httpx.RequestError, httpx.TimeoutException) as e:
                    logger.error(f"[HF Client] Network error on attempt {attempt}: {str(e)}")
                    if attempt == self.max_retries:
                        raise Exception("Failed to connect to inference server after max retries.")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                except Exception as e:
                    if attempt == self.max_retries or "Authentication failed" in str(e) or "Inference error" in str(e):
                        raise e
                    await asyncio.sleep(2 ** attempt)
        
        raise Exception("Inference request failed.")
