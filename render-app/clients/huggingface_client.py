import httpx        # Async HTTP client — requests library-ஓட async version
import logging
import asyncio
from config.settings import settings

logger = logging.getLogger(__name__)

class HuggingFaceClient:
    def __init__(self):
        self.base_url = settings.hf_space_url.rstrip('/')  # Trailing slash remove பண்ணு
        self.timeout = settings.hf_timeout                  # 60 seconds max wait
        self.max_retries = settings.hf_max_retries          # 3 retries
        self.api_key = settings.hf_api_key                  # Secret key for auth

    def _get_headers(self) -> dict:
        """hf-space authenticate பண்ண X-API-Key header உருவாக்கு."""
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key   # hf-space-ல verify_api_key இதை check பண்ணும்
        return headers

    async def infer(self, file_bytes: bytes, filename: str, content_type: str) -> bytes:
        """Send the image to Hugging Face Space for background removal."""
        url = f"{self.base_url}/infer"  # hf-space POST /infer endpoint
        
        # multipart/form-data format-ல file pack பண்ணு
        files = {
            "file": (filename, file_bytes, content_type)
        }

        # AsyncClient — connection leak avoid பண்ண `async with` use பண்ணோம்
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(1, self.max_retries + 1):
                try:
                    logger.info(f"[HF Client] Sending request to {url} (Attempt {attempt}/{self.max_retries})")
                    response = await client.post(
                        url,
                        files=files,
                        headers=self._get_headers()     # API Key header attach பண்ணு
                    )
                    
                    if response.status_code == 200:
                        return response.content         # PNG bytes return பண்ணு
                    elif response.status_code == 401:
                        # API Key தப்பா இருந்தா — retry பண்ண வேண்டாம், directly fail
                        logger.error("[HF Client] Authentication failed. Check HF_API_KEY.")
                        raise Exception("Authentication failed with inference server.")
                    else:
                        logger.error(f"[HF Client] Error {response.status_code}: {response.text}")
                        # 4xx errors (bad request) — retry பண்ண வேண்டாம்
                        if 400 <= response.status_code < 500:
                            raise Exception(f"Inference error: {response.json().get('detail', 'Unknown error')}")
                        
                        raise Exception(f"Server returned status {response.status_code}")

                except (httpx.RequestError, httpx.TimeoutException) as e:
                    # Network error அல்லது timeout — retry eligible
                    logger.error(f"[HF Client] Network error on attempt {attempt}: {str(e)}")
                    if attempt == self.max_retries:
                        raise Exception("Failed to connect to inference server after max retries.")
                    # Exponential backoff: attempt 1 → 2s, attempt 2 → 4s, attempt 3 → 8s
                    await asyncio.sleep(2 ** attempt)
                except Exception as e:
                    # Auth/inference errors → immediately raise; others → retry with backoff
                    if attempt == self.max_retries or "Authentication failed" in str(e) or "Inference error" in str(e):
                        raise e
                    await asyncio.sleep(2 ** attempt)
        
        raise Exception("Inference request failed.")
