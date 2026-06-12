from clients.huggingface_client import HuggingFaceClient   # HTTP client — hf-space-ஐ call பண்ணும்
import logging

logger = logging.getLogger(__name__)

class BackgroundRemovalService:
    @staticmethod
    async def remove_background(file_bytes: bytes, filename: str, content_type: str) -> bytes:
        """
        Orchestrates the background removal process by delegating to the Hugging Face Space.
        """
        client = HuggingFaceClient()    # HF Space client instance உருவாக்கு
        try:
            logger.info("Delegating background removal to Hugging Face inference engine...")
            # client.infer() — hf-space POST /infer-க்கு file அனுப்பு, result வாங்கு
            result_bytes = await client.infer(file_bytes, filename, content_type)
            logger.info("Successfully received processed image from inference engine.")
            return result_bytes         # Transparent PNG bytes return பண்ணு
        except Exception as e:
            logger.error(f"BackgroundRemovalService Error: {str(e)}")
            raise e                     # Error-ஐ caller-க்கு (remove_bg.py) propagate பண்ணு
