from clients.huggingface_client import HuggingFaceClient
import logging

logger = logging.getLogger(__name__)

class BackgroundRemovalService:
    @staticmethod
    async def remove_background(file_bytes: bytes, filename: str, content_type: str) -> bytes:
        """
        Orchestrates the background removal process by delegating to the Hugging Face Space.
        """
        client = HuggingFaceClient()
        try:
            logger.info("Delegating background removal to Hugging Face inference engine...")
            result_bytes = await client.infer(file_bytes, filename, content_type)
            logger.info("Successfully received processed image from inference engine.")
            return result_bytes
        except Exception as e:
            logger.error(f"BackgroundRemovalService Error: {str(e)}")
            raise e
