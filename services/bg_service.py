import io
import asyncio
from typing import Tuple
from PIL import Image
import rembg


class BackgroundRemovalService:
    """Service for handling background removal operations."""
    
    @staticmethod
    async def remove_background(image_bytes: bytes) -> bytes:
        """
        Remove background from an image using rembg.
        
        Args:
            image_bytes: Image file contents as bytes
            
        Returns:
            PNG image with transparent background as bytes
            
        Raises:
            ValueError: If image processing fails
        """
        try:
            input_image = Image.open(io.BytesIO(image_bytes))
            
            input_image = input_image.convert("RGB")
            
            output_image = await asyncio.to_thread(
                rembg.remove,
                input_image,
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10
            )
            
            output_bytes = io.BytesIO()
            output_image.save(output_bytes, format="PNG")
            output_bytes.seek(0)
            
            return output_bytes.getvalue()
        
        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}")
