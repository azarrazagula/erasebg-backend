import io
import asyncio
import logging
from PIL import Image

from services.image_analyzer import ImageAnalyzer
from services.router import SmartRouter
from services.executor import SegmentationPipeline
from models.loader import ModelLoader

logger = logging.getLogger(__name__)

class InferenceService:
    @staticmethod
    async def process_image(image_bytes: bytes) -> bytes:
        """
        Runs the full AI pipeline:
        Analyze -> Route -> Infer -> Encode
        """
        import time
        t_start = time.perf_counter()
        
        input_img = Image.open(io.BytesIO(image_bytes))
        width, height = input_img.size
        
        # 1. Analyze
        metrics = ImageAnalyzer.analyze(input_img)
        
        # 2. Route
        route = SmartRouter.route(metrics)
        logger.info(f"Selected Route: {route} | Original Res: {width}x{height}")
        
        # 3. Process
        loader = ModelLoader()
        pipeline = SegmentationPipeline(loader)  # We update executor.py to accept loader instead of model_manager
        output_img, model_name, processing_time = await pipeline.process(input_img, route)
        
        logger.info(f"Used Model: {model_name} | AI Processing Time: {processing_time:.2f}s")
        
        # 4. Encode
        buf = io.BytesIO()
        await asyncio.to_thread(output_img.save, buf, format="PNG", optimize=True)
        buf.seek(0)
        
        total_time = time.perf_counter() - t_start
        logger.info(f"Total Inference Pipeline Time: {total_time:.2f}s")
        
        return buf.getvalue()
