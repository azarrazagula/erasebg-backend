import io
import asyncio
import logging
from PIL import Image

from services.image_analyzer import ImageAnalyzer       # Image metrics எடு (skin ratio, face, etc.)
from services.router import SmartRouter                  # Metrics பார்த்து model decide பண்ணு
from services.executor import SegmentationPipeline       # Actual model run பண்ணு
from models.loader import ModelLoader                    # ONNX session manage பண்ணு

logger = logging.getLogger(__name__)

class InferenceService:
    @staticmethod
    async def process_image(image_bytes: bytes) -> bytes:
        """
        Runs the full AI pipeline:
        Analyze -> Route -> Infer -> Encode
        """
        import time
        t_start = time.perf_counter()   # Total time tracking
        
        # bytes → PIL Image
        input_img = Image.open(io.BytesIO(image_bytes))
        width, height = input_img.size
        
        # Step 1: Analyze — skin ratio, face, text, edge density எல்லாம் measure பண்ணு
        metrics = ImageAnalyzer.analyze(input_img)
        
        # Step 2: Route — metrics பார்த்து ROUTE_PORTRAIT / ROUTE_GRAPHIC / ROUTE_SIMPLE decide
        route = SmartRouter.route(metrics)
        logger.info(f"Selected Route: {route} | Original Res: {width}x{height}")
        
        # Step 3: Process — route-க்கு ஏத்த model use பண்ணி background remove பண்ணு
        loader = ModelLoader()          # Singleton — already loaded session reuse
        pipeline = SegmentationPipeline(loader)
        output_img, model_name, processing_time = await pipeline.process(input_img, route)
        
        logger.info(f"Used Model: {model_name} | AI Processing Time: {processing_time:.2f}s")
        
        # Step 4: Encode — PIL Image → PNG bytes (optimize=True → smaller file size)
        buf = io.BytesIO()
        await asyncio.to_thread(output_img.save, buf, format="PNG", optimize=True)
        buf.seek(0)     # Buffer cursor-ஐ beginning-க்கு reset
        
        total_time = time.perf_counter() - t_start
        logger.info(f"Total Inference Pipeline Time: {total_time:.2f}s")
        
        return buf.getvalue()   # PNG bytes return பண்ணு
