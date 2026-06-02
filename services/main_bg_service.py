import io
import threading
from PIL import Image

from services.model_manager import ModelManager
from services.image_analyzer import ImageAnalyzer
from services.router import SmartRouter
from services.executor import SegmentationPipeline

class PerformanceTracker:
    """Thread-safe rolling performance statistics, printed every N requests."""
    _instance = None
    _lock = threading.Lock()
    LOG_EVERY = 10

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._processing_times: list[float] = []
                cls._instance._api_times: list[float] = []
                cls._instance._request_count = 0
            return cls._instance

    def record(self, processing_time: float, api_time: float) -> None:
        with self._lock:
            self._processing_times.append(processing_time)
            self._api_times.append(api_time)
            self._request_count += 1
            if self._request_count % self.LOG_EVERY == 0:
                avg_proc = sum(self._processing_times) / len(self._processing_times)
                avg_api  = sum(self._api_times)        / len(self._api_times)
                print(
                    f"[Perf] === Stats after {self._request_count} requests === "
                    f"Avg Processing: {avg_proc:.2f}s | Avg API: {avg_api:.2f}s"
                )


class BackgroundRemovalService:
    @staticmethod
    async def remove_background(image_bytes: bytes) -> tuple[bytes, float]:
        """
        Returns (result_png_bytes, processing_time_seconds).
        The caller is responsible for measuring total API time.
        """
        try:
            input_img = Image.open(io.BytesIO(image_bytes))
            width, height = input_img.size

            # Analyze
            metrics = ImageAnalyzer.analyze(input_img)

            # Route
            route = SmartRouter.route(metrics)

            # Process
            mm = ModelManager()
            pipeline = SegmentationPipeline(mm)
            output_img, model_name, processing_time = await pipeline.process(input_img, route)

            # Detailed logs
            print(f"[BG Service] Selected Route: {route}")
            print(f"[BG Service] Model: {model_name}")
            print(f"[BG Service] Resolution: {width}x{height}")
            print(f"[BG Service] Processing Time: {processing_time:.2f}s")
            print(f"[BG Service] Done → mode={output_img.mode} size={output_img.size}")

            # Encode
            buf = io.BytesIO()
            output_img.save(buf, format="PNG")
            buf.seek(0)
            return buf.getvalue(), processing_time

        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}")
