import io
import threading
import asyncio
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
        import time
        try:
            t_start = time.perf_counter()
            # Note: We do NOT downscale here. ImageAnalyzer downscales for 
            # metrics, and Executor downscales for inference, but both need
            # the original high-res image to work from.
            input_img = Image.open(io.BytesIO(image_bytes))
            width, height = input_img.size
            t_load = time.perf_counter()
            print(f"[Timing] Image Load Time: {t_load - t_start:.4f}s")

            # Analyze
            metrics = ImageAnalyzer.analyze(input_img)
            t_analyze = time.perf_counter()
            print(f"[Timing] Analyze Time: {t_analyze - t_load:.4f}s")

            # Route
            route = SmartRouter.route(metrics)
            t_route = time.perf_counter()
            print(f"[Timing] Route Time: {t_route - t_analyze:.4f}s")

            # Process
            mm = ModelManager()
            pipeline = SegmentationPipeline(mm)
            output_img, model_name, processing_time = await pipeline.process(input_img, route)
            t_process = time.perf_counter()

            # Detailed logs
            print(f"[BG Service] Selected Route: {route}")
            print(f"[BG Service] Model: {model_name}")
            print(f"[BG Service] Original Resolution: {width}x{height}")
            print(f"[BG Service] Processing Time: {processing_time:.2f}s")
            print(f"[BG Service] Output mode={output_img.mode} size={output_img.size}")

            # Encode (Async to avoid blocking event loop on large PNGs)
            buf = io.BytesIO()
            await asyncio.to_thread(output_img.save, buf, format="PNG", optimize=True)
            buf.seek(0)
            t_save = time.perf_counter()
            print(f"[Timing] PNG Save Time: {t_save - t_process:.4f}s")
            print(f"[Timing] Total Time: {t_save - t_start:.4f}s")
            
            return buf.getvalue(), processing_time

        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}")
