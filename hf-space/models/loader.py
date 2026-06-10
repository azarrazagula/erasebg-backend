import asyncio
import threading
import logging
import os
from PIL import Image

from config.settings import settings
from models.registry import verify_model_exists, MODEL_REGISTRY

logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Singleton class that handles lazy loading of ONNX models into memory,
    strictly from the local model_files directory to prevent runtime downloads.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._sessions = {}
                cls._instance._sam_predictor = None
                cls._instance._sam_lock = asyncio.Lock()
                cls._instance._init_env()
            return cls._instance

    def _init_env(self):
        # We explicitly enforce rembg to look inside our model_files directory.
        model_files_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_files"))
        os.environ["U2NET_HOME"] = model_files_dir
        logger.info(f"ModelLoader initialized. U2NET_HOME={model_files_dir}")

    def get_session(self, model_name: str):
        """Lazy loads and caches the rembg session for the requested model."""
        if model_name not in self._sessions:
            # FREE MEMORY: Only keep ONE model loaded at a time to prevent 16GB OOM on HF Space
            if len(self._sessions) > 0:
                import gc
                logger.info("Freeing previously loaded models to conserve memory...")
                # Explicit delete + gc.collect() ensures memory is released BEFORE loading new model.
                # dict.clear() alone does NOT guarantee immediate RAM free in Python.
                old_sessions = list(self._sessions.values())
                self._sessions.clear()
                for s in old_sessions:
                    try:
                        del s.inner_session
                    except Exception:
                        pass
                    del s
                del old_sessions
                gc.collect()
                logger.info("Memory freed. Loading new model...")

            if not verify_model_exists(model_name):
                logger.error(f"Missing local model file for {model_name}.")
                raise FileNotFoundError(
                    f"Model file for {model_name} not found in model_files/. "
                    "Runtime downloads are strictly disabled."
                )
            
            logger.info(f"Loading local model into memory: {model_name}")
            from rembg import new_session

            # Use CPUExecutionProvider to avoid Apple Silicon JIT hangs, 
            # and as safe default for hugging face space unless specific hardware is available.
            import onnxruntime as ort
            sess_opts = ort.SessionOptions()
            sess_opts.enable_cpu_mem_arena = False
            sess_opts.intra_op_num_threads = 4
            sess_opts.inter_op_num_threads = 4

            session = new_session(
                model_name,
                providers=["CPUExecutionProvider"],
                sess_options=sess_opts
            )
            
            # Monkeypatch predict to add granular timing logs
            def timed_predict(img, *args, **kwargs):
                import time
                import numpy as np
                from PIL import Image

                t0 = time.perf_counter()
                norm_inputs = session.normalize(
                    img,
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                    (1024, 1024),
                )
                t1 = time.perf_counter()
                ort_outs = session.inner_session.run(None, norm_inputs)
                t2 = time.perf_counter()
                pred = session.sigmoid(ort_outs[0][:, 0, :, :])
                ma = np.max(pred)
                mi = np.min(pred)
                pred = (pred - mi) / (ma - mi)
                pred = np.squeeze(pred)
                mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
                mask = mask.resize(img.size, Image.Resampling.LANCZOS)
                t3 = time.perf_counter()
                logger.info(f"Inference TIMING | Pre: {t1-t0:.3f}s | Fwd: {t2-t1:.3f}s | Post: {t3-t2:.3f}s")
                return [mask]

            if "birefnet" in model_name:
                session.predict = timed_predict

            self._sessions[model_name] = session
            logger.info(f"Successfully loaded and cached: {model_name}")

        return self._sessions[model_name]

    def preload_models(self):
        """Loads required models synchronously at startup."""
        for model in ["birefnet-general", "birefnet-portrait"]:
            try:
                self.get_session(model)
            except FileNotFoundError as e:
                logger.warning(str(e))

    def run_warmup(self):
        """Fires dummy inference to compile JIT/MPS kernels."""
        from rembg import remove
        dummy = Image.new("RGB", (256, 256), (128, 128, 128))
        for model_name, session in self._sessions.items():
            logger.info(f"Warming up: {model_name}...")
            try:
                remove(dummy, session=session, alpha_matting=False)
            except Exception as exc:
                logger.error(f"Warmup failed for {model_name}: {exc}")
        logger.info("Warmup complete.")

    def clear_mps_cache(self):
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass

    async def get_sam2_predictor(self):
        import urllib.request
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        async with self._sam_lock:
            if self._sam_predictor is None:
                logger.info("Loading SAM 2 model...")
                checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_files"))
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, "sam2_hiera_base_plus.pt")
                checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"

                if not os.path.exists(checkpoint_path):
                    logger.info(f"Downloading SAM 2 checkpoint to {checkpoint_path}...")
                    await asyncio.to_thread(urllib.request.urlretrieve, checkpoint_url, checkpoint_path)
                    logger.info("Download complete.")

                def _load():
                    # Simplified device handling for SAM2
                    device = "cpu"
                    try:
                        import torch
                        if torch.cuda.is_available(): device = "cuda"
                        elif torch.backends.mps.is_available(): device = "mps"
                    except ImportError:
                        pass
                    return build_sam2("sam2_hiera_b+.yaml", checkpoint_path, device=device)

                sam2_model = await asyncio.to_thread(_load)
                self._sam_predictor = SAM2ImagePredictor(sam2_model)
                logger.info("SAM 2 loaded successfully.")

            return self._sam_predictor
