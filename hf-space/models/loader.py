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
    _instance = None            # Singleton instance
    _lock = threading.Lock()    # Thread-safe instance creation

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._sessions = {}            # model_name → rembg session cache
                cls._instance._sam_predictor = None     # SAM2 predictor cache
                cls._instance._sam_lock = asyncio.Lock()  # SAM2 load async lock
                cls._instance._init_env()
            return cls._instance  # எப்பவும் same instance return

    def _init_env(self):
        # rembg-ஐ local model_files/ directory-ல look பண்ண force பண்ணு
        # இல்லன்னா rembg internet-ல இருந்து download பண்ண முயற்சிக்கும்
        model_files_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_files"))
        os.environ["U2NET_HOME"] = model_files_dir
        logger.info(f"ModelLoader initialized. U2NET_HOME={model_files_dir}")

    def get_session(self, model_name: str):
        """Lazy loads and caches the rembg session for the requested model."""
        if model_name not in self._sessions:
            # Memory management: ஒரே ஒரு model மட்டும் RAM-ல வைக்கு
            # HF Space 16GB RAM — 2 models load பண்ணா OOM risk
            if len(self._sessions) > 0:
                import gc
                logger.info("Freeing previously loaded models to conserve memory...")
                # dict.clear() மட்டும் போதாது — Python GC immediately free பண்றதில்லை
                # Explicit del + gc.collect() — RAM உடனே free ஆகும்
                old_sessions = list(self._sessions.values())
                self._sessions.clear()
                for s in old_sessions:
                    try:
                        del s.inner_session  # ONNX InferenceSession free
                    except Exception:
                        pass
                    del s
                del old_sessions
                gc.collect()    # Python garbage collector manually trigger
                logger.info("Memory freed. Loading new model...")

            # Model file disk-ல இருக்கான்னு verify — இல்லன்னா runtime download block
            if not verify_model_exists(model_name):
                logger.error(f"Missing local model file for {model_name}.")
                raise FileNotFoundError(
                    f"Model file for {model_name} not found in model_files/. "
                    "Runtime downloads are strictly disabled."
                )
            
            logger.info(f"Loading local model into memory: {model_name}")
            from rembg import new_session

            # CPUExecutionProvider — Apple Silicon MPS JIT hang avoid
            # HF Space-ல GPU இல்லன்னா CPU safe default
            import onnxruntime as ort
            sess_opts = ort.SessionOptions()
            sess_opts.enable_cpu_mem_arena = False  # Memory arena disable — leak prevent
            sess_opts.intra_op_num_threads = 4      # Op-level parallelism: 4 threads
            sess_opts.inter_op_num_threads = 4      # Graph-level parallelism: 4 threads

            session = new_session(
                model_name,
                providers=["CPUExecutionProvider"],  # CPU மட்டும் use பண்ணு
                sess_options=sess_opts
            )
            
            # Monkeypatch: predict method-ல granular timing logs add பண்ணு
            # Pre-process / Forward pass / Post-process time separately track
            def timed_predict(img, *args, **kwargs):
                import time
                import numpy as np
                from PIL import Image

                t0 = time.perf_counter()
                # Image normalize பண்ணு (ImageNet mean/std, 1024x1024 resize)
                norm_inputs = session.normalize(
                    img,
                    (0.485, 0.456, 0.406),   # ImageNet mean (R, G, B)
                    (0.229, 0.224, 0.225),   # ImageNet std (R, G, B)
                    (1024, 1024),            # BiRefNet input size
                )
                t1 = time.perf_counter()
                # ONNX forward pass — actual AI computation
                ort_outs = session.inner_session.run(None, norm_inputs)
                t2 = time.perf_counter()
                # Sigmoid → normalize → squeeze → PIL mask
                pred = session.sigmoid(ort_outs[0][:, 0, :, :])
                ma = np.max(pred)
                mi = np.min(pred)
                pred = (pred - mi) / (ma - mi)      # 0-1 range normalize
                pred = np.squeeze(pred)              # Batch dim remove
                mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
                mask = mask.resize(img.size, Image.Resampling.LANCZOS)  # Original size-க்கு resize
                t3 = time.perf_counter()
                logger.info(f"Inference TIMING | Pre: {t1-t0:.3f}s | Fwd: {t2-t1:.3f}s | Post: {t3-t2:.3f}s")
                return [mask]

            # BiRefNet models மட்டும் timed_predict use பண்ணு
            if "birefnet" in model_name:
                session.predict = timed_predict

            self._sessions[model_name] = session    # Cache-ல store
            logger.info(f"Successfully loaded and cached: {model_name}")

        return self._sessions[model_name]

    def preload_models(self):
        """Loads required models synchronously at startup."""
        for model in ["birefnet-general", "birefnet-portrait"]:
            try:
                self.get_session(model)     # Cache-ல load பண்ணு
            except FileNotFoundError as e:
                logger.warning(str(e))      # File இல்லன்னா warn பண்ணு, crash ஆகாது

    def run_warmup(self):
        """Fires dummy inference to compile JIT/MPS kernels."""
        from rembg import remove
        dummy = Image.new("RGB", (256, 256), (128, 128, 128))  # Blank grey image
        for model_name, session in self._sessions.items():
            logger.info(f"Warming up: {model_name}...")
            try:
                remove(dummy, session=session, alpha_matting=False)  # Dummy run
            except Exception as exc:
                logger.error(f"Warmup failed for {model_name}: {exc}")
        logger.info("Warmup complete.")

    def clear_mps_cache(self):
        """Apple Silicon MPS GPU memory free பண்ணு."""
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()     # MPS memory pool clear
        except ImportError:
            pass    # torch இல்லன்னா skip

    async def get_sam2_predictor(self):
        """SAM2 model lazy load — first call-ல மட்டும் download + load ஆகும்."""
        import urllib.request
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        async with self._sam_lock:  # Concurrent calls-ல double load prevent
            if self._sam_predictor is None:
                logger.info("Loading SAM 2 model...")
                checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_files"))
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, "sam2_hiera_base_plus.pt")
                checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"

                # SAM2 checkpoint இல்லன்னா download பண்ணு
                if not os.path.exists(checkpoint_path):
                    logger.info(f"Downloading SAM 2 checkpoint to {checkpoint_path}...")
                    await asyncio.to_thread(urllib.request.urlretrieve, checkpoint_url, checkpoint_path)
                    logger.info("Download complete.")

                def _load():
                    # Device auto-detect: CUDA > MPS > CPU
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
