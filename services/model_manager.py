import os
import asyncio
import threading

import numpy as np
import torch
from PIL import Image
from rembg import new_session, remove


class ModelManager:
    """
    Singleton model manager for BiRefNet sessions and (optional) SAM2.

    Key improvements over the original:
      - preload_all()   : loads both BiRefNet models synchronously at startup.
      - _run_warmup()   : fires dummy inference to trigger MPS JIT compilation
                          before the first real request arrives.
      - _clear_mps_cache(): releases MPS memory between inferences to avoid
                            fragmentation on long-running processes.
      - SAM2 is still supported but disabled by default (settings.sam2_disabled).
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_manager()
            return cls._instance

    # ── Initialisation ──────────────────────────────────────────────────────

    def _init_manager(self) -> None:
        self._sessions: dict = {}
        self._sam_predictor = None
        self._sam_lock = asyncio.Lock()

        # Device selection (CUDA > MPS > CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"[ModelManager] Using device: {self.device}")

    # ── Startup preloading & warmup ─────────────────────────────────────────

    def preload_all(self) -> None:
        """
        Load all BiRefNet models into memory synchronously.
        Call once at application startup (inside the lifespan handler).
        """
        # Importing here to keep the top-level import fast if not needed
        from config import settings  # avoid circular at module level

        print("[ModelManager] Preloading BiRefNet models...")
        self.get_birefnet_session("birefnet-general")
        self.get_birefnet_session("birefnet-portrait")
        print("[ModelManager] All models loaded.")

        if settings.warmup_on_startup:
            self._run_warmup()

    def _run_warmup(self) -> None:
        """
        Run a small dummy inference through every loaded model to trigger
        MPS kernel compilation.  This eliminates the 15-30s JIT cold-start
        that otherwise hits the first real request.
        """
        dummy = Image.new("RGB", (256, 256), (128, 128, 128))
        for model_name, session in self._sessions.items():
            print(f"[ModelManager] Warming up: {model_name}...")
            try:
                remove(dummy, session=session, alpha_matting=False)
            except Exception as exc:
                print(f"[ModelManager] Warmup failed for {model_name}: {exc}")
        self._clear_mps_cache()
        print("[ModelManager] Warmup complete — MPS kernels compiled.")

    # ── Model accessors ─────────────────────────────────────────────────────

    def get_birefnet_session(self, model_name: str):
        """Return (and lazily load) a rembg session for the given model."""
        if model_name not in self._sessions:
            print(f"[ModelManager] Loading BiRefNet model: {model_name}")

            # Note: CoreMLExecutionProvider causes ANECompilerService hangs on Apple Silicon
            # due to massive JIT compilation (241 partitions) for BiRefNet. 
            # Defaulting to CPUExecutionProvider is drastically faster and safer.
            session = new_session(
                model_name,
                providers=["CPUExecutionProvider"]
            )

            print(
                "[ModelManager] PROVIDERS =",
                session.inner_session.get_providers()
            )

            # Monkeypatch predict to add granular timing logs inside inference
            def timed_predict(img, *args, **kwargs):
                import time
                import numpy as np
                from PIL import Image

                t0 = time.perf_counter()

                # 1. Preprocess
                norm_inputs = session.normalize(
                    img,
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                    (1024, 1024),
                )

                t1 = time.perf_counter()
                print(f"[Timing] Inference Preprocess: {t1 - t0:.4f}s")

                # 2. Model Forward
                ort_outs = session.inner_session.run(None, norm_inputs)

                t2 = time.perf_counter()
                print(f"[Timing] Inference Model Forward: {t2 - t1:.4f}s")

                # 3. Postprocess
                pred = session.sigmoid(ort_outs[0][:, 0, :, :])

                ma = np.max(pred)
                mi = np.min(pred)

                pred = (pred - mi) / (ma - mi)
                pred = np.squeeze(pred)

                mask = Image.fromarray(
                    (pred * 255).astype("uint8"),
                    mode="L"
                )

                mask = mask.resize(
                    img.size,
                    Image.Resampling.LANCZOS
                )

                t3 = time.perf_counter()
                print(f"[Timing] Inference Postprocess: {t3 - t2:.4f}s")

                return [mask]

            if "birefnet" in model_name:
                session.predict = timed_predict

            self._sessions[model_name] = session
            print(f"[ModelManager] Loaded: {model_name}")

        return self._sessions[model_name]
    async def get_sam2_predictor(self):
        """
        Lazy-load SAM2 (guarded by sam2_disabled flag in settings).
        Import is deferred to avoid loading heavy SAM2 deps at startup.
        """
        import urllib.request
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        async with self._sam_lock:
            if self._sam_predictor is None:
                print("[ModelManager] Loading SAM 2 model...")

                checkpoint_dir = os.path.join(
                    os.path.dirname(__file__), "..", "checkpoints"
                )
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(
                    checkpoint_dir, "sam2_hiera_base_plus.pt"
                )
                checkpoint_url = (
                    "https://dl.fbaipublicfiles.com/segment_anything_2/"
                    "072824/sam2_hiera_base_plus.pt"
                )

                if not os.path.exists(checkpoint_path):
                    print(
                        f"[ModelManager] Downloading SAM 2 checkpoint to "
                        f"{checkpoint_path}..."
                    )
                    await asyncio.to_thread(
                        urllib.request.urlretrieve, checkpoint_url, checkpoint_path
                    )
                    print("[ModelManager] Download complete.")

                def _load():
                    return build_sam2(
                        "sam2_hiera_b+.yaml", checkpoint_path, device=self.device
                    )

                sam2_model = await asyncio.to_thread(_load)
                self._sam_predictor = SAM2ImagePredictor(sam2_model)
                print("[ModelManager] SAM 2 loaded successfully.")

            return self._sam_predictor

    # ── MPS memory management ────────────────────────────────────────────────

    def _clear_mps_cache(self) -> None:
        """
        Release unused MPS memory.  Call after each inference to prevent
        fragmentation on long-running processes.
        """
        if self.device.type == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass  # Available from PyTorch 2.0; safe to ignore if older
