import os
import asyncio
import threading
import urllib.request
import torch
from rembg import new_session
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class ModelManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance._init_manager()
            return cls._instance
            
    def _init_manager(self):
        self._sessions = {}
        self._sam_predictor = None
        self._sam_lock = asyncio.Lock()
        
        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        print(f"[ModelManager] Using device: {self.device}")
        
    def get_birefnet_session(self, model_name: str):
        if model_name not in self._sessions:
            print(f"[ModelManager] Loading BiRefNet model: {model_name}")
            self._sessions[model_name] = new_session(model_name)
        return self._sessions[model_name]

    async def get_sam2_predictor(self):
        async with self._sam_lock:
            if self._sam_predictor is None:
                print("[ModelManager] Loading SAM 2 model...")
                
                # Checkpoint configuration
                checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, "sam2_hiera_base_plus.pt")
                checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"
                
                if not os.path.exists(checkpoint_path):
                    print(f"[ModelManager] Downloading SAM 2 checkpoint to {checkpoint_path}...")
                    await asyncio.to_thread(urllib.request.urlretrieve, checkpoint_url, checkpoint_path)
                    print("[ModelManager] Download complete.")

                model_cfg = "sam2_hiera_b+.yaml"
                
                # Load SAM2
                def _load():
                    # For Mac MPS, SAM2 might need autocast disabled or specific settings,
                    # but build_sam2 generally handles the device mapping.
                    return build_sam2(model_cfg, checkpoint_path, device=self.device)
                    
                sam2_model = await asyncio.to_thread(_load)
                self._sam_predictor = SAM2ImagePredictor(sam2_model)
                print("[ModelManager] SAM 2 loaded successfully.")
                
            return self._sam_predictor
