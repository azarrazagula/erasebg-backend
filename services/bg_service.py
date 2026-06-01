import io
import os
import time
import asyncio
import threading
import numpy as np
import cv2
import torch
import urllib.request
from PIL import Image
from rembg import remove, new_session
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from config import settings

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

class GraphicAssetDetector:
    @staticmethod
    def detect(cv_img: np.ndarray) -> float:
        # Scale down for speed
        h, w = cv_img.shape[:2]
        max_dim = 400
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            small_img = cv2.resize(cv_img, (int(w * scale), int(h * scale)))
        else:
            small_img = cv_img.copy()

        # Unique Colors (quantized)
        pixels = small_img.reshape(-1, 3)
        quantized = pixels // 32  # Group similar colors
        unique_colors = len(np.unique(quantized, axis=0))
        color_score = max(0.0, 100.0 - (unique_colors / 5.0)) # Graphics have fewer colors

        # Saturation Score
        hsv = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)
        mean_saturation = np.mean(hsv[:, :, 1])
        saturation_score = min(100.0, mean_saturation / 255.0 * 150.0)

        # Shape / Edge Density
        gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shape_density = len(contours) / (small_img.shape[0] * small_img.shape[1])
        shape_score = min(100.0, shape_density * 50000.0)

        # Weighted combination
        score = (color_score * 0.5) + (saturation_score * 0.2) + (shape_score * 0.3)
        return float(min(100.0, max(0.0, score)))


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


class ImageAnalyzer:
    @staticmethod
    def analyze(image: Image.Image) -> dict:
        rgb = image.convert("RGB")
        cv_img = np.array(rgb)
        cv_img = cv_img[:, :, ::-1].copy() # RGB to BGR
        
        # Basic dimensions
        h, w = cv_img.shape[:2]
        aspect_ratio = w / h if h > 0 else 1.0
        
        # Skin ratio (using HSV)
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_ratio = cv2.countNonZero(skin_mask) / (w * h)
        
        # Edge density (using Canny)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = cv2.countNonZero(edges) / (w * h)
        
        # Contrast and Brightness
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        
        # Complexity (heuristic combination)
        complexity = (edge_density * 1000) + (contrast / 2)
        
        # Coverage (heuristic based on central bounding box of edges)
        coords = cv2.findNonZero(edges)
        coverage = 0.5 # default
        if coords is not None:
            x, y, ew, eh = cv2.boundingRect(coords)
            coverage = (ew * eh) / (w * h)
            
        # Graphic Score
        graphic_score = GraphicAssetDetector.detect(cv_img)
        
        metrics = {
            "skin_ratio": skin_ratio,
            "brightness": brightness,
            "contrast": contrast,
            "aspect_ratio": aspect_ratio,
            "edge_density": edge_density,
            "complexity": complexity,
            "coverage": coverage,
            "graphic_score": graphic_score
        }
        print(f"[Analyze] Metrics: {metrics}")
        return metrics

class SmartRouter:
    @staticmethod
    def route(metrics: dict) -> str:
        # Route 0: Graphic (Priority)
        if metrics["graphic_score"] > settings.graphic_asset_threshold:
            print(f"[Router] graphic_score={metrics['graphic_score']:.1f} -> ROUTE_GRAPHIC")
            return "ROUTE_GRAPHIC"

        # Route 1: Portrait
        if metrics["skin_ratio"] > settings.portrait_skin_threshold:
            return "ROUTE_PORTRAIT"
            
        # Route 3: Complex
        if metrics["complexity"] > settings.sam2_complexity_threshold or metrics["coverage"] > settings.sam2_coverage_threshold:
            return "ROUTE_COMPLEX"
            
        # Route 2: Simple (Fallback)
        return "ROUTE_SIMPLE"

class SmallComponentRecovery:
    """
    Lightweight post-processing to recover small disconnected foreground
    components (dots, thin lines, decorative accents) that BiRefNet may
    have missed.  Uses only OpenCV + NumPy — no additional models.

    Only activated on ROUTE_GRAPHIC.  Controlled via config.py:
        recovery_enabled                 – master on/off switch
        recovery_min_area                – minimum pixel area to evaluate
        recovery_max_distance            – max distance from main fg (fraction of diagonal)
        recovery_color_similarity        – max L2 colour distance to fg mean
        recovery_edge_score_threshold    – min fraction of component pixels on edges
        recovery_compactness_threshold   – min 4π·area/perimeter² (filters amorphous noise)
    """

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _compactness(contour) -> float:
        """
        Isoperimetric compactness: 4π·Area / Perimeter².
        Circle → 1.0   |   thin line → ~0.0   |   random noise blob → very low.
        We KEEP components whose compactness is ABOVE the threshold,
        so decorative dots (high compactness) pass while diffuse noise (low) fails.
        """
        area = cv2.contourArea(contour)
        perim = cv2.arcLength(contour, closed=True)
        if perim < 1e-6:
            return 0.0
        return (4.0 * np.pi * area) / (perim ** 2)

    @staticmethod
    def _mean_fg_color(orig_bgr: np.ndarray, main_mask: np.ndarray) -> np.ndarray:
        """Global mean colour of the main foreground — used as colour fallback."""
        fg_pixels = orig_bgr[main_mask > 0]
        if len(fg_pixels) == 0:
            return np.array([128.0, 128.0, 128.0])
        return fg_pixels.mean(axis=0)

    # ── Main entry point ───────────────────────────────────────────────────────

    @staticmethod
    def recover(original_img: Image.Image, birefnet_rgba: Image.Image) -> Image.Image:
        """
        Args:
            original_img  : the original RGB/RGBA input image (before removal).
            birefnet_rgba : the RGBA output from BiRefNet (alpha = foreground mask).
        Returns:
            Refined RGBA image with recovered components merged into the mask.
        """
        if not settings.recovery_enabled:
            return birefnet_rgba

        # ── 1. Extract binary mask from BiRefNet alpha ─────────────────────────
        rgba_arr = np.array(birefnet_rgba.convert("RGBA"))
        alpha    = rgba_arr[:, :, 3]                        # uint8 [0-255]
        binary   = (alpha > 127).astype(np.uint8) * 255

        # ── 2. Prepare original image in BGR for colour comparisons ───────────
        orig_rgb = np.array(original_img.convert("RGB"))
        orig_bgr = orig_rgb[:, :, ::-1].copy()

        img_h, img_w = binary.shape
        diag        = float(np.sqrt(img_h ** 2 + img_w ** 2))
        max_dist_px = settings.recovery_max_distance * diag

        # ── 3. Connected-components analysis ──────────────────────────────────
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        # label 0 = background; find the *largest* foreground label as main fg
        fg_labels = list(range(1, num_labels))
        total_components = len(fg_labels)           # ← real total, before any filter

        if not fg_labels:
            print(f"[Recovery] Components Found: 0 | Evaluated: 0 | Recovered: 0 | Rejected: 0")
            return birefnet_rgba

        areas      = stats[1:, cv2.CC_STAT_AREA]    # skip background row
        main_label = int(np.argmax(areas)) + 1       # 1-indexed

        main_mask  = (labels == main_label).astype(np.uint8) * 255

        # Distance transform: every pixel gets its distance from the main fg blob
        dist_from_main = cv2.distanceTransform(
            cv2.bitwise_not(main_mask), cv2.DIST_L2, 5
        )

        # Edge map of the original image (used for edge-score test)
        gray_orig  = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
        edges_orig = cv2.Canny(gray_orig, 50, 150)

        # Global fg colour — used as fallback when local neighbourhood is empty
        global_fg_mean = SmallComponentRecovery._mean_fg_color(orig_bgr, main_mask)

        # ── 4. Filter: only evaluate components above min area ─────────────────
        small_labels = [
            lbl for lbl in fg_labels
            if lbl != main_label
            and stats[lbl, cv2.CC_STAT_AREA] >= settings.recovery_min_area
        ]
        evaluated = len(small_labels)

        refined_mask = binary.copy()
        recovered    = 0
        rejected     = 0

        for lbl in small_labels:
            area      = int(stats[lbl, cv2.CC_STAT_AREA])
            comp_mask = (labels == lbl)             # bool array, full image size

            # ── a) Distance to main foreground ─────────────────────────────────
            min_dist = float(np.min(dist_from_main[comp_mask]))
            if min_dist > max_dist_px:
                rejected += 1
                continue

            # ── b) Shape compactness (filters diffuse noise / random artifacts) ─
            x  = int(stats[lbl, cv2.CC_STAT_LEFT])
            y  = int(stats[lbl, cv2.CC_STAT_TOP])
            bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
            bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])

            comp_roi = comp_mask[y:y + bh, x:x + bw].astype(np.uint8) * 255
            contours, _ = cv2.findContours(comp_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                best_compactness = max(
                    SmallComponentRecovery._compactness(c) for c in contours
                )
            else:
                best_compactness = 0.0

            if best_compactness < settings.recovery_compactness_threshold:
                rejected += 1
                continue

            # ── c) Colour similarity to nearby foreground pixels ───────────────
            pad    = max(10, int(max(bw, bh) * 0.5))
            y1, y2 = max(0, y - pad), min(img_h, y + bh + pad)
            x1, x2 = max(0, x - pad), min(img_w, x + bw + pad)

            region_comp  = orig_bgr[comp_mask]                      # component pixels
            fg_neighbour = main_mask[y1:y2, x1:x2]

            if fg_neighbour.any():
                # Use local fg pixels from the neighbourhood
                fg_pixels = orig_bgr[y1:y2, x1:x2][fg_neighbour > 0]
                mean_fg   = fg_pixels.mean(axis=0)
            else:
                # Fallback: compare against the global foreground mean
                mean_fg = global_fg_mean

            mean_comp   = region_comp.mean(axis=0)
            colour_dist = float(np.linalg.norm(
                mean_comp.astype(float) - mean_fg.astype(float)
            ))

            if colour_dist > settings.recovery_color_similarity:
                rejected += 1
                continue

            # ── d) Edge score — component should sit on real edges ─────────────
            edge_region = edges_orig[y:y + bh, x:x + bw]
            comp_region = comp_mask[y:y + bh, x:x + bw]
            edge_pixels = int(edge_region[comp_region].sum()) // 255
            edge_score  = edge_pixels / max(area, 1)

            if edge_score < settings.recovery_edge_score_threshold:
                rejected += 1
                continue

            # ✅ All checks passed — recover this component
            refined_mask[comp_mask] = 255
            recovered += 1

        # ── 5. Logging ─────────────────────────────────────────────────────────
        print(f"[Recovery] Components Found: {total_components}")
        print(f"[Recovery] Evaluated (above min area): {evaluated}")
        print(f"[Recovery] Recovered: {recovered}")
        print(f"[Recovery] Rejected:  {rejected}")
        if recovered > 0:
            print("[Recovery] Final Mask Improved")

        # ── 6. Rebuild RGBA with refined mask ──────────────────────────────────
        orig_rgba          = np.array(original_img.convert("RGBA"))
        orig_rgba[:, :, 3] = refined_mask
        return Image.fromarray(orig_rgba, "RGBA")


class SegmentationPipeline:
    def __init__(self, model_manager: ModelManager):
        self.mm = model_manager

    async def run_birefnet(self, image: Image.Image, model_name: str, **kwargs) -> tuple[Image.Image, str, float]:
        """Returns (output_image, model_name, processing_time_seconds)."""
        session = self.mm.get_birefnet_session(model_name)

        matting_params = {"alpha_matting": False}
        matting_params.update(kwargs)

        start = time.perf_counter()
        output_img = await asyncio.to_thread(
            remove,
            image,
            session=session,
            **matting_params
        )
        elapsed = time.perf_counter() - start

        return output_img, model_name, elapsed

    async def run_sam2_complex(self, image: Image.Image) -> tuple[Image.Image, str, float]:
        predictor = await self.mm.get_sam2_predictor()
        cv_img = np.array(image.convert("RGB"))

        def _predict():
            predictor.set_image(cv_img)
            h, w = cv_img.shape[:2]
            point_coords = np.array([[w // 2, h // 2]])
            point_labels = np.array([1])
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False
            )
            return masks[0]

        start = time.perf_counter()
        mask = await asyncio.to_thread(_predict)

        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
        image_with_sam_mask = image.copy().convert("RGBA")
        image_with_sam_mask.putalpha(mask_img)

        output_img, model_name, birefnet_time = await self.run_birefnet(image_with_sam_mask, "birefnet-general")
        total_elapsed = time.perf_counter() - start
        return output_img, "sam2+birefnet-general", total_elapsed

    async def process(self, image: Image.Image, route: str) -> tuple[Image.Image, str, float]:
        if route == "ROUTE_GRAPHIC":
            output_img, model_name, elapsed = await self.run_birefnet(
                image,
                "birefnet-general",
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=30,
                alpha_matting_erode_size=2
            )
            # Post-process: recover small decorative components
            output_img = await asyncio.to_thread(
                SmallComponentRecovery.recover, image, output_img
            )
            return output_img, model_name, elapsed
        elif route == "ROUTE_PORTRAIT":
            return await self.run_birefnet(image, "birefnet-portrait")
        elif route == "ROUTE_SIMPLE":
            return await self.run_birefnet(image, "birefnet-general")
        elif route == "ROUTE_COMPLEX":
            return await self.run_sam2_complex(image)
        else:
            return await self.run_birefnet(image, "birefnet-general")

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