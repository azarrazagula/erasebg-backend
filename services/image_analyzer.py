import numpy as np
import cv2
from PIL import Image

from config import settings


class FaceDetector:
    """
    Lightweight face detection using OpenCV's built-in Haar cascade.

    Zero new dependencies — uses haarcascade_frontalface_default.xml that ships
    with opencv-python-headless.

    Performance: ~10–30ms on a 400px thumbnail.

    Limitations:
      - Only detects frontal faces (not profile/tilted).
      - Mitigated: skin_ratio fallback covers tilted portraits.
      - Haar false positives (paintings, posters) are guarded by face_skin_floor.
    """

    _cascade = None

    @classmethod
    def detect(cls, cv_img_bgr: np.ndarray) -> bool:
        """Return True if at least one frontal face is detected."""
        if not settings.face_detection_enabled:
            return False

        if cls._cascade is None:
            cls._cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

        gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
        min_sz = settings.face_detection_min_size
        faces = cls._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_sz, min_sz),
        )
        return len(faces) > 0


class GraphicAssetDetector:
    @staticmethod
    def detect(cv_img: np.ndarray) -> float:
        """Score 0–100 indicating how likely the image is a graphic asset."""
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
        color_score = max(0.0, 100.0 - (unique_colors / 5.0))

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


class ImageAnalyzer:
    """
    Analyze an image and return routing metrics.

    FIX vs original:
      The original ran HSV skin detection, Canny edges, and brightness/contrast
      on the FULL-RESOLUTION image — wasting 1–2s on 4K images. Now ALL CV
      operations run on a 400px thumbnail (same downscale that
      GraphicAssetDetector already used internally).
    """

    # Analysis thumbnail size — shared across all CV ops.
    _ANALYSIS_MAX_DIM = 400

    @staticmethod
    def analyze(image: Image.Image) -> dict:
        rgb = image.convert("RGB")
        cv_img_full = np.array(rgb)[:, :, ::-1].copy()  # RGB → BGR

        # ── Downscale ONCE for all analysis ────────────────────────────────
        h, w = cv_img_full.shape[:2]
        max_dim = ImageAnalyzer._ANALYSIS_MAX_DIM
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            cv_img = cv2.resize(cv_img_full, (int(w * scale), int(h * scale)))
        else:
            cv_img = cv_img_full

        sh, sw = cv_img.shape[:2]
        aspect_ratio = sw / sh if sh > 0 else 1.0

        # ── Skin ratio (HSV range) ─────────────────────────────────────────
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_ratio = cv2.countNonZero(skin_mask) / (sw * sh)

        # ── Edge density (Canny) ───────────────────────────────────────────
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = cv2.countNonZero(edges) / (sw * sh)

        # ── Contrast & Brightness ──────────────────────────────────────────
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        # ── Complexity (heuristic) ─────────────────────────────────────────
        complexity = (edge_density * 1000) + (contrast / 2)

        # ── Coverage (bounding box of edges / total area) ──────────────────
        coords = cv2.findNonZero(edges)
        coverage = 0.5
        if coords is not None:
            x, y, ew, eh = cv2.boundingRect(coords)
            coverage = (ew * eh) / (sw * sh)

        # ── Graphic score ──────────────────────────────────────────────────
        # GraphicAssetDetector already downscales internally, but since we
        # already have cv_img at 400px, pass it directly to avoid double work.
        graphic_score = GraphicAssetDetector.detect(cv_img)

        # ── Face detection (Haar cascade on 400px thumbnail) ───────────────
        face_detected = FaceDetector.detect(cv_img)

        # Guard against Haar false positives: require minimum skin presence.
        if face_detected and skin_ratio < settings.face_skin_floor:
            print(
                f"[Analyze] Face detected but skin_ratio ({skin_ratio:.4f}) "
                f"below floor ({settings.face_skin_floor}) — ignoring face"
            )
            face_detected = False

        metrics = {
            "skin_ratio": skin_ratio,
            "brightness": brightness,
            "contrast": contrast,
            "aspect_ratio": aspect_ratio,
            "edge_density": edge_density,
            "complexity": complexity,
            "coverage": coverage,
            "graphic_score": graphic_score,
            "face_detected": face_detected,
        }
        print(f"[Analyze] Metrics: {metrics}")
        return metrics
