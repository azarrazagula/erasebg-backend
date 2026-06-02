import numpy as np
import cv2
from PIL import Image

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
