import numpy as np
import cv2
from PIL import Image
from config import settings

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
