"""
ResolutionManager
-----------------
All image resizing logic for the BiRefNet inference pipeline lives here.

Design:
  1. `resize_for_inference()`:
     - Apply the user's tiered resolution strategy (based on longest side).
     - Then apply a per-route cap (portrait ≤ 1024, graphic ≤ 1280).
     - Returns the resized image and the original (w, h) for later restoration.

  2. `upscale_mask_to_original()`:
     - Extract the alpha channel from the inference output (small RGBA image).
     - Upscale the alpha mask back to original dimensions with LANCZOS.
     - For scale ratios ≥ threshold, apply cv2.bilateralFilter to preserve
       sharp hair/beard edge transitions without over-blurring.
     - Composite the refined mask with the original full-resolution pixels.
     - Returns an RGBA image at original dimensions.

Hair/Beard Quality Notes:
  - BiRefNet-portrait is trained at 1024px. It captures hair detail well at
    that resolution. Going higher does not improve model output.
  - The bilateral filter (sigmaColor=25) only blends alpha pixels with very
    similar values. This keeps hard fg/bg boundaries at hair strands sharp
    while smoothing compression artefacts in flat regions.
  - opencv-python-headless does NOT include ximgproc, so we use standard
    bilateral (cv2.bilateralFilter) rather than joint bilateral filtering.
"""

from __future__ import annotations

import numpy as np
import cv2
from PIL import Image
from typing import Tuple

from config.settings import settings


class ResolutionManager:
    """Static helpers for resolution-aware inference and mask restoration."""

    # ── Public API ─────────────────────────────────────────────────────────

    @staticmethod
    def resize_for_inference(
        image: Image.Image,
        route: str,
    ) -> Tuple[Image.Image, Tuple[int, int]]:
        """
        Resize `image` to the optimal inference size for `route`.

        Returns:
            (resized_image, (original_w, original_h))
            If no resize is needed, `resized_image` is the original object.
        """
        original_w, original_h = image.size
        infer_w, infer_h = ResolutionManager._compute_inference_size(
            original_w, original_h, route
        )

        if (infer_w, infer_h) == (original_w, original_h):
            return image, (original_w, original_h)

        resized = image.resize((infer_w, infer_h), Image.LANCZOS)
        print(
            f"[ResolutionManager] {original_w}×{original_h} → {infer_w}×{infer_h} "
            f"(route={route})"
        )
        return resized, (original_w, original_h)

    @staticmethod
    def upscale_mask_to_original(
        inference_output: Image.Image,
        original_image: Image.Image,
        original_size: Tuple[int, int],
    ) -> Image.Image:
        """
        Upscale the alpha mask from `inference_output` to `original_size`,
        apply edge-preserving refinement, then composite with `original_image`.

        Args:
            inference_output : RGBA image at inference (small) resolution.
            original_image   : Original full-resolution input image.
            original_size    : (original_w, original_h).

        Returns:
            RGBA image at `original_size` with refined alpha mask.
        """
        orig_w, orig_h = original_size
        infer_w, infer_h = inference_output.size

        # Already at original size — nothing to do.
        if (infer_w, infer_h) == (orig_w, orig_h):
            return inference_output

        # ── 1. Extract alpha from inference output ─────────────────────────
        alpha_small = np.array(inference_output.convert("RGBA"))[:, :, 3]  # uint8

        # ── 2. Upscale alpha with LANCZOS ──────────────────────────────────
        alpha_large = np.array(
            Image.fromarray(alpha_small, "L").resize((orig_w, orig_h), Image.LANCZOS),
            dtype=np.float32,
        )

        # ── 3. Edge-preserving bilateral filter (hair/beard preservation) ──
        scale_ratio = max(orig_w / infer_w, orig_h / infer_h)
        if (
            settings.mask_guided_upscale
            and scale_ratio >= settings.mask_guided_min_scale_ratio
        ):
            alpha_large = ResolutionManager._bilateral_refine(alpha_large)

        # ── 4. Optional Gaussian smoothing (removes block artefacts) ───────
        if settings.mask_upscale_blur_sigma > 0:
            sigma = settings.mask_upscale_blur_sigma
            ksize = max(3, int(sigma * 6) | 1)  # odd kernel, min 3
            alpha_large = cv2.GaussianBlur(alpha_large, (ksize, ksize), sigma)

        alpha_final = np.clip(alpha_large, 0, 255).astype(np.uint8)

        # ── 5. Composite: original pixels + refined alpha ──────────────────
        orig_arr = np.array(original_image.convert("RGBA"))
        orig_arr[:, :, 3] = alpha_final
        return Image.fromarray(orig_arr, "RGBA")

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _compute_inference_size(
        original_w: int,
        original_h: int,
        route: str,
    ) -> Tuple[int, int]:
        """
        Two-stage size computation:
          Stage 1 — General tiered strategy (longest-side based).
          Stage 2 — Per-route cap (takes the minimum).
        Ensures output dimensions are even numbers (MPS/ONNX compatibility).
        """
        longest = max(original_w, original_h)

        # Stage 1: General tier
        if longest <= settings.res_tier1_threshold:
            infer_w, infer_h = original_w, original_h
        elif longest <= 4000:
            scale = settings.res_tier2_target / longest
            infer_w = int(original_w * scale)
            infer_h = int(original_h * scale)
        else:
            scale = settings.res_tier3_target / longest
            infer_w = int(original_w * scale)
            infer_h = int(original_h * scale)

        # Stage 2: Per-route cap
        route_cap = {
            "ROUTE_PORTRAIT": settings.portrait_max_resolution,
            "ROUTE_GRAPHIC": settings.graphic_max_resolution,
            "ROUTE_SIMPLE": settings.simple_max_resolution,
            "ROUTE_COMPLEX": settings.simple_max_resolution,
        }.get(route, settings.simple_max_resolution)

        longest_after = max(infer_w, infer_h)
        if longest_after > route_cap:
            cap_scale = route_cap / longest_after
            infer_w = int(infer_w * cap_scale)
            infer_h = int(infer_h * cap_scale)

        # Force even dimensions; floor to at least 32×32
        infer_w = max(32, infer_w - (infer_w % 2))
        infer_h = max(32, infer_h - (infer_h % 2))

        return infer_w, infer_h

    @staticmethod
    def _bilateral_refine(alpha_float: np.ndarray) -> np.ndarray:
        """
        Apply cv2.bilateralFilter to the float32 alpha channel.

        Parameters chosen for hair/beard edge preservation:
          d=9          — 9-pixel neighbourhood diameter (local, fast)
          sigmaColor=25 — only blends alpha values within ±25 of each other,
                          keeping sharp fg/bg transitions at hair strands intact
          sigmaSpace=9  — spatial extent (matches d)

        Input/output: float32 H×W array, values in [0, 255].
        """
        src_u8 = np.clip(alpha_float, 0, 255).astype(np.uint8)
        refined = cv2.bilateralFilter(src_u8, d=9, sigmaColor=25, sigmaSpace=9)
        return refined.astype(np.float32)
