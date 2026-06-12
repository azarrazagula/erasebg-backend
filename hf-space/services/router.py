from config.settings import settings


class SmartRouter:
    """
    Route images to the optimal BiRefNet model based on robust analysis metrics.

    Decision Stages:
      1. Human Frontal Face Detection (ROUTE_PORTRAIT)
      2. Definitive High Skin Detection (ROUTE_PORTRAIT)
      3. Text-Heavy/Banner/Poster Detection (ROUTE_GRAPHIC)
      4. Product/Standalone Object Detection (ROUTE_GRAPHIC)
      5. Temple/Building/Architectural Scene Detection (ROUTE_SIMPLE)
      6. Moderate Skin / Portrait Fallback (ROUTE_PORTRAIT)
      7. Complex/Detail Scene (ROUTE_COMPLEX - if enabled)
      8. Default Route (ROUTE_SIMPLE)
    """

    @staticmethod
    def route(metrics: dict) -> str:
        # ImageAnalyzer-ல இருந்து வந்த metrics எடு
        face_detected = metrics.get("face_detected", False)
        skin_ratio = metrics.get("skin_ratio", 0.0)                 # Raw skin ratio
        skin_ratio_robust = metrics.get("skin_ratio_robust", skin_ratio)  # HSV + YCrCb combined
        
        graphic_score = metrics.get("graphic_score", 0.0)           # 0-100 graphic confidence
        coverage = metrics.get("coverage", 0.0)                     # Edge bounding box / total area
        complexity = metrics.get("complexity", 0.0)                 # Edge density heuristic
        edge_density = metrics.get("edge_density", 0.0)             # Canny edge ratio
        
        # New features
        text_count = metrics.get("text_count", 0)                       # Word-like blocks count
        edge_dispersion_std = metrics.get("edge_dispersion_std", 0.0)   # Edge distribution spread
        largest_contour_ratio = metrics.get("largest_contour_ratio", 0.0)  # Dominant object ratio

        # ── Stage 1: Frontal Face → ROUTE_PORTRAIT ─────────────────────────
        # Haar cascade face detect + skin floor guard (false positive இல்லாம்)
        if face_detected:
            print(f"[Router] Face detected → ROUTE_PORTRAIT")
            return "ROUTE_PORTRAIT"

        # ── Stage 2: Definitive Skin & Low Graphic → ROUTE_PORTRAIT ────────
        # 25%+ skin pixels + low graphic score = மனிதன் (headless/tilted portrait)
        if skin_ratio_robust > 0.25 and graphic_score < 60.0:
            print(f"[Router] Definitive skin ratio ({skin_ratio_robust:.4f}) and low graphic score → ROUTE_PORTRAIT")
            return "ROUTE_PORTRAIT"

        # ── Stage 3: Text-Heavy/Poster/Banner → ROUTE_GRAPHIC ──────────────
        # text_count >= 3 = banner/poster/logo — graphic model better
        if text_count >= settings.text_count_threshold:
            print(f"[Router] Text-heavy asset detected (text_count={text_count}) → ROUTE_GRAPHIC")
            return "ROUTE_GRAPHIC"

        # ── Stage 4: Standalone Product / Logo → ROUTE_GRAPHIC ─────────────
        # Single dominant object + centered edges + no skin + high graphic = product photo
        if (
            largest_contour_ratio > settings.product_contour_ratio_min
            and edge_dispersion_std > 0.12
            and skin_ratio_robust < settings.robust_skin_threshold
            and (graphic_score > settings.graphic_asset_threshold or (graphic_score > 40.0 and text_count >= 1))
        ):
            print(
                f"[Router] Standalone product detected: "
                f"largest_contour_ratio={largest_contour_ratio:.4f}, "
                f"edge_dispersion_std={edge_dispersion_std:.4f}, "
                f"graphic_score={graphic_score:.1f} → ROUTE_GRAPHIC"
            )
            return "ROUTE_GRAPHIC"

        # Fallback graphic routing — skin penalty போட்டு pure graphic confidence check
        graphic_confidence = graphic_score - (skin_ratio_robust * 100.0)
        if graphic_confidence > settings.graphic_asset_threshold:
            print(f"[Router] Graphic confidence ({graphic_confidence:.1f}) → ROUTE_GRAPHIC")
            return "ROUTE_GRAPHIC"

        # ── Stage 5: Temple / Building / Architecture → ROUTE_SIMPLE ───────
        # High edge density + uniform distribution = structural scene (buildings, temples)
        is_architecture = False
        if edge_density >= settings.architecture_edge_density_min and edge_dispersion_std <= settings.architecture_dispersion_max:
            if skin_ratio_robust < settings.robust_skin_threshold:
                is_architecture = True
            elif edge_density > 0.12 and edge_dispersion_std < 0.10 and skin_ratio_robust < 0.25:
                # Extreme structural confidence — skin ratio override பண்ணு
                print(f"[Router] High structural complexity overrides skin_ratio_robust ({skin_ratio_robust:.4f})")
                is_architecture = True

        if is_architecture:
            print(
                f"[Router] Architectural scene / Temple detected: "
                f"edge_density={edge_density:.4f}, "
                f"edge_dispersion_std={edge_dispersion_std:.4f} → ROUTE_SIMPLE"
            )
            return "ROUTE_SIMPLE"

        # ── Stage 6: Moderate Skin / Portrait Fallback → ROUTE_PORTRAIT ────
        # profile view, tilted face, partial body — face detect miss ஆனாலும் catch
        if skin_ratio_robust > settings.portrait_skin_threshold:
            print(f"[Router] Moderate skin_ratio_robust ({skin_ratio_robust:.4f}) → ROUTE_PORTRAIT")
            return "ROUTE_PORTRAIT"

        # ── Stage 7: ROUTE_COMPLEX (SAM2 if enabled) ───────────────────────
        # SAM2 disabled by default (sam2_disabled=True in settings)
        # Enable பண்ணினா complex scenes-க்கு SAM2 + BiRefNet combo use ஆகும்
        if (
            not settings.sam2_disabled
            and complexity > settings.sam2_complexity_threshold
            and coverage > settings.sam2_coverage_threshold
        ):
            print(f"[Router] High complexity/coverage → ROUTE_COMPLEX")
            return "ROUTE_COMPLEX"

        # ── Stage 8: Default Route → ROUTE_SIMPLE ──────────────────────────
        # Pets, animals, general objects — birefnet-general use பண்ணு
        print(f"[Router] Default → ROUTE_SIMPLE")
        return "ROUTE_SIMPLE"
