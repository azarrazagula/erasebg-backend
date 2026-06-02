from config import settings


class SmartRouter:
    """
    Route images to the optimal BiRefNet model based on analysis metrics.

    Priority order:
      1. face_detected            → ROUTE_PORTRAIT  (strongest, fastest signal)
      2. skin_ratio > 0.35        → ROUTE_PORTRAIT  (definitive human)
      3. graphic_confidence > thr  → ROUTE_GRAPHIC
      4. skin_ratio > 0.06        → ROUTE_PORTRAIT  (moderate human)
      5. ROUTE_COMPLEX             → near-disabled   (SAM2 too slow on MPS)
      6. default                   → ROUTE_SIMPLE

    Changes from original:
      - Face detection is the top-priority signal (no false-positive risk
        because ImageAnalyzer already guards with face_skin_floor).
      - SAM2 / ROUTE_COMPLEX is disabled by default (settings.sam2_disabled).
      - portrait_skin_threshold lowered from 0.08 → 0.06.
      - graphic_asset_threshold lowered from 70 → 65.
    """

    @staticmethod
    def route(metrics: dict) -> str:
        face_detected = metrics.get("face_detected", False)
        skin_ratio = metrics.get("skin_ratio", 0.0)
        graphic_score = metrics.get("graphic_score", 0.0)
        coverage = metrics.get("coverage", 0.0)
        complexity = metrics.get("complexity", 0.0)

        # ── Rule 1: Face detected → always portrait ────────────────────────
        if face_detected:
            print(f"[Router] Face detected → ROUTE_PORTRAIT")
            return "ROUTE_PORTRAIT"

        # ── Rule 2: High skin ratio → definitive human ─────────────────────
        if skin_ratio > 0.35:
            print(f"[Router] Override: High skin_ratio ({skin_ratio:.4f}) → ROUTE_PORTRAIT")
            return "ROUTE_PORTRAIT"

        # ── Rule 3: Weighted graphic confidence (penalised by skin) ────────
        graphic_confidence = graphic_score - (skin_ratio * 100.0)

        if graphic_confidence > settings.graphic_asset_threshold:
            print(f"[Router] Graphic confidence ({graphic_confidence:.1f}) → ROUTE_GRAPHIC")
            return "ROUTE_GRAPHIC"

        # ── Rule 4: Moderate skin → portrait ───────────────────────────────
        if skin_ratio > settings.portrait_skin_threshold:
            print(f"[Router] Moderate skin_ratio ({skin_ratio:.4f}) → ROUTE_PORTRAIT")
            return "ROUTE_PORTRAIT"

        # ── Rule 5: ROUTE_COMPLEX (near-disabled; SAM2 too slow on MPS) ───
        if (
            not settings.sam2_disabled
            and complexity > settings.sam2_complexity_threshold
            and coverage > settings.sam2_coverage_threshold
        ):
            print(f"[Router] High complexity/coverage → ROUTE_COMPLEX")
            return "ROUTE_COMPLEX"

        # ── Default ────────────────────────────────────────────────────────
        print(f"[Router] Default → ROUTE_SIMPLE")
        return "ROUTE_SIMPLE"
