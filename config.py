from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── Server ───────────────────────────────────────────────────────────────
    frontend_url: str = "http://localhost:3000"
    max_file_size: int = 12582912  # 12 MB
    allowed_extensions: str = "png,jpg,jpeg,webp"
    log_level: str = "INFO"

    # ── Smart Routing Thresholds ─────────────────────────────────────────────
    portrait_skin_threshold: float = 0.06      # was 0.08; catches more medium-skin images
    graphic_asset_threshold: float = 65.0      # was 70.0; slightly more aggressive
    graphic_color_threshold: float = 15.0
    graphic_edge_threshold: float = 0.05

    # Redesigned Routing Parameters
    robust_skin_sat_cap: float = 170.0         # Max saturation in HSV for skin to filter neon/graphics
    text_count_threshold: int = 3              # Count of word-like blocks to classify as ROUTE_GRAPHIC
    product_contour_ratio_min: float = 0.12    # Area ratio of largest contour for standalone objects/products
    architecture_edge_density_min: float = 0.06 # Minimum edge density for temples/architecture
    architecture_dispersion_max: float = 0.11   # Max standard dev of edge dispersion (spread edges across frame)
    robust_skin_threshold: float = 0.04        # Minimal robust skin ratio fallback for portraits without a face


    # ── SAM2 (disabled by default — 18-25s on MPS, not viable for production) ─
    # Set sam2_disabled=False via env to re-enable for ?quality=high requests.
    sam2_disabled: bool = True
    sam2_complexity_threshold: float = 999.0   # effectively unreachable
    sam2_coverage_threshold: float = 0.99

    # ── Resolution Capping Strategy ──────────────────────────────────────────
    # General tiered rules (applied first, based on longest side):
    #   longest ≤ res_tier1_threshold  →  process at original resolution
    #   longest ≤ 4000                 →  resize so longest = res_tier2_target
    #   longest >  4000                →  resize so longest = res_tier3_target
    res_tier1_threshold: int = 2000
    res_tier2_target: int = 2048
    res_tier3_target: int = 2560

    # Per-route caps (applied after general tiers; takes the minimum):
    portrait_max_resolution: int = 1024   # BiRefNet-portrait trained at 1024
    graphic_max_resolution: int = 1280    # logos need sharper edges; no alpha_matting
    simple_max_resolution: int = 1024     # birefnet-general also trained at 1024

    # ── Model Loading & Warmup ───────────────────────────────────────────────
    preload_models_on_startup: bool = True   # load both models before first request
    warmup_on_startup: bool = True           # run dummy 256×256 inference at startup

    # ── Face Detection (Haar Cascade — no extra dependencies) ───────────────
    face_detection_enabled: bool = True
    # Minimum face size (pixels) on the 400px analysis thumbnail:
    face_detection_min_size: int = 30
    # Guard against Haar false positives (paintings, posters):
    # face_detected is only trusted if skin_ratio is also above this floor:
    face_skin_floor: float = 0.03

    # ── Mask Upscaling ───────────────────────────────────────────────────────
    # After inference on a downscaled image, the alpha mask must be upscaled
    # back to original dimensions and applied to the original pixels.
    #
    # Upscaling pipeline:
    #   1. Image.LANCZOS resize
    #   2. cv2.bilateralFilter (edge-preserving, activated when scale ≥ threshold)
    #   3. Optional Gaussian smoothing (activated when sigma > 0)
    mask_guided_upscale: bool = True
    mask_guided_min_scale_ratio: float = 2.0  # activate bilateral when scale ≥ this
    mask_upscale_blur_sigma: float = 0.5       # post-bilateral Gaussian (0 = off)

    # ── Small Component Recovery ─────────────────────────────────────────────
    recovery_enabled: bool = True
    recovery_min_area: int = 30
    recovery_max_distance: float = 0.25        # fraction of image diagonal
    recovery_color_similarity: float = 50.0
    recovery_edge_score_threshold: float = 0.05
    recovery_compactness_threshold: float = 0.03

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
