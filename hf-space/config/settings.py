import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # API Authentication
    hf_api_key: str = ""  # render-app → hf-space authenticate பண்ண secret key; empty = open access

    # Smart Routing Thresholds — ImageAnalyzer metrics → SmartRouter decision
    portrait_skin_threshold: float = 0.06      # Stage 6: இந்த % skin இருந்தா portrait route
    graphic_asset_threshold: float = 65.0      # Stage 4: இந்த score இருந்தா graphic route
    graphic_color_threshold: float = 15.0      # (unused now — legacy)
    graphic_edge_threshold: float = 0.05       # (unused now — legacy)

    # Redesigned Routing Parameters
    robust_skin_sat_cap: float = 170.0          # HSV skin detection — saturation cap (neon colors exclude)
    text_count_threshold: int = 3               # Stage 3: word blocks >= 3 = text-heavy graphic
    product_contour_ratio_min: float = 0.12    # Stage 4: dominant object minimum area ratio
    architecture_edge_density_min: float = 0.06  # Stage 5: building/temple edge density minimum
    architecture_dispersion_max: float = 0.11    # Stage 5: edge uniform distribution maximum
    robust_skin_threshold: float = 0.04        # Skin presence very low = not human

    # SAM2 — disabled by default (slow, requires extra model download)
    sam2_disabled: bool = True                  # True = SAM2 skip பண்ணு, ROUTE_COMPLEX never trigger
    sam2_complexity_threshold: float = 999.0    # Very high = effectively disabled
    sam2_coverage_threshold: float = 0.99       # Very high = effectively disabled

    # Resolution Tiers — image size-ஐ பொறுத்து inference resolution decide
    res_tier1_threshold: int = 2000             # <= 2000px → original size maintain
    res_tier2_target: int = 2048               # 2000-4000px → 2048px-க்கு scale down
    res_tier3_target: int = 2560               # > 4000px → 2560px-க்கு scale down

    # Per-route maximum resolution caps
    portrait_max_resolution: int = 1024        # Portrait → max 1024px (BiRefNet trained size)
    graphic_max_resolution: int = 1280         # Graphic → max 1280px
    simple_max_resolution: int = 1024          # Simple/General → max 1024px

    # Startup behavior
    preload_models_on_startup: bool = False    # False = lazy load (first request-ல load)
    warmup_on_startup: bool = False            # False = warmup skip (first request slow-ஆ இருக்கும்)

    # Face Detection — OpenCV Haar cascade
    face_detection_enabled: bool = True        # True = face detect enable
    face_detection_min_size: int = 30          # 30x30 pixels minimum face size
    face_skin_floor: float = 0.03             # Face detect + skin < 3% = false positive (ignore)

    # Mask Guided Upscale — bilateral filter for hair/beard edges
    mask_guided_upscale: bool = True           # True = bilateral filter apply
    mask_guided_min_scale_ratio: float = 2.0   # Scale ratio >= 2x இருந்தா மட்டும் filter apply
    mask_upscale_blur_sigma: float = 0.5      # Gaussian blur sigma (block artifacts remove)

    # Small Component Recovery — ROUTE_GRAPHIC-ல missed dots/lines recover
    recovery_enabled: bool = True              # True = recovery enable
    recovery_min_area: int = 30               # 30 pixels minimum component area
    recovery_max_distance: float = 0.25       # Main fg-ல இருந்து max 25% diagonal distance
    recovery_color_similarity: float = 50.0   # Max L2 color distance from fg mean
    recovery_edge_score_threshold: float = 0.05  # Minimum edge pixels ratio
    recovery_compactness_threshold: float = 0.03  # Minimum shape compactness (noise filter)

    log_level: str = "INFO"   # DEBUG/INFO/WARNING/ERROR
    
    model_config = SettingsConfigDict(
        env_file=".env",            # hf-space/.env படி
        env_file_encoding="utf-8",
        extra="ignore"              # Unknown keys ignore
    )

settings = Settings()

# rembg-க்கு local model_files/ directory சொல்லு — internet download prevent
model_files_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_files"))
os.environ["U2NET_HOME"] = model_files_dir      # rembg இந்த env var-ஐ check பண்ணும்
os.makedirs(model_files_dir, exist_ok=True)     # Directory இல்லன்னா create பண்ணு
