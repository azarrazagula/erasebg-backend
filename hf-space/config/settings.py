import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # API Authentication
    hf_api_key: str = ""  # If set, all requests to /infer must provide this key

    # Smart Routing Thresholds 
    portrait_skin_threshold: float = 0.06
    graphic_asset_threshold: float = 65.0
    graphic_color_threshold: float = 15.0
    graphic_edge_threshold: float = 0.05

    # Redesigned Routing Parameters
    robust_skin_sat_cap: float = 170.0
    text_count_threshold: int = 3
    product_contour_ratio_min: float = 0.12
    architecture_edge_density_min: float = 0.06
    architecture_dispersion_max: float = 0.11
    robust_skin_threshold: float = 0.04

    sam2_disabled: bool = True
    sam2_complexity_threshold: float = 999.0
    sam2_coverage_threshold: float = 0.99

    res_tier1_threshold: int = 2000
    res_tier2_target: int = 2048
    res_tier3_target: int = 2560

    portrait_max_resolution: int = 1024
    graphic_max_resolution: int = 1280
    simple_max_resolution: int = 1024

    preload_models_on_startup: bool = True
    warmup_on_startup: bool = True

    face_detection_enabled: bool = True
    face_detection_min_size: int = 30
    face_skin_floor: float = 0.03

    mask_guided_upscale: bool = True
    mask_guided_min_scale_ratio: float = 2.0
    mask_upscale_blur_sigma: float = 0.5

    recovery_enabled: bool = True
    recovery_min_area: int = 30
    recovery_max_distance: float = 0.25
    recovery_color_similarity: float = 50.0
    recovery_edge_score_threshold: float = 0.05
    recovery_compactness_threshold: float = 0.03

    log_level: str = "INFO"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Set U2NET_HOME to our local model_files directory
model_files_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_files"))
os.environ["U2NET_HOME"] = model_files_dir
os.makedirs(model_files_dir, exist_ok=True)
