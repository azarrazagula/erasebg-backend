from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    frontend_url: str = "http://localhost:3000"
    max_file_size: int = 12582912
    allowed_extensions: str = "png,jpg,jpeg,webp"
    log_level: str = "INFO"

    # Smart Routing Thresholds
    sam2_complexity_threshold: float = 65.0
    sam2_coverage_threshold: float = 0.85
    portrait_skin_threshold: float = 0.08
    graphic_asset_threshold: float = 70.0
    graphic_color_threshold: float = 15.0
    graphic_edge_threshold: float = 0.05

    # Small Component Recovery
    recovery_enabled: bool = True
    recovery_min_area: int = 30
    recovery_max_distance: float = 0.25        # fraction of image diagonal
    recovery_color_similarity: float = 50.0
    recovery_edge_score_threshold: float = 0.05
    recovery_compactness_threshold: float = 0.03  # 4π·area/perimeter²; rejects blobs more amorphous than this

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
