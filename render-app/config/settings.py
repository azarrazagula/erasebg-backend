from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Server config
    frontend_url: str = "http://localhost:3000"
    log_level: str = "INFO"
    
    # Hugging Face Space config
    hf_space_url: str = "http://localhost:8001"
    hf_api_key: str = ""  # Will be provided via env
    hf_timeout: int = 60
    hf_max_retries: int = 3
    
    # Constraints
    max_file_size: int = 12582912  # 12 MB
    allowed_extensions: str = "png,jpg,jpeg,webp"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
