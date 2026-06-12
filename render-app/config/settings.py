from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Server config
    frontend_url: str = "http://localhost:3000"     # CORS allow பண்ண frontend URL (default: localhost)
    log_level: str = "INFO"                          # Logging verbosity: DEBUG/INFO/WARNING/ERROR
    
    # Hugging Face Space config
    hf_space_url: str = "http://localhost:8001"     # hf-space எங்கே run ஆகுதுன்னு point பண்ணு
    hf_api_key: str = ""                             # hf-space-க்கு authenticate பண்ண secret key
    hf_timeout: int = 60                             # hf-space response வர max 60 seconds காத்திரு
    hf_max_retries: int = 3                          # Fail ஆனா max 3 முறை retry பண்ணு
    
    # Constraints
    max_file_size: int = 12582912                    # 12 MB = 12 * 1024 * 1024 bytes
    allowed_extensions: str = "png,jpg,jpeg,webp"   # இந்த formats மட்டும் accept பண்ணு

    # pydantic-settings config — .env file படி
    model_config = SettingsConfigDict(
        env_file=".env",            # render-app/.env file படி
        env_file_encoding="utf-8",
        extra="ignore"              # .env-ல unknown keys இருந்தாலும் error வராது
    )

# Module level singleton — import பண்ணும்போது ஒரே ஒரு instance உருவாகும்
settings = Settings()
