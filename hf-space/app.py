import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from api.server_info import router as server_info_router    # GET / — service info
from api.wake_check import router as wake_check_router      # GET /health — alive check
from api.inference import router as inference_router         # POST /infer — AI inference
from models.loader import ModelLoader                        # ONNX model loader

# App-level logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# lifespan — FastAPI startup/shutdown lifecycle events handle பண்ணு
# `yield` முன்னாடி = startup code; `yield` பின்னாடி = shutdown code
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events for loading AI models on startup."""
    logger.info("Starting up Inference Space...")
    
    loader = ModelLoader()  # Singleton — ஒரே instance எப்பவும்
    
    # settings.preload_models_on_startup = True ஆனா startup-லயே models load பண்ணு
    # False ஆனா lazy load — முதல் request வரும்போது load ஆகும்
    if settings.preload_models_on_startup:
        logger.info("Preloading models from local model_files...")
        # asyncio.to_thread() — blocking ONNX load-ஐ thread pool-ல run பண்ணு
        await asyncio.to_thread(loader.preload_models)
        
        # Warmup — dummy image run பண்ணி JIT kernels compile பண்ணு (first request faster)
        if settings.warmup_on_startup:
            logger.info("Warming up models...")
            await asyncio.to_thread(loader.run_warmup)
            
    logger.info("Inference Engine Ready.")
    yield   # App running — requests accept பண்ணு
    logger.info("Shutting down Inference Space...")  # Server stop ஆகும்போது

# FastAPI app — lifespan context manager attach பண்ணு
app = FastAPI(
    title="Erasebg AI Inference Space",
    description="Dedicated AI Engine for background removal.",
    version="1.0.0",
    lifespan=lifespan   # Startup/shutdown events register
)

# hf-space render-app மட்டும் call பண்ணும் (server-to-server)
# But CORS allow_origins=["*"] — safety net for internal testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # யாரும் call பண்ணலாம் — API key மூலம் security கொடுக்கோம்
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes register
app.include_router(server_info_router)  # GET /
app.include_router(wake_check_router)   # GET /health
app.include_router(inference_router)    # POST /infer (protected by API key)

# Direct python app.py மூலம் run பண்ணும்போது
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",     # All interfaces listen
        port=8001,          # render-app-ல HF_SPACE_URL=:8001 point பண்ணியிருக்கோம்
        reload=False,
        log_level=settings.log_level.lower()
    )
