import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from api.health import router as health_router
from api.inference import router as inference_router
from models.loader import ModelLoader

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events for loading AI models on startup."""
    logger.info("Starting up Inference Space...")
    
    loader = ModelLoader()
    if settings.preload_models_on_startup:
        logger.info("Preloading models from local model_files...")
        await asyncio.to_thread(loader.preload_models)
        
        if settings.warmup_on_startup:
            logger.info("Warming up models...")
            await asyncio.to_thread(loader.run_warmup)
            
    logger.info("Inference Engine Ready.")
    yield
    logger.info("Shutting down Inference Space...")

app = FastAPI(
    title="Erasebg AI Inference Space",
    description="Dedicated AI Engine for background removal.",
    version="1.0.0",
    lifespan=lifespan
)

# HF Space typically doesn't need broad CORS if requested strictly server-to-server,
# but we allow all internally just in case.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(inference_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level=settings.log_level.lower()
    )
