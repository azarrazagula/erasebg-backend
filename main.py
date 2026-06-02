import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routes.remove_bg import router as bg_router
from services.model_manager import ModelManager


logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle - startup and shutdown events.
    """
    logger.info("Application starting up...")
    logger.info(f"Frontend URL: {settings.frontend_url}")
    logger.info(f"Max file size: {settings.max_file_size / (1024 * 1024):.1f}MB")
    logger.info(f"Allowed extensions: {settings.allowed_extensions}")
    
    if settings.preload_models_on_startup:
        logger.info("Preloading BiRefNet models...")
        mm = ModelManager()
        # preload_all blocks, but that's fine/intended during lifespan startup
        await asyncio.to_thread(mm.preload_all)
        logger.info("Models ready. Server accepting requests.")
    
    yield
    logger.info("Application shutting down...")


app = FastAPI(
    title="Erasebg Backend",
    description="AI Background Removal API powered by BiRefNet",
    version="1.0.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_url,
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=600
)


app.include_router(bg_router)


@app.get("/")
async def root() -> dict[str, Any]:
    """Root endpoint."""
    return {
        "message": "Erasebg Backend API",
        "version": "1.0.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level=settings.log_level.lower()
    )