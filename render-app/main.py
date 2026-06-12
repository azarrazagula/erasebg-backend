import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings                        # .env-ல இருந்து settings load பண்ணு
from api.remove_bg import router as bg_router               # POST /remove-bg route
from api.server_info import router as server_info_router    # GET / route (server info)
from api.wake_check import router as wake_check_router      # GET /health route (uptime ping)

# App-level logging setup — .env-ல LOG_LEVEL-ஐ பொறுத்து INFO/DEBUG/ERROR set ஆகும்
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI app instance உருவாக்கு — /docs-ல Swagger UI கிடைக்கும்
app = FastAPI(
    title="Erasebg Backend API Gateway",
    description="API Gateway that forwards background removal requests to the AI Inference Space.",
    version="1.0.0"
)

# CORS Middleware — browser-ல frontend மட்டும் API call பண்ண அனுமதி
# allow_origins-ல frontend URL மட்டும் add பண்ணியிருக்கோம் (security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_url,          # .env-ல இருந்து வரும் (production frontend URL)
        "http://localhost:3000",         # Next.js dev server
        "http://127.0.0.1:3000",        # localhost alternative
        "http://localhost:5173",         # Vite dev server
        "http://127.0.0.1:5173"         # Vite localhost alternative
    ],
    allow_credentials=True,             # Cookie/auth header அனுப்ப அனுமதி
    allow_methods=["*"],                # GET, POST, PUT, DELETE எல்லாம் allow
    allow_headers=["*"],                # எல்லா headers-உம் allow
    max_age=600                         # CORS preflight response 10 min cache
)

# Routers register பண்ணு — ஒவ்வொரு router-உம் ஒரு feature group
app.include_router(server_info_router)  # GET / → server alive-ஆ இருக்கான்னு check
app.include_router(wake_check_router)   # GET /health → UptimeRobot ping
app.include_router(bg_router)           # POST /remove-bg → main feature

# Direct python main.py மூலம் run பண்ணும்போது மட்டும் இந்த block execute ஆகும்
# uvicorn main:app மூலம் run பண்ணும்போது இது bypass ஆகும்
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",             # எல்லா network interfaces-லயும் listen பண்ணு
        port=8000,
        reload=False,               # Direct run-ல hot reload வேண்டாம்
        log_level=settings.log_level.lower()
    )
