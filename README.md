# EraseBG Backend — AI Background Removal System

> **EraseBG** is a full-stack AI background removal application.
> This repository contains the **backend system**, split into two services:
> - `render-app/` — API Gateway (deployed on Render)
> - `hf-space/` — AI Inference Engine (deployed on Hugging Face Spaces)

---

## 📐 Architecture Overview

```
User (Browser)
    │
    │  Upload image
    ▼
┌─────────────────────────────┐
│   Frontend (Next.js)        │  https://erasebg-frontend.vercel.app
│   erasebg-frontend (Vercel) │
└────────────┬────────────────┘
             │  POST /remove-bg
             ▼
┌─────────────────────────────┐
│   render-app/               │  https://erasebg-backend.onrender.com
│   API Gateway (Render)      │  • Validates file
│                             │  • Forwards to HF Space
└────────────┬────────────────┘
             │  POST /infer  (X-API-Key header)
             ▼
┌─────────────────────────────┐
│   hf-space/                 │  https://azaribrahim-erasebg-inference.hf.space
│   AI Inference (HF Spaces)  │  • Analyzes image
│                             │  • Routes to best model
│                             │  • Returns PNG (no background)
└─────────────────────────────┘
```

**Why two services?**
Render Free Tier doesn't have enough RAM or storage for heavy AI models. So the AI inference runs separately on Hugging Face Spaces, which provides a free Docker environment with more resources.

---

## 📁 Project Structure

```
erasebg-backend/
│
├── render-app/                      # API Gateway — deployed on Render
│   ├── main.py                      # FastAPI app entry point
│   ├── requirements.txt             # Python dependencies
│   ├── .env                         # Environment variables (local only)
│   ├── Dockerfile                   # Container config for Render
│   ├── api/
│   │   ├── health.py                # GET / and GET /health
│   │   └── remove_bg.py             # POST /remove-bg  ← main endpoint
│   ├── auth/
│   │   └── api_key.py               # API key validation
│   ├── clients/
│   │   └── huggingface_client.py    # HTTP client that calls HF Space /infer
│   ├── config/
│   │   └── settings.py              # All settings via pydantic-settings
│   ├── core/                        # Core utilities
│   ├── schemas/                     # Pydantic request/response models
│   ├── services/
│   │   └── background_removal_service.py  # Business logic
│   └── utils/
│       └── file_helper.py           # File validation (type, size)
│
├── hf-space/                        # AI Inference Engine — deployed on HF Spaces
│   ├── app.py                       # FastAPI app entry point
│   ├── requirements.txt             # Python dependencies (includes rembg)
│   ├── Dockerfile                   # Downloads models at build time
│   ├── api/
│   │   ├── health.py                # GET /health + GET /
│   │   └── inference.py             # POST /infer  ← core AI endpoint
│   ├── auth/
│   │   └── api_key.py               # API key verification (X-API-Key)
│   ├── config/
│   │   └── settings.py              # All settings via pydantic-settings
│   ├── models/
│   │   ├── loader.py                # Singleton model loader (lazy, one model at a time)
│   │   └── registry.py             # Maps model names to ONNX file paths
│   ├── services/
│   │   ├── image_analyzer.py        # Detects face / skin / text / complexity
│   │   ├── router.py                # Smart routing logic → picks best model
│   │   ├── executor.py              # Runs ONNX inference + mask upscaling
│   │   ├── inference_service.py     # Orchestrates the full pipeline
│   │   ├── resolution_manager.py    # Resize before inference, upscale after
│   │   └── image_re_analyzer.py     # Small component recovery (for graphics)
│   └── model_files/                 # ONNX models (downloaded during Docker build)
│       ├── birefnet-general.onnx
│       ├── birefnet-portrait.onnx
│       ├── bria-rmbg.onnx
│       ├── isnet-general-use.onnx
│       ├── u2net.onnx
│       └── u2net_human_seg.onnx
│
├── .github/
│   └── workflows/
│       └── sync-hf-space.yml        # Auto-syncs hf-space/ to HF Spaces on git push
│
├── render.yaml                      # Render deployment config
└── README.md                        # This file
```

---

## 🛠️ Step-by-Step: How This Project Was Built

### Step 1 — Project Idea & Architecture Planning

The goal was to build a free, production-ready AI background removal service.

**The problem with a single server approach:**
- Render Free Tier has limited RAM (~512MB) — not enough to load AI models.
- Hugging Face Spaces provides free Docker with up to 16GB RAM — ideal for AI models.

**Solution:** Split into two microservices:
1. **render-app** (API Gateway) — lightweight, handles the user request, validates the file, forwards it to HF Space.
2. **hf-space** (Inference Engine) — heavy, runs actual AI models, returns the processed image.

---

### Step 2 — Setting Up the API Gateway (`render-app`)

**Tech chosen:** FastAPI + Uvicorn (lightweight, async-native Python web framework)

Created the following structure:

#### `main.py` — FastAPI App Setup
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Erasebg Backend API Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### `config/settings.py` — Environment Settings via Pydantic
All configuration is read from `.env` using `pydantic-settings`:
```python
class Settings(BaseSettings):
    frontend_url: str
    hf_space_url: str
    hf_api_key: str
    max_file_size: int = 12582912   # 12MB
    allowed_extensions: str = "png,jpg,jpeg,webp"
    log_level: str = "INFO"
```

#### `utils/file_helper.py` — File Validation
Before forwarding to HF Space, the file is validated:
- Check file extension is in `allowed_extensions`.
- Check file size is within `max_file_size`.
- Return a clean HTTP error if validation fails.

#### `api/remove_bg.py` — Main Endpoint
```
POST /remove-bg
  → Validates file (type, size)
  → Calls HuggingFaceClient.infer(file)
  → Returns PNG response
```

#### `clients/huggingface_client.py` — HTTP Client
Uses `httpx` (async) to forward the image to HF Space:
```python
async with httpx.AsyncClient(timeout=60.0) as client:
    response = await client.post(
        f"{settings.hf_space_url}/infer",
        files={"file": (filename, file_bytes, content_type)},
        headers={"X-API-Key": settings.hf_api_key},
    )
```
- Retries up to `HF_MAX_RETRIES` times on failure.
- Sends `X-API-Key` header to authenticate with HF Space.

#### `api/health.py` — Health Check
```
GET /health → returns {"status": "ok"}
```
Used by UptimeRobot to ping the server every 5 minutes so Render's free tier doesn't go to sleep.

---

### Step 3 — Setting Up the AI Inference Engine (`hf-space`)

**Tech chosen:** FastAPI + rembg + ONNX Runtime + Pillow + OpenCV + NumPy

This service has the actual AI models and runs the background removal.

#### `app.py` — HF Space FastAPI Entry
```python
from fastapi import FastAPI
from api.inference import router as inference_router
from api.health import router as health_router

app = FastAPI(title="Erasebg Inference Engine")
app.include_router(inference_router)
app.include_router(health_router)
```

#### `auth/api_key.py` — Secure the Inference Endpoint
Only `render-app` is allowed to call `/infer`. This is enforced via an API key:
```python
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != settings.hf_api_key:
        raise HTTPException(status_code=403, detail="Invalid API Key")
```
Both sides share the same `HF_API_KEY` value via their respective `.env` files.

#### `config/settings.py` — HF Space Settings
```python
class Settings(BaseSettings):
    hf_api_key: str
    preload_models_on_startup: bool = False
    warmup_on_startup: bool = False
```

---

### Step 4 — Building the Smart AI Pipeline

The real intelligence of this system is how it automatically picks the best AI model for each image. Here is the complete pipeline:

```
Upload → Analyze → Route → Resize → Infer → Upscale → Return PNG
```

#### `services/image_analyzer.py` — Image Analysis
Before choosing a model, the image is analyzed:
- **Face detection** — Uses OpenCV's Haar Cascade classifier.
- **Skin ratio** — Detects skin-tone pixels to confirm portrait.
- **Text detection** — Counts edges to identify graphic/text images.
- **Edge density** — Measures complexity of the scene.
- **Subject size** — Estimates how large the main subject is.

Returns an `ImageMetrics` object with all these values.

#### `services/router.py` — Smart Model Routing
Based on `ImageMetrics`, the router picks one of these routes:
| Route | Condition | Model Used |
|-------|-----------|-----------|
| `ROUTE_PORTRAIT` | Face detected | `birefnet-portrait` |
| `ROUTE_GRAPHIC` | Text-heavy, no face | `birefnet-general` |
| `ROUTE_SIMPLE` | Simple background | `birefnet-general` |
| `ROUTE_COMPLEX` | Complex scene | `birefnet-general` (with extra processing) |

#### `models/loader.py` — Singleton Model Loader (Memory-Safe)
**Critical for free-tier hosting:**
```python
# Only ONE model loaded in memory at a time
if self.current_model_name != model_name:
    del self.current_session  # free old model
    gc.collect()              # force garbage collection
    self.current_session = ort.InferenceSession(model_path)
    self.current_model_name = model_name
```
This prevents Out-of-Memory (OOM) crashes on HF Spaces free tier (16GB RAM limit).

#### `models/registry.py` — Model Registry
Maps friendly model names to ONNX file paths:
```python
MODEL_REGISTRY = {
    "birefnet-general": "/app/model_files/birefnet-general.onnx",
    "birefnet-portrait": "/app/model_files/birefnet-portrait.onnx",
    "u2net": "/app/model_files/u2net.onnx",
    ...
}
```

#### `services/resolution_manager.py` — Smart Resize & Upscale
Large images would crash the inference. This service:
1. **Before inference:** Downsizes the image to a safe resolution (e.g., 1024px max).
2. **After inference:** Upscales the transparency mask back to the original resolution.
3. Applies a **bilateral filter** to preserve fine details (hair, beard edges).

#### `services/executor.py` — ONNX Model Inference
Runs the actual AI:
1. Preprocesses the image (normalize, convert to tensor).
2. Passes it through the ONNX model using `onnxruntime`.
3. Extracts the alpha mask.
4. Applies the mask to create a transparent PNG.

#### `services/inference_service.py` — Full Pipeline Orchestrator
```python
async def remove_background(image: Image) -> bytes:
    metrics = image_analyzer.analyze(image)
    route = smart_router.route(metrics)
    resized = resolution_manager.resize_for_inference(image, route)
    result = executor.process(resized, route)
    upscaled = resolution_manager.upscale_mask_to_original(result, image)
    return encode_as_png(upscaled)
```

#### `services/image_re_analyzer.py` — Small Component Recovery
For graphic images (logos, icons), the pipeline also checks for small disconnected transparent areas that were incorrectly removed and recovers them.

---

### Step 5 — Containerization with Docker

#### `hf-space/Dockerfile` — Baking Models at Build Time
Models are downloaded **during Docker build**, not at runtime:
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Download ONNX models at build time (not at startup)
RUN mkdir -p model_files && \
    wget -q -O model_files/birefnet-general.onnx \
      https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-epoch_244.onnx && \
    wget -q -O model_files/birefnet-portrait.onnx \
      https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-portrait-epoch_150.onnx

COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

This ensures the Space starts instantly (no download delay).

#### `render-app/Dockerfile` — Lightweight Gateway Container
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### Step 6 — CI/CD with GitHub Actions

#### `.github/workflows/sync-hf-space.yml` — Auto-sync to HF Spaces
When any change is pushed to `hf-space/` folder on the `main` branch, this action automatically uploads all files to the Hugging Face Space using the `HF_TOKEN` secret:
```yaml
on:
  push:
    branches: [main]
    paths: ["hf-space/**"]

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Push to HF Space
        uses: huggingface/huggingface-hub-push-action@v1
        with:
          token: ${{ secrets.HF_TOKEN }}
          src: hf-space/
          repo_id: azaribrahim/erasebg-inference
          repo_type: space
```

#### Render Auto-Deploy
Render auto-deploys `render-app/` whenever `main` is pushed. Configured via `render.yaml`.

---

### Step 7 — Monitoring with UptimeRobot

Both services are on **free hosting** (Render + HF Spaces) — they sleep after inactivity.

To keep them awake, UptimeRobot pings the `/health` endpoints every **5 minutes**:

| Service | Health URL |
|---------|-----------|
| API Gateway | `https://erasebg-backend.onrender.com/health` |
| HF Inference | `https://azaribrahim-erasebg-inference.hf.space/health` |

Both endpoints return `{"status": "ok"}`.

---

## 🚀 Local Setup & Running (Step by Step)

### Prerequisites
- Python 3.9+
- pip
- Git

---

### 1. Clone the Repository

```bash
git clone https://github.com/azarrazagula/erasebg-backend.git
cd erasebg-backend
```

---

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
```

---

### 3. Run the API Gateway (render-app)

```bash
cd render-app
pip install -r requirements.txt
```

Create `.env` file inside `render-app/`:

```env
FRONTEND_URL=http://localhost:3000
HF_SPACE_URL=http://localhost:8001
HF_API_KEY=your_secret_api_key_here
LOG_LEVEL=INFO
MAX_FILE_SIZE=12582912
ALLOWED_EXTENSIONS=png,jpg,jpeg,webp
```

Start the API Gateway:

```bash
uvicorn main:app --reload
# Runs at http://localhost:8000
```

---

### 4. Run the Inference Engine (hf-space)

> ⚠️ Models (~2GB total) must be downloaded first.

```bash
cd ../hf-space
pip install -r requirements.txt
```

Download models manually (one-time):

```bash
mkdir -p model_files

wget -O model_files/birefnet-general.onnx \
  https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-epoch_244.onnx

wget -O model_files/birefnet-portrait.onnx \
  https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-portrait-epoch_150.onnx
```

Create `.env` file inside `hf-space/`:

```env
HF_API_KEY=your_secret_api_key_here
```

> ⚠️ `HF_API_KEY` must be the **exact same value** as in `render-app/.env`.

Start the inference engine:

```bash
uvicorn app:app --port 8001 --reload
# Runs at http://localhost:8001
```

---

### 5. Test the Full Flow

Open a new terminal and run:

```bash
curl -X POST http://localhost:8000/remove-bg \
  -F "file=@your-image.jpg" \
  --output result.png
```

Or open the frontend at `http://localhost:3000` and upload an image.

---

## 🌐 API Endpoints

### render-app (API Gateway)

| Method | Endpoint | Status | Description |
|--------|----------|--------|-------------|
| `GET` | `/` | ⚠️ Unused | Root info |
| `GET` | `/health` | ✅ Active | UptimeRobot ping to keep Render awake |
| `POST` | `/remove-bg` | ✅ Active | Main endpoint — called by the Next.js frontend |

### hf-space (Inference Engine)

| Method | Endpoint | Status | Description |
|--------|----------|--------|-------------|
| `GET` | `/` | ⚠️ Unused | Root info |
| `GET` | `/health` | ✅ Active | UptimeRobot ping to keep HF Space awake |
| `POST` | `/infer` | ✅ Active | Core AI endpoint — called by render-app only |

---

## ⚙️ Environment Variables

### render-app/.env

| Variable | Default | Description |
|----------|---------|-------------|
| `FRONTEND_URL` | `http://localhost:3000` | CORS allowed origin |
| `HF_SPACE_URL` | `http://localhost:8001` | URL of HF Space inference engine |
| `HF_API_KEY` | `""` | Secret key sent to HF Space |
| `HF_TIMEOUT` | `60` | Request timeout (seconds) |
| `HF_MAX_RETRIES` | `3` | Retry attempts on failure |
| `MAX_FILE_SIZE` | `12582912` | Max upload size (12MB) |
| `ALLOWED_EXTENSIONS` | `png,jpg,jpeg,webp` | Accepted file types |
| `LOG_LEVEL` | `INFO` | Logging level |

### hf-space/.env

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_API_KEY` | `""` | Must match render-app's `HF_API_KEY` |
| `PRELOAD_MODELS_ON_STARTUP` | `false` | Load models at boot (disabled to save RAM) |
| `WARMUP_ON_STARTUP` | `false` | Run dummy inference at boot (disabled) |

---

## 🚢 Production Deployment

### render-app → Render

Deployed via `render.yaml`:
```yaml
services:
  - type: web
    name: erasebg-backend
    runtime: python
    rootDir: render-app
    buildCommand: "pip install -r requirements.txt"
    startCommand: "uvicorn main:app --host 0.0.0.0 --port $PORT"
    plan: free
```

Set these in Render Dashboard → Environment:
- `HF_SPACE_URL` = `https://azaribrahim-erasebg-inference.hf.space`
- `HF_API_KEY` = your shared secret key
- `FRONTEND_URL` = your Vercel frontend URL

### hf-space → Hugging Face Spaces

Push changes to `hf-space/` on `main` branch → GitHub Action syncs automatically.

Add `HF_TOKEN` as a GitHub repository secret for the sync workflow.

---

## 🔧 Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `503 Service Unavailable` from HF Space | Space is sleeping | Check HF Space logs at huggingface.co/spaces/azaribrahim/erasebg-inference |
| `Memory limit exceeded (16Gi)` | Two models loaded at same time | Fixed: `gc.collect()` is called before loading a new model |
| `500 Internal Server Error` | HF Space returned error | Check HF Space logs, verify `HF_API_KEY` matches on both sides |
| `ImportError: cannot import settings` | Wrong import path in hf-space | Use `from config.settings import settings` |
| Model not found at startup | Model files missing in Docker image | Check `hf-space/Dockerfile` wget commands and HF Space build logs |

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI |
| ASGI Server | Uvicorn |
| AI Models | BiRefNet, U2Net (ONNX) via rembg |
| Image Processing | Pillow, OpenCV, NumPy |
| HTTP Client | httpx (async) |
| Config Management | pydantic-settings |
| API Gateway Hosting | Render (Free) |
| Inference Hosting | Hugging Face Spaces (Free) |
| CI/CD | GitHub Actions |
| Monitoring | UptimeRobot (Free) |

---

## 📄 License

See [LICENSE](./LICENSE) file for details.
