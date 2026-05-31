# Erasebg Backend - AI Background Removal API

Production-ready FastAPI backend for removing image backgrounds using rembg AI model.

## Features

- **Fast Background Removal**: Uses rembg for high-quality AI-powered background removal
- **CORS Enabled**: Pre-configured for Next.js frontend at `http://localhost:3000`
- **File Validation**: Strict validation for file types (PNG, JPG, WEBP) and size limits (12MB max)
- **Async Processing**: Non-blocking async file handling for optimal performance
- **Type Safety**: Full Python type hints throughout the codebase
- **Environment Configuration**: All settings configurable via `.env` file
- **Production Ready**: Proper error handling, logging, and HTTP status codes
- **Clean Architecture**: Separation of concerns: routes → services → utilities

## Project Structure

```
erasebg-backend/
├── main.py                 # FastAPI application setup
├── requirements.txt        # Python dependencies with pinned versions
├── .env                   # Environment variables (configurable)
├── .gitignore            # Git ignore rules
├── routes/
│   ├── __init__.py
│   └── remove_bg.py      # API endpoints
├── services/
│   ├── __init__.py
│   └── bg_service.py     # Background removal business logic
└── utils/
    ├── __init__.py
    └── file_helper.py    # File validation and utilities
```

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Setup

1. Create and activate virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment (optional - defaults are provided):

```bash
# Edit .env if needed
cat .env
```

## Running the Application

### Development Server

```bash
python main.py
```

The API will start at `http://localhost:8000`

### Access Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Production Deployment

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### POST /remove-bg

Remove background from an image.

**Request:**

- Content-Type: `multipart/form-data`
- Body: `file` (image file - PNG, JPG, WEBP, max 12MB)

**Response:**

- Content-Type: `image/png`
- Body: PNG image with transparent background

**Status Codes:**

- `200`: Success - returns processed image
- `400`: Invalid file type or bad request
- `413`: File too large
- `500`: Processing error

**Example using curl:**

```bash
curl -X POST "http://localhost:8000/remove-bg" \
  -F "file=@image.jpg" \
  --output output.png
```

**Example using Python:**

```python
import requests

with open("image.jpg", "rb") as img:
    response = requests.post(
        "http://localhost:8000/remove-bg",
        files={"file": img}
    )
    with open("output.png", "wb") as out:
        out.write(response.content)
```

### GET /health

Health check endpoint.

**Response:**

```json
{
  "status": "ok",
  "model": "rembg"
}
```

### GET /

Root endpoint with API information.

**Response:**

```json
{
  "message": "Erasebg Backend API",
  "version": "1.0.0",
  "docs": "/docs"
}
```

## Configuration

Edit `.env` to customize settings:

```env
# Frontend CORS origin
FRONTEND_URL=http://localhost:3000

# Maximum file size in bytes (12MB default)
MAX_FILE_SIZE=12582912

# Allowed file extensions (comma-separated)
ALLOWED_EXTENSIONS=png,jpg,jpeg,webp

# Logging level (INFO, DEBUG, WARNING, ERROR)
LOG_LEVEL=INFO
```

## Error Handling

The API returns appropriate HTTP status codes and descriptive error messages:

| Status | Scenario                          |
| ------ | --------------------------------- |
| 400    | Invalid file type or missing file |
| 413    | File size exceeds 12MB limit      |
| 500    | Image processing failed           |

Error response format:

```json
{
  "detail": "Error description here"
}
```

## Dependencies

- **FastAPI**: Modern web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI
- **rembg**: AI-powered background removal
- **Pillow**: Image processing library
- **python-multipart**: File upload handling
- **python-dotenv**: Environment variable management
- **pydantic**: Data validation

## Performance Notes

- Images are processed asynchronously to prevent blocking
- rembg uses alpha matting for smooth edge detection
- PNG output with transparency for optimal quality
- All file I/O is non-blocking for high throughput

## Troubleshooting

### rembg downloads model on first run

The first execution will download the U²-Net model (~180MB). This is normal and happens once.

### CORS errors from frontend

Ensure `FRONTEND_URL` in `.env` matches your frontend URL exactly.

### Out of memory on large images

The 12MB file size limit should prevent this, but if needed, adjust `MAX_FILE_SIZE` in `.env`.

## Development

### Adding new routes

1. Create endpoint in `routes/remove_bg.py`
2. Add business logic to `services/bg_service.py`
3. Use utilities from `utils/file_helper.py`
4. Include proper type hints and docstrings

### Code Quality

- Type hints on all functions and parameters
- Comprehensive error handling
- Logging for debugging and monitoring
- Clean separation of concerns

## License

See LICENSE file for details.
