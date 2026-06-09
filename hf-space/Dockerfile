FROM python:3.9-slim

# Install system dependencies required for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create user to run the app (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements and install
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=user:user . .

# Download models during build time so they are baked into the Docker image
# This prevents the 1GB Hugging Face repo limit issue and avoids runtime downloads!
RUN mkdir -p model_files && \
    wget -qO model_files/birefnet-general.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/birefnet-general.onnx && \
    wget -qO model_files/birefnet-portrait.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/birefnet-portrait.onnx && \
    wget -qO model_files/bria-rmbg.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/bria-rmbg.onnx && \
    wget -qO model_files/isnet-general-use.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx && \
    wget -qO model_files/u2net.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx && \
    wget -qO model_files/u2net_human_seg.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx

# Expose port 7860 which is the default for Hugging Face Spaces
EXPOSE 7860

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
