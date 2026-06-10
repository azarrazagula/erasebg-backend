FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create user (Hugging Face requirement)
RUN useradd -m -u 1000 user

# Create app dir and set ownership BEFORE switching user
RUN mkdir /app && chown user:user /app

USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Step 1: Copy and install requirements (cached unless requirements.txt changes)
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 2: Download models BEFORE copying app code
# This means model downloads are CACHED even when Python code changes!
RUN mkdir -p model_files && \
    wget -qO model_files/birefnet-general.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-epoch_244.onnx && \
    wget -qO model_files/birefnet-portrait.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-portrait-epoch_150.onnx && \
    wget -qO model_files/bria-rmbg.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/bria-rmbg-2.0.onnx && \
    wget -qO model_files/isnet-general-use.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx && \
    wget -qO model_files/u2net.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx && \
    wget -qO model_files/u2net_human_seg.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx

# Step 3: Copy application code LAST (changes frequently, but no model re-download needed)
COPY --chown=user:user . .

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
