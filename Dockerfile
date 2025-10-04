# Base image with CUDA + PyTorch
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install dependencies (git, ffmpeg, and build tools)
RUN apt-get update && apt-get install -y git ffmpeg build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy repository files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -e .
RUN pip install runpod requests torchvision torchaudio

# Optional — helps Hugging Face cache models
ENV HF_HOME=/workspace/.cache/huggingface

# Allow environment variables for model + token
ENV MODEL_REPO="stepfun-ai/Step-Video-TI2V"
# You can set HF_TOKEN from RunPod dashboard instead of hardcoding it here

# Expose workspace and set defaults
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Run the serverless handler
CMD ["python3", "handler.py"]
