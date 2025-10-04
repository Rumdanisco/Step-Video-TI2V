# Base image with CUDA + PyTorch
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install dependencies (git, ffmpeg, and build tools)
RUN apt-get update && apt-get install -y git ffmpeg build-essential

# Copy repository files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -e .
RUN pip install runpod requests

# (Optional but recommended) install torchvision + torchaudio matching torch version
RUN pip install torchvision torchaudio

# Expose workspace
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Run the serverless handler
CMD ["python3", "handler.py"]
