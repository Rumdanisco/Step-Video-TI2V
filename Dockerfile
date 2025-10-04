FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
WORKDIR /workspace
RUN apt-get update && apt-get install -y git ffmpeg
COPY . .
RUN pip install -e .
RUN pip install runpod requests
CMD ["python3", "handler.py"]
