# Dockerfile for Whisper Distillation with RTX 5060 support
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir \
    transformers==4.57.1 \
    accelerate \
    librosa \
    soundfile \
    jiwer \
    pyyaml \
    tqdm \
    tensorboard \
    pandas \
    datasets \
    numpy \
    huggingface-hub

# Set environment variables
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers

# Default command
CMD ["/bin/bash"]
