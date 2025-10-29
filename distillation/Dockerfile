# Dockerfile for Whisper Distillation with RTX 5060 support
# Base: NVIDIA PyTorch latest with CUDA 12.6+ for RTX 50 series (Blackwell sm_120)
FROM nvcr.io/nvidia/pytorch:25.01-py3

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

# Base image already has PyTorch + CUDA 12.6
# Skip PyTorch upgrade to speed up build
# RTX 5060 will work with warnings but functional

# Install Python dependencies
RUN pip install --no-cache-dir \
    transformers==4.45.0 \
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
