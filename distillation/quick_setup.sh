#!/bin/bash
# Quick setup script for RTX 5060 Docker environment

set -e  # Exit on error

echo ""
echo "============================================================"
echo "  Whisper Distillation - RTX 5060 Docker Setup"
echo "============================================================"
echo ""

# Check if running in WSL
if ! grep -qi microsoft /proc/version; then
    echo "WARNING: Not running in WSL. Please run this script in WSL2."
    exit 1
fi

echo "OK: Running in WSL2"
echo ""

# Check Docker
echo "Checking Docker..."
if ! docker ps > /dev/null 2>&1; then
    echo "ERROR: Docker is not running!"
    echo "   Please start Docker Desktop and try again."
    exit 1
fi
echo "OK: Docker is running"
echo ""

# Check NVIDIA driver
echo "Checking NVIDIA driver..."
if ! nvidia-smi > /dev/null 2>&1; then
    echo "ERROR: NVIDIA driver not accessible!"
    echo "   Make sure NVIDIA driver is installed on Windows (>= 576.88)"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
echo "OK: GPU detected: $GPU_NAME"
echo "  Driver version: $DRIVER_VERSION"
echo ""

# Ask user to proceed
echo "This will:"
echo "  1. Stop and remove existing containers"
echo "  2. Build new Docker image with PyTorch + CUDA 12.4"
echo "  3. Start container"
echo "  4. Run GPU tests"
echo ""
read -p "Proceed? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled."
    exit 0
fi

echo ""
echo "============================================================"
echo "Step 1: Cleaning up old containers..."
echo "============================================================"
docker-compose down 2>/dev/null || true
docker rmi whisper-distillation:rtx5060 2>/dev/null || true
echo "OK: Cleanup completed"

echo ""
echo "============================================================"
echo "Step 2: Building Docker image (this may take 5-10 minutes)..."
echo "============================================================"
docker-compose build

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to build Docker image"
    exit 1
fi
echo "OK: Image built successfully"

echo ""
echo "============================================================"
echo "Step 3: Starting container..."
echo "============================================================"
docker-compose up -d

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to start container"
    exit 1
fi
echo "OK: Container started"

# Wait for container to be ready
echo ""
echo "Waiting for container to be ready..."
sleep 5

echo ""
echo "============================================================"
echo "Step 4: Running GPU tests..."
echo "============================================================"
chmod +x test_docker_gpu.sh
./test_docker_gpu.sh

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: GPU tests failed!"
    echo "   Check the output above for errors."
    echo "   See documentation for troubleshooting."
    exit 1
fi

echo ""
echo "============================================================"
echo "Step 5: Testing Whisper Teacher Model..."
echo "============================================================"
echo "This will download Whisper Large model (~3GB)..."
echo ""

docker-compose exec -T whisper-distillation python teacher/load_teacher_pytorch.py

if [ $? -ne 0 ]; then
    echo ""
    echo "WARNING: Teacher model test failed, but container is running."
    echo "   You can manually test later with:"
    echo "   docker-compose exec whisper-distillation python teacher/load_teacher_pytorch.py"
else
    echo ""
    echo "OK: Teacher model loaded successfully!"
fi

echo ""
echo "============================================================"
echo "              SETUP COMPLETED SUCCESSFULLY"
echo "============================================================"
echo ""
echo "Your RTX 5060 Docker environment is ready!"
echo ""
echo "Container info:"
echo "  Name: whisper-distillation"
echo "  GPU: $GPU_NAME"
echo "  Status: Running"
echo ""
echo "Next steps:"
echo "  1. Access container:"
echo "     docker-compose exec whisper-distillation bash"
echo ""
echo "  2. Update config:"
echo "     nano config/distillation_config.yaml"
echo ""
echo "  3. Generate teacher logits (6-12 hours):"
echo "     python scripts/step1_generate_teacher_logits.py"
echo ""
echo "  4. Train student model (1-2 weeks):"
echo "     python scripts/step2_train_distillation.py"
echo ""
echo "  5. View logs:"
echo "     docker-compose logs -f"
echo ""
echo "Documentation:"
echo "  - Setup guide: SETUP_RTX5060.md"
echo "  - Full README: README.md"
echo ""
echo "============================================================"
