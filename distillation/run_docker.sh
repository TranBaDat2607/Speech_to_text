#!/bin/bash
# Script to build and run Docker container for Whisper distillation

echo "============================================================"
echo "Whisper Distillation - Docker Setup (RTX 5060)"
echo "============================================================"

# Check if Docker is running
if ! docker ps > /dev/null 2>&1; then
    echo "Error: Docker is not running!"
    echo "Please start Docker Desktop first."
    exit 1
fi

# Build image
echo ""
echo "Step 1: Building Docker image..."
docker-compose build

if [ $? -ne 0 ]; then
    echo "Error: Failed to build Docker image"
    exit 1
fi

echo ""
echo "Step 2: Starting container..."
docker-compose up -d

if [ $? -ne 0 ]; then
    echo "Error: Failed to start container"
    exit 1
fi

echo ""
echo "============================================================"
echo "Container started successfully!"
echo "============================================================"
echo ""
echo "To access the container:"
echo "  docker-compose exec whisper-distillation bash"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f"
echo ""
echo "To stop container:"
echo "  docker-compose down"
echo ""
echo "============================================================"
