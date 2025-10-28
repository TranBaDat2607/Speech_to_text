#!/bin/bash
# Quick test script for Docker GPU setup

echo "============================================================"
echo "Testing GPU in Docker Container"
echo "============================================================"

# Test 1: GPU detection
echo ""
echo "Test 1: GPU Detection"
docker-compose exec whisper-distillation nvidia-smi

# Test 2: TensorFlow GPU
echo ""
echo "Test 2: TensorFlow GPU"
docker-compose exec whisper-distillation python -c "
import tensorflow as tf
print(f'TensorFlow: {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs found: {len(gpus)}')
for gpu in gpus:
    print(f'  {gpu}')
"

# Test 3: Transformers import
echo ""
echo "Test 3: Transformers Library"
docker-compose exec whisper-distillation python -c "
from transformers import TFWhisperModel, WhisperProcessor
print('✓ Transformers imported successfully')
print('✓ TFWhisperModel available')
"

echo ""
echo "============================================================"
echo "All tests completed!"
echo "============================================================"
