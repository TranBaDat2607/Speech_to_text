#!/bin/bash
# Quick test script for Docker GPU setup (RTX 5060)

echo "============================================================"
echo "Testing GPU in Docker Container (RTX 5060)"
echo "============================================================"

# Test 1: GPU detection
echo ""
echo "Test 1: GPU Detection"
docker-compose exec whisper-distillation nvidia-smi

if [ $? -ne 0 ]; then
    echo "ERROR: GPU not detected!"
    exit 1
fi

# Test 2: PyTorch + CUDA
echo ""
echo "Test 2: PyTorch + CUDA (Critical for RTX 5060)"
docker-compose exec whisper-distillation python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    print(f'GPU Count: {torch.cuda.device_count()}')
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'GPU Memory: {mem_gb:.1f} GB')
    print('OK: PyTorch GPU support: WORKING')
else:
    print('ERROR: PyTorch cannot access GPU!')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "ERROR: PyTorch GPU test failed!"
    exit 1
fi

# Test 3: Transformers library
echo ""
echo "Test 3: Transformers Library"
docker-compose exec whisper-distillation python -c "
from transformers import WhisperProcessor, WhisperForConditionalGeneration
print('OK: Transformers imported successfully')
print('OK: WhisperForConditionalGeneration available (PyTorch)')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Transformers test failed!"
    exit 1
fi

# Test 4: Quick tensor operation on GPU
echo ""
echo "Test 4: GPU Tensor Operation"
docker-compose exec whisper-distillation python -c "
import torch
if torch.cuda.is_available():
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    print(f'OK: GPU tensor operation successful')
    print(f'  Input shape: {x.shape}')
    print(f'  Output shape: {y.shape}')
else:
    print('ERROR: GPU not available for tensor operations')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "ERROR: GPU tensor operation failed!"
    exit 1
fi

echo ""
echo "============================================================"
echo "ALL TESTS PASSED! RTX 5060 is ready for training!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Test teacher model: docker-compose exec whisper-distillation python teacher/load_teacher_pytorch.py"
echo "  2. Configure: nano config/distillation_config.yaml"
echo "  3. Start training: python scripts/step1_generate_teacher_logits.py"
echo ""
