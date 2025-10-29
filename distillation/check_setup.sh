#!/bin/bash
# Script kiểm tra Docker setup đã sẵn sàng chưa

echo "============================================================"
echo "KIEM TRA DOCKER SETUP"
echo "============================================================"
echo ""

# Test 1: Container có chạy không?
echo "1. Kiem tra container..."
if docker ps | grep -q whisper-distillation; then
    echo "   OK: Container dang chay"
    CONTAINER_RUNNING=true
else
    echo "   ERROR: Container KHONG chay"
    echo "   -> Chay: docker-compose up -d"
    CONTAINER_RUNNING=false
fi
echo ""

# Test 2: GPU có được nhận diện không?
echo "2. Kiem tra GPU..."
if [ "$CONTAINER_RUNNING" = true ]; then
    GPU_OUTPUT=$(docker-compose exec -T whisper-distillation nvidia-smi 2>&1)
    if echo "$GPU_OUTPUT" | grep -q "GeForce RTX"; then
        GPU_NAME=$(echo "$GPU_OUTPUT" | grep "GeForce RTX" | awk '{print $4, $5, $6}' | head -1)
        echo "   OK: GPU: $GPU_NAME"
    else
        echo "   ERROR: GPU khong duoc nhan dien"
    fi
else
    echo "   SKIP: Bo qua (container chua chay)"
fi
echo ""

# Test 3: PyTorch có hoạt động không?
echo "3. Kiem tra PyTorch + CUDA..."
if [ "$CONTAINER_RUNNING" = true ]; then
    PYTORCH_TEST=$(docker-compose exec -T whisper-distillation python -c "
import torch
print(f'PyTorch:{torch.__version__}')
print(f'CUDA:{torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:{torch.cuda.get_device_name(0)}')
" 2>&1)
    
    if echo "$PYTORCH_TEST" | grep -q "CUDA:True"; then
        echo "   OK: PyTorch CUDA hoat dong"
        echo "$PYTORCH_TEST" | sed 's/^/      /'
    else
        echo "   ERROR: PyTorch khong truy cap duoc GPU"
        echo "$PYTORCH_TEST" | sed 's/^/      /'
    fi
else
    echo "   SKIP: Bo qua (container chua chay)"
fi
echo ""

# Test 4: Transformers có cài đúng không?
echo "4. Kiem tra Transformers library..."
if [ "$CONTAINER_RUNNING" = true ]; then
    TRANSFORMERS_TEST=$(docker-compose exec -T whisper-distillation python -c "
import transformers
print(f'Version:{transformers.__version__}')
from transformers import WhisperProcessor
print('WhisperProcessor:OK')
" 2>&1)
    
    if echo "$TRANSFORMERS_TEST" | grep -q "WhisperProcessor:OK"; then
        echo "   OK: Transformers cai dat dung"
        VERSION=$(echo "$TRANSFORMERS_TEST" | grep "Version:" | cut -d: -f2)
        echo "      Version: $VERSION"
    else
        echo "   ERROR: Transformers co loi"
        echo "$TRANSFORMERS_TEST" | sed 's/^/      /'
    fi
else
    echo "   SKIP: Bo qua (container chua chay)"
fi
echo ""

# Test 5: Dataset có sẵn không?
echo "5. Kiem tra dataset..."
if [ -d "./preprocessing_data/phoaudiobook_100h/audio" ]; then
    AUDIO_COUNT=$(find ./preprocessing_data/phoaudiobook_100h/audio -name "*.wav" 2>/dev/null | wc -l)
    if [ "$AUDIO_COUNT" -gt 0 ]; then
        echo "   OK: Dataset san sang: $AUDIO_COUNT audio files"
    else
        echo "   WARNING: Folder ton tai nhung chua co audio files"
    fi
    
    if [ -f "./preprocessing_data/phoaudiobook_100h/metadata.json" ]; then
        echo "   OK: metadata.json co san"
    else
        echo "   ERROR: metadata.json khong tim thay"
    fi
else
    echo "   ERROR: Dataset chua download"
    echo "   -> Chay: python scripts/download_asr_dataset_100h.py"
fi
echo ""

# Test 6: Config file
echo "6. Kiem tra config file..."
if [ -f "./config/distillation_config.yaml" ]; then
    echo "   OK: Config file ton tai"
else
    echo "   ERROR: Config file khong tim thay"
fi
echo ""

# Summary
echo "============================================================"
echo "KET QUA TONG HOP"
echo "============================================================"

if [ "$CONTAINER_RUNNING" = true ] && \
   echo "$PYTORCH_TEST" | grep -q "CUDA:True" && \
   echo "$TRANSFORMERS_TEST" | grep -q "WhisperProcessor:OK" && \
   [ "$AUDIO_COUNT" -gt 0 ]; then
    echo ""
    echo "SETUP HOAN CHINH - SAN SANG TRAINING!"
    echo ""
    echo "Buoc tiep theo:"
    echo "  1. Vào container:"
    echo "     docker-compose exec whisper-distillation bash"
    echo ""
    echo "  2. Test teacher model:"
    echo "     python teacher/load_teacher_pytorch.py"
    echo ""
    echo "  3. Bắt đầu training:"
    echo "     python scripts/step1_generate_teacher_logits.py"
    echo ""
else
    echo ""
    echo "WARNING: SETUP CHUA HOAN CHINH"
    echo ""
    echo "Can khac phuc:"
    
    if [ "$CONTAINER_RUNNING" = false ]; then
        echo "  - Start container: docker-compose up -d"
    fi
    
    if ! echo "$PYTORCH_TEST" | grep -q "CUDA:True"; then
        echo "  - Fix PyTorch GPU: Rebuild Docker image"
    fi
    
    if ! echo "$TRANSFORMERS_TEST" | grep -q "WhisperProcessor:OK"; then
        echo "  - Fix Transformers: Reinstall trong container"
    fi
    
    if [ "$AUDIO_COUNT" -eq 0 ]; then
        echo "  - Download dataset: python scripts/download_asr_dataset_100h.py"
    fi
    echo ""
fi

echo "============================================================"
