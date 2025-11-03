#!/bin/bash

# Quick Test Checkpoint Quality
# Run after each mini-batch training

cd /mnt/c/Users/Admin/Desktop/dat301m/Speech_to_text

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================================================${NC}"
echo -e "${BLUE}QUICK CHECKPOINT QUALITY TEST${NC}"
echo -e "${BLUE}=======================================================================${NC}"

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dat301m

# Test file
AUDIO_FILE="/mnt/c/Users/Admin/Desktop/dat301m/Speech_to_text/test/waves/VIVOSDEV01/VIVOSDEV01_R002.wav"
GROUND_TRUTH="Tuyến cộng cận, cứng lại có ở những cấp sắc."

# Test with latest checkpoint
echo -e "\n${GREEN}Testing latest checkpoint...${NC}\n"

python test_checkpoint_quality.py \
    --audio "$AUDIO_FILE" \
    --ground-truth "$GROUND_TRUTH" \
    --model base \
    --language vi

echo ""
echo -e "${YELLOW}=======================================================================${NC}"
echo -e "${YELLOW}Test completed!${NC}"
echo -e "${YELLOW}=======================================================================${NC}"
echo ""
echo "If quality is OK:"
echo "  → Resume training: cd distillation && python scripts/mini_batch_orchestrator.py --config config/distillation_config.yaml --resume"
echo ""
echo "If quality degraded:"
echo "  → Review training config and checkpoints"
echo ""
