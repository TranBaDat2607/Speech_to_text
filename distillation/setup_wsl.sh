#!/bin/bash
# WSL Setup Script for Speech-to-Text Distillation
# This script sets up the environment on WSL with minimal disk usage

set -e  # Exit on error

echo "============================================================"
echo "WSL Setup for Speech-to-Text Distillation"
echo "============================================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get Windows username
WIN_USER=$(powershell.exe -Command 'Write-Output $env:USERNAME' | tr -d '\r')
echo -e "${GREEN}Detected Windows user: ${WIN_USER}${NC}"

# ============================================
# Step 1: Configure cache directories
# ============================================
echo ""
echo "Step 1: Configuring cache directories to Windows disk..."

# Create cache directories on Windows
WIN_CACHE_DIR="/mnt/c/Users/${WIN_USER}/.cache"
mkdir -p "${WIN_CACHE_DIR}/huggingface/hub"
mkdir -p "${WIN_CACHE_DIR}/pip"
mkdir -p "/mnt/c/Users/${WIN_USER}/.keras"

echo -e "${GREEN}✓ Cache directories created on Windows${NC}"

# ============================================
# Step 2: Configure bash environment
# ============================================
echo ""
echo "Step 2: Configuring bash environment..."

# Backup existing .bashrc
if [ -f ~/.bashrc ]; then
    cp ~/.bashrc ~/.bashrc.backup
    echo -e "${YELLOW}Backed up existing .bashrc to ~/.bashrc.backup${NC}"
fi

# Add cache configuration to .bashrc
cat >> ~/.bashrc << 'EOL'

# ============================================
# Cache directories - Point to Windows disk
# Added by setup_wsl.sh
# ============================================
EOL

# Get WIN_USER for .bashrc
cat >> ~/.bashrc << EOL
# Hugging Face cache (saves ~7.5GB in WSL disk)
export HF_HOME="/mnt/c/Users/${WIN_USER}/.cache/huggingface"
export TRANSFORMERS_CACHE="/mnt/c/Users/${WIN_USER}/.cache/huggingface/hub"
export HF_DATASETS_CACHE="/mnt/c/Users/${WIN_USER}/.cache/huggingface/datasets"

# TensorFlow/Keras cache
export KERAS_HOME="/mnt/c/Users/${WIN_USER}/.keras"

# pip cache
export PIP_CACHE_DIR="/mnt/c/Users/${WIN_USER}/.cache/pip"

# Python user base
export PYTHONUSERBASE="/mnt/c/Users/${WIN_USER}/.local"

# CUDA environment variables (if needed)
export CUDA_VISIBLE_DEVICES=0

EOL

echo -e "${GREEN}✓ Environment variables added to ~/.bashrc${NC}"

# Source the new configuration
source ~/.bashrc

# ============================================
# Step 3: Check/Install Miniconda
# ============================================
echo ""
echo "Step 3: Checking Miniconda installation..."

if command -v conda &> /dev/null; then
    echo -e "${GREEN}✓ Conda already installed${NC}"
    conda --version
else
    echo -e "${YELLOW}Installing Miniconda...${NC}"
    
    # Download Miniconda
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    
    # Install Miniconda
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    
    # Initialize conda
    $HOME/miniconda3/bin/conda init bash
    
    # Clean up
    rm /tmp/miniconda.sh
    
    echo -e "${GREEN}✓ Miniconda installed${NC}"
    
    # Source to get conda command
    source ~/.bashrc
fi

# ============================================
# Step 4: Create/Update conda environment
# ============================================
echo ""
echo "Step 4: Setting up conda environment..."

# Check if environment exists
if conda env list | grep -q "dat301m"; then
    echo -e "${YELLOW}Environment 'dat301m' already exists${NC}"
    read -p "Do you want to update it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda activate dat301m
        echo -e "${GREEN}Activated existing environment${NC}"
    fi
else
    echo -e "${YELLOW}Creating new environment 'dat301m'...${NC}"
    conda create -n dat301m python=3.10 -y
    conda activate dat301m
    echo -e "${GREEN}✓ Environment created and activated${NC}"
fi

# ============================================
# Step 5: Install requirements
# ============================================
echo ""
echo "Step 5: Installing Python packages..."

# Navigate to project directory
PROJECT_DIR="/mnt/c/Users/${WIN_USER}/Desktop/dat301m/Speech_to_text"

if [ ! -d "$PROJECT_DIR" ]; then
    echo -e "${YELLOW}Warning: Project directory not found at ${PROJECT_DIR}${NC}"
    echo "Please adjust the path if needed"
else
    cd "$PROJECT_DIR"
    
    if [ -f "requirements.txt" ]; then
        echo "Installing packages from requirements.txt..."
        pip install -r requirements.txt --cache-dir "/mnt/c/Users/${WIN_USER}/.cache/pip"
        echo -e "${GREEN}✓ Packages installed${NC}"
        
        # RTX 5000 series specific TensorFlow setup
        echo ""
        echo "Checking for RTX 5000 series GPU..."
        if nvidia-smi | grep -q "RTX 50"; then
            echo -e "${YELLOW}RTX 5000 series detected!${NC}"
            echo "For TensorFlow GPU support, run:"
            echo "  pip install tf-nightly[and-cuda]"
        fi
    else
        echo -e "${YELLOW}requirements.txt not found, skipping package installation${NC}"
    fi
fi

# ============================================
# Step 6: Verify GPU support
# ============================================
echo ""
echo "Step 6: Verifying GPU support..."

python3 << 'EOF'
import sys

# Check PyTorch
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch not installed")

print()

# Check TensorFlow
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPU devices: {len(gpus)}")
    if gpus:
        for gpu in gpus:
            print(f"  - {gpu}")
except ImportError:
    print("TensorFlow not installed")
EOF

# ============================================
# Step 7: Clean up pip cache
# ============================================
echo ""
echo "Step 7: Cleaning up temporary caches..."

pip cache purge
conda clean --all -y

echo -e "${GREEN}✓ Temporary caches cleaned${NC}"

# ============================================
# Step 8: Display disk usage
# ============================================
echo ""
echo "Step 8: Disk usage summary..."

echo ""
echo "WSL Disk Usage:"
df -h ~ | grep -v "Filesystem"

echo ""
echo "Cache locations:"
echo "  Hugging Face: /mnt/c/Users/${WIN_USER}/.cache/huggingface"
echo "  pip: /mnt/c/Users/${WIN_USER}/.cache/pip"
echo "  Keras: /mnt/c/Users/${WIN_USER}/.keras"

# ============================================
# Final instructions
# ============================================
echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Close and reopen your WSL terminal (or run: source ~/.bashrc)"
echo "  2. Activate environment: conda activate dat301m"
echo "  3. Navigate to project: cd /mnt/c/Users/${WIN_USER}/Desktop/dat301m/Speech_to_text/distillation"
echo "  4. For TensorFlow GPU (RTX 5000): pip install tf-nightly[and-cuda]"
echo "  5. Test GPU setup: python test_gpu_setup.py"
echo "  6. Run your scripts!"
echo ""
echo "Environment info:"
echo "  - Conda environment: dat301m"
echo "  - All caches stored on Windows disk (minimal WSL usage)"
echo "  - Expected WSL disk usage: ~6-8 GB"
echo ""
echo "To check WSL disk usage anytime:"
echo "  df -h ~"
echo ""
echo "============================================================"
