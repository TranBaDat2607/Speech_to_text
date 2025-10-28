# Knowledge Distillation Pipeline for Whisper Vietnamese

**Docker + PyTorch + RTX 5060 Setup**

Distillation from **Whisper Large-v2** (teacher) → **Whisper Small** (student) on PhoWhisper dataset.

---

## 🎯 Project Goal

Train Whisper Small model achieving **85-90%** of Whisper Large performance using knowledge distillation.

**Target Metrics (100h dataset):**
- WER: 8.5-10% (vs Large: ~6.5%)
- Model size: 244M params (vs Large: 1550M)
- Speed: 4-5x faster inference
- Hardware: RTX 5060 8GB with Docker

---

## 🖥️ System Requirements

### Hardware
- GPU: RTX 5060 8GB (hoặc GPU tương đương)
- RAM: 16GB+
- Storage: 60GB+ free space

### Software
- Windows 11 (hoặc Windows 10 21H2+)
- WSL2 (Ubuntu 22.04 recommended)
- Docker Desktop for Windows
- NVIDIA Driver: 576.88+

---

# 📖 HƯỚNG DẪN SETUP HOÀN CHỈNH

## PHẦN 1: SETUP WSL2 + DOCKER

### Bước 1: Cài đặt WSL2 (trên Windows PowerShell admin)

```powershell
# Enable WSL
wsl --install

# Hoặc nếu đã có WSL1, upgrade lên WSL2
wsl --set-default-version 2

# Cài Ubuntu 22.04
wsl --install -d Ubuntu-22.04
```

**Restart máy sau khi cài!**

---

### Bước 2: Cài Docker Desktop for Windows

1. Download Docker Desktop từ: https://www.docker.com/products/docker-desktop/
2. Cài đặt và khởi động Docker Desktop
3. Trong Docker Desktop Settings:
   - Vào **Resources** → **WSL Integration**
   - Enable cho Ubuntu-22.04
   - Click **Apply & Restart**

---

### Bước 3: Verify Docker và GPU trong WSL

Mở WSL terminal:

```bash
# Check Docker
docker --version
docker ps

# Check GPU accessible
nvidia-smi
```

**Expected:**
```
Docker version 28.x.x
NVIDIA GeForce RTX 5060
```

---

### Bước 4: Test NVIDIA Container Toolkit

```bash
# Test GPU in Docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

**Expected:** Thấy RTX 5060 info

---

## PHẦN 2: SETUP DOCKER CONTAINER

### Bước 5: Pull NVIDIA TensorFlow Image

```bash
# Navigate to project
cd /mnt/c/Users/Admin/Desktop/dat301m/Speech_to_text/distillation

# Pull image (chứa PyTorch + TensorFlow + CUDA)
docker pull nvcr.io/nvidia/tensorflow:25.02-tf2-py3
```

**Thời gian:** ~5-10 phút (tải ~10GB)

---

### Bước 6: Build Custom Docker Image

```bash
# Build image với dependencies
chmod +x run_docker.sh
./run_docker.sh
```

**Image bao gồm:**
- PyTorch 2.9.0 + CUDA 12.8 (hỗ trợ RTX 5060)
- Transformers + Whisper models
- librosa, soundfile, jiwer
- TensorBoard, pandas, numpy

---

### Bước 7: Test GPU trong Container

```bash
# Test script
./test_docker_gpu.sh
```

**Expected output:**
```
GPUs found: 1
PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')
✓ Transformers imported successfully
```

---

## PHẦN 3: TEST TEACHER MODEL

### Bước 8: Access Container và Test

```bash
# Vào container
docker-compose exec whisper-distillation bash

# Trong container, test PyTorch teacher
python teacher/load_teacher_pytorch.py
```

**Expected:**
```
Loading Whisper Teacher Model (PyTorch)
Model: openai/whisper-large-v2
Device: cuda
GPU: NVIDIA GeForce RTX 5060
✓ Model loaded on cuda
✓ Forward pass successful!
ALL TESTS PASSED!
```

---

### Bước 9: Configure Pipeline

Edit `config/distillation_config.yaml`:

```yaml
paths:
  preprocessed_dataset: "../preprocessing_data/processed_dataset/"
  teacher_logits_dir: "./data/teacher_logits/"
  checkpoints_dir: "./checkpoints/"

data:
  use_full_dataset: false
  max_hours: 100           # 100 giờ data
  train_split: 0.95

distillation:
  epochs: 20               # Tăng vì ít data
  batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 5.0e-5
  mixed_precision: true
```

---

## PHẦN 4: CHẠY DISTILLATION PIPELINE

### Pipeline Overview

```
Step 1: Generate Teacher Logits (6-12 giờ với 100h data)
   ↓
Step 2: Train Student (1-2 tuần)
   ↓
Step 3: Evaluate (1 ngày)
```

---

### Step 1: Generate Teacher Logits

```bash
# Trong Docker container
python scripts/step1_generate_teacher_logits.py
```

**Output:**
- `data/teacher_logits/` chứa soft labels (~30-35GB cho 100h)
- Thời gian: 6-12 giờ với RTX 5060

---

### Step 2: Train Student

```bash
# Trong Docker container
python scripts/step2_train_distillation.py
```

**Monitoring:**
```bash
# Từ WSL (ngoài container)
docker-compose exec whisper-distillation tensorboard --logdir=/workspace/distillation/logs/tensorboard/ --host=0.0.0.0
```

Access: http://localhost:6006

---

### Step 3: Evaluate

```bash
# Trong Docker container
python scripts/step3_evaluate_model.py
```

**Expected WER:** 8.5-10% trên Vietnamese test set (100h data)

---

## 📁 Project Structure

```
distillation/
├── README.md                    # Bạn đang đọc file này
├── Dockerfile                   # Docker image definition
├── docker-compose.yml           # Docker orchestration
├── run_docker.sh                # Build & start container
├── test_docker_gpu.sh           # Test GPU setup
├── config/
│   └── distillation_config.yaml # Cấu hình chính
│
├── teacher/                     # Teacher model (PyTorch)
│   ├── load_teacher_pytorch.py # Load Whisper Large (MAIN)
│   ├── load_teacher_tf.py      # TF version (deprecated)
│   ├── verify_teacher.py       # Verify teacher
│   └── teacher_utils.py        # Utilities
│
├── student/                     # Student model
│   ├── load_student.py
│   └── model_setup.py
│
├── data/                        # Data processing
│   └── teacher_logits/         # Generated soft labels
│
├── training/                    # Training components
│   ├── distillation_trainer.py
│   └── loss_functions.py
│
├── evaluation/                  # Evaluation
│   ├── evaluator.py
│   └── metrics.py
│
├── scripts/                     # Main execution scripts
│   ├── step1_generate_teacher_logits.py
│   ├── step2_train_distillation.py
│   └── step3_evaluate_model.py
│
├── checkpoints/                 # Model checkpoints (gitignored)
└── logs/                        # Training logs (gitignored)
```

---

## 🔧 Configuration Details

### Dataset Configuration (100h)

```yaml
data:
  use_full_dataset: false
  max_hours: 100           # Giới hạn 100 giờ
  max_samples: null
  train_split: 0.95        # 95h train, 5h validation
  random_seed: 42
```

### Distillation Hyperparameters

```yaml
distillation:
  soft_loss_weight: 0.7    # KL divergence với teacher
  hard_loss_weight: 0.3    # Cross entropy với ground truth
  temperature: 3.0         # Temperature scaling
  
  epochs: 20               # Nhiều epochs vì ít data
  batch_size: 2
  gradient_accumulation_steps: 4
  effective_batch_size: 8
  
  learning_rate: 5.0e-5
  warmup_steps: 200
  mixed_precision: true    # FP16 cho 8GB VRAM
```

---

## 📊 Expected Results

### Performance (100h dataset)

| Model | WER | Params | Inference Speed |
|-------|-----|--------|----------------|
| Teacher (Large) | 6.5% | 1550M | 1.0x |
| Student (Pretrained) | 11-12% | 244M | 5.2x |
| **Student (Distilled)** | **8.5-10%** | **244M** | **5.2x** |

**Improvement:** 2-3% WER reduction vs pretrained Small

---

### Timeline (RTX 5060 8GB)

```
Week 1:
  Day 1-2: Setup WSL + TensorFlow
  Day 3-4: Generate teacher logits (overnight)
  Day 5-7: Start training

Week 2-3:
  Main distillation training
  
Week 4:
  Fine-tuning + evaluation
  
Total: 3-4 weeks
```

---

## 💡 Tips & Troubleshooting

### Memory Optimization

**OOM Error:**
```yaml
batch_size: 1
gradient_accumulation_steps: 8
```

### Training Tips

1. **Monitor GPU memory:**
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Save checkpoints frequently:**
   ```yaml
   save_every_n_steps: 500
   ```

3. **Enable gradient checkpointing nếu OOM:**
   ```yaml
   use_gradient_checkpointing: true
   ```

---

### Common Issues

**Issue 1: Docker không start**
```bash
# Check Docker Desktop đang chạy
# Enable WSL integration trong Docker Settings
```

**Issue 2: Container không thấy GPU**
```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# Nếu lỗi: Update NVIDIA driver trên Windows
```

**Issue 3: Out of memory trong container**
```yaml
# Trong load_teacher_pytorch.py, GPU memory đã set 6GB
# Nếu vẫn OOM: giảm batch_size trong config
batch_size: 1
```

**Issue 4: Container restart mất data**
```bash
# Data được persist qua Docker volumes
# Check: docker volume ls
# Checkpoints và logs trong ./checkpoints/ và ./logs/
```

---

## 🚀 Quick Start Commands

### Setup Docker (one-time)

```bash
# Navigate to project
cd /mnt/c/Users/Admin/Desktop/dat301m/Speech_to_text/distillation

# Build and start container
./run_docker.sh

# Test GPU
./test_docker_gpu.sh
```

---

### Run Distillation Pipeline

```bash
# 1. Access container
docker-compose exec whisper-distillation bash

# 2. Test teacher model
python teacher/load_teacher_pytorch.py

# 3. Generate teacher logits (6-12 giờ)
python scripts/step1_generate_teacher_logits.py

# 4. Train student (1-2 tuần)
python scripts/step2_train_distillation.py

# 5. Evaluate
python scripts/step3_evaluate_model.py
```

---

### Docker Management

```bash
# Start container
docker-compose up -d

# Stop container
docker-compose down

# View logs
docker-compose logs -f

# Restart
docker-compose restart
```

---

## 📚 References

- [Whisper Paper](https://arxiv.org/abs/2212.04356)
- [Knowledge Distillation](https://arxiv.org/abs/1503.02531)
- [Docker + NVIDIA GPU](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [PyTorch RTX 5060 Support](https://github.com/pytorch/pytorch/issues)
- [NVIDIA NGC Containers](https://catalog.ngc.nvidia.com/)

---

## 📧 Support

**Docker issues:** Check `./test_docker_gpu.sh` output  
**GPU issues:** Run `nvidia-smi` trong container  
**Training issues:** Check `logs/` folder  
**Performance:** Review `config/distillation_config.yaml`

---

## ⚙️ Technical Stack

- **Base Image:** `nvcr.io/nvidia/tensorflow:25.02-tf2-py3`
- **PyTorch:** 2.9.0 + CUDA 12.8
- **Transformers:** 4.57.1
- **Teacher Model:** `openai/whisper-large-v2` (PyTorch)
- **Student Model:** `openai/whisper-small` (TensorFlow - planned)
- **GPU:** NVIDIA RTX 5060 8GB
