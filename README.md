# Speech-to-Text with Whisper TensorFlow

This repository contains various components for speech recognition development. The components are independent and serve different purposes in the speech-to-text workflow.

## Repository Structure

### model/
TensorFlow implementation of OpenAI Whisper architecture.

**Contents:**
- `model.py` - Main Whisper model implementation
- `transformer_encoder_block.py` - Audio encoder blocks
- `transformer_decoder_block.py` - Text decoder blocks
- `model_dimensions.py` - Model size configurations (tiny, base, small)
- `tokenizer.py` - Text tokenization utilities
- `data_loader.py` - Dataset loading utilities

**Purpose:** Provides TensorFlow implementation of Whisper model architecture with Conformer encoder and compact vocabulary optimization.

See [`model/README.md`](model/README.md) for details.

---

### crawl_data/
YouTube data collection pipeline for creating speech datasets.

**Contents:**
- `channel_to_dataset_pipeline.py` - Main pipeline orchestrator
- `config.json` - Configuration file
- `youtube_audio_processor.py` - Audio download and segmentation
- `download_subtitles_ytdlp.py` - Subtitle extraction
- `create_dataset_labels.py` - Dataset label generation
- `checkpoint_manager.py` - Resume capability
- `file_validators.py` - Data validation

**Purpose:** Automated system to crawl YouTube channels, download audio and subtitles, segment audio into 30-second chunks, and generate paired WAV + JSON dataset files.

**Usage:**
```bash
cd crawl_data
cp config.json.example config.json
# Edit config.json with channel URLs and settings
python channel_to_dataset_pipeline.py
```

**Output:** `datasets/{channel}/dataset/*.wav` and `*.json` files

See [`crawl_data/README.md`](crawl_data/README.md) for details.

---

### preprocessing_data/
Audio preprocessing to convert WAV files into mel spectrograms.

**Contents:**
- `preprocess_dataset.py` - Main preprocessing script
- `audio_processor.py` - Audio processing functions
- `mel_filters.py` - Mel filter bank generation
- `audio_constants.py` - Whisper audio constants
- `test_preprocessing.py` - Testing suite

**Purpose:** Converts audio files (WAV, 16kHz, 30s) into log mel spectrograms matching OpenAI Whisper specifications. Outputs `.npy` files with shape (80, 3000) and corresponding `.json` metadata files.

**Usage:**
```bash
cd preprocessing_data
python preprocess_dataset.py \
  --input_dir ../crawl_data/datasets/vtv24/dataset \
  --output_dir ./preprocessed_data
```

**Output:** `*_mel.npy` (mel spectrograms) and `*_processed.json` (metadata)

See [`preprocessing_data/README.md`](preprocessing_data/README.md) for details.

---

### distillation/
Experimental code for offline knowledge distillation training for low vram gpu. 

**Contents:**
- `config/` - Configuration files
- `data/` - Dataset loaders
- `losses/` - Distillation loss functions
- `scripts/` - Training and generation scripts
- `student/` - Student model implementations
- `teacher/` - Teacher model loaders
- `training/` - Training utilities

**Warning:** This folder contains highly experimental code from research experiments. It includes:
- Multiple incomplete implementations
- Inconsistent configurations
- Undocumented code paths
- Memory optimization hacks

**Not recommended for use.** This code is chaotic and only god know what it does. Kept for reference purposes only. 
---

## Requirements

- Python 3.8+
- TensorFlow 2.12+
- FFmpeg (for audio processing)

Install dependencies:
```bash
pip install tensorflow numpy yt-dlp pydub requests beautifulsoup4
```

FFmpeg installation:
- Windows: Download from https://ffmpeg.org/download.html
- macOS: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`

---

## Component Usage

Each component can be used independently:

**1. Collect speech data from YouTube:**
```bash
cd crawl_data
python channel_to_dataset_pipeline.py
```

**2. Convert audio to mel spectrograms:**
```bash
cd preprocessing_data
python preprocess_dataset.py --input_dir <path> --output_dir <path>
```

**3. Use Whisper model implementation:**
```python
from model import Whisper
from model_dimensions import get_whisper_dimensions

dims = get_whisper_dimensions("base")
model = Whisper(dims)
```

---

## Notes

- These components are independent tools, not a complete training pipeline
- The `distillation/` folder is experimental and should not be used
- For model training, you will need to implement your own training loop using the components provided
- Each component has its own README with detailed documentation

---

## Acknowledgments

- OpenAI Whisper - Original architecture
- TensorFlow Team - Deep learning framework
- yt-dlp - YouTube downloading library
- PhoWhisper - Vietnamese Whisper model by VinAI
