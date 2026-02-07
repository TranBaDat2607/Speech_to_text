# Whisper Audio Preprocessing for TensorFlow

TensorFlow implementation để chuyển đổi audio dataset sang format log mel spectrogram cho Whisper model training.

## Chức năng

- Convert WAV files (30s, 16kHz) thành log mel spectrograms
- Format output khớp với OpenAI Whisper specification
- Batch processing cho toàn bộ dataset
- Lưu processed data dưới dạng .npy và .json files

## Files trong project

- `audio_constants.py`: Constants và hyperparameters từ OpenAI Whisper
- `mel_filters.py`: Mel filter bank generation cho TensorFlow
- `audio_processor.py`: Core audio processing functions
- `preprocess_dataset.py`: Main script để process dataset
- `test_preprocessing.py`: Test suite để verify preprocessing pipeline
- `README.md`: Documentation này

## Requirements

```bash
pip install tensorflow numpy
```

## Cách sử dụng

### 1. Test preprocessing pipeline

```bash
cd c:\Users\Admin\Desktop\dat301m\Speech_to_text\preprocessing_data
python test_preprocessing.py
```

### 2. Process default dataset

```bash
python preprocess_dataset.py
```

### 3. Process custom dataset

```bash
python preprocess_dataset.py --input_dir path/to/dataset --output_dir path/to/output
```

## Input format

Dataset directory phải chứa:
- `*.wav`: Audio files (30 giây, 16kHz, mono)  
- `*.json`: Transcript files với cùng tên

Example:
```
dataset/
├── video1_001.wav
├── video1_001.json
├── video1_002.wav 
├── video1_002.json
└── ...
```

## Output format

Processed data sẽ được lưu trong output directory:
- `*_mel.npy`: Log mel spectrograms (shape: 80x3000)
- `*_processed.json`: Metadata và transcripts
- `processing_summary.json`: Summary file

### Mel spectrogram specifications

- Shape: (80, 3000) 
- 80 mel frequency bins
- 3000 time frames (30 seconds)
- Sample rate: 16kHz
- FFT size: 400
- Hop length: 160 (10ms)
- Log scale với normalization [0, 1]

### Processed JSON format

```json
{
  "original_audio_file": "path/to/original.wav",
  "mel_spectrogram_file": "path/to/output_mel.npy",
  "mel_shape": [80, 3000],
  "transcript": "transcript text here",
  "duration": 30.0,
  "sample_rate": 16000,
  "n_mels": 80,
  "n_frames": 3000
}
```

## Sử dụng processed data

### Load single mel spectrogram

```python
import numpy as np
import json

# Load mel spectrogram
mel_spec = np.load('output_mel.npy')  # Shape: (80, 3000)

# Load metadata
with open('output_processed.json', 'r') as f:
    metadata = json.load(f)
    transcript = metadata['transcript']
```

### TensorFlow Dataset

```python
import tensorflow as tf
import numpy as np
import json
from pathlib import Path

def create_tf_dataset(processed_dir):
    """Create TensorFlow dataset from processed files"""
    processed_path = Path(processed_dir)
    mel_files = list(processed_path.glob("*_mel.npy"))
    
    def load_sample(mel_file_path):
        # Load mel spectrogram
        mel_spec = np.load(mel_file_path)
        
        # Load corresponding JSON
        json_path = str(mel_file_path).replace('_mel.npy', '_processed.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        return mel_spec.astype(np.float32), metadata['transcript']
    
    # Create dataset
    dataset = tf.data.Dataset.from_generator(
        lambda: (load_sample(f) for f in mel_files),
        output_signature=(
            tf.TensorSpec(shape=[80, 3000], dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.string)
        )
    )
    
    return dataset.batch(4).prefetch(tf.data.AUTOTUNE)

# Usage
dataset = create_tf_dataset('processed_dataset')
for batch_mel, batch_text in dataset.take(1):
    print(f"Mel batch shape: {batch_mel.shape}")  # (4, 80, 3000)
    print(f"Text batch: {batch_text}")
```

## Troubleshooting

### "No WAV files found"
- Kiểm tra input directory path
- Đảm bảo có files .wav trong directory

### "Audio loading failed"
- Kiểm tra WAV files có format đúng không (16kHz, mono)
- Thử với file WAV khác

### "Shape mismatch in mel spectrogram"
- Audio file có thể không đúng 30 giây
- Preprocessing sẽ tự động pad/trim về đúng length

### "Memory error"
- Process dataset theo batch nhỏ hơn
- Giảm số files process cùng lúc

## Technical notes

Preprocessing pipeline implement chính xác theo OpenAI Whisper specification:
- STFT với Hann window
- 80 mel filter banks
- Log10 transform với clamping
- Dynamic range normalization
- Final normalization về [0, 1] range

Output mel spectrograms tương thích hoàn toàn với Whisper TensorFlow model input requirements.
