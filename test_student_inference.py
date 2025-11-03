"""
Test Student Model Inference with OpenAI Pretrained Weights
Load OpenAI weights (converted to TensorFlow), transcribe audio file
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "model"))
sys.path.insert(0, str(Path(__file__).parent / "distillation" / "student"))

from load_student_tensorflow import WhisperStudentTensorFlow

print("="*70)
print("TESTING STUDENT MODEL INFERENCE WITH OPENAI WEIGHTS")
print("="*70)


def load_audio(audio_path: str, sr: int = 16000):
    """
    Load audio file and convert to mel spectrogram
    
    Args:
        audio_path: Path to audio file (.wav, .mp3, etc.)
        sr: Target sampling rate (16000 for Whisper)
        
    Returns:
        mel: Mel spectrogram [1, 80, 3000]
    """
    try:
        import librosa
    except ImportError:
        print("Error: librosa not installed")
        print("Install with: pip install librosa")
        sys.exit(1)
    
    # Load audio
    print(f"\n[1] Loading audio: {audio_path}")
    audio, _ = librosa.load(audio_path, sr=sr)
    print(f"    Audio shape: {audio.shape}, duration: {len(audio)/sr:.2f}s")
    
    # Pad or trim to 30 seconds (480000 samples at 16kHz)
    target_length = sr * 30  # 30 seconds
    if len(audio) > target_length:
        audio = audio[:target_length]
        print(f"    Trimmed to 30s")
    else:
        audio = np.pad(audio, (0, target_length - len(audio)))
        print(f"    Padded to 30s")
    
    # Compute mel spectrogram
    print(f"\n[2] Computing mel spectrogram...")
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=400,
        hop_length=160,
        n_mels=80,
        fmin=0,
        fmax=8000
    )
    
    # Trim to exactly 3000 frames (OpenAI Whisper spec)
    # Librosa may produce 3001 frames due to rounding
    if mel.shape[1] > 3000:
        mel = mel[:, :3000]
        print(f"    Trimmed mel frames: {mel.shape[1]} -> 3000")
    elif mel.shape[1] < 3000:
        # Pad if needed (rare case)
        mel = np.pad(mel, ((0, 0), (0, 3000 - mel.shape[1])), mode='constant')
        print(f"    Padded mel frames: {mel.shape[1]} -> 3000")
    
    # Convert to log scale
    mel = np.log10(np.maximum(mel, 1e-10))
    
    # Normalize
    mel = (mel + 4.0) / 4.0
    
    # Add batch dimension: [80, 3000] -> [1, 80, 3000]
    mel = mel[np.newaxis, ...]
    
    print(f"    Mel shape: {mel.shape}")
    
    return tf.convert_to_tensor(mel, dtype=tf.float32)


def greedy_decode(model, audio_features, tokenizer, max_length: int = 224, language: str = "vi"):
    """
    Simple greedy decoding
    
    Args:
        model: Whisper model
        audio_features: Encoder output [1, 1500, hidden_size]
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        language: Language code (default: "vi" for Vietnamese)
        
    Returns:
        text: Decoded text
    """
    # Create initial tokens: <|startoftranscript|><|vi|><|transcribe|><|notimestamps|>
    sot = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    lang = tokenizer.convert_tokens_to_ids(f"<|{language}|>")
    task = tokenizer.convert_tokens_to_ids("<|transcribe|>")
    no_timestamps = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    eot = tokenizer.eos_token_id
    
    # Initialize token sequence
    tokens = [sot, lang, task, no_timestamps]
    tokens_tensor = tf.constant([tokens], dtype=tf.int32)
    
    print(f"\n[4] Greedy decoding...")
    print(f"    Initial tokens: {tokens}")
    
    # Autoregressive generation
    for i in range(max_length):
        # Get logits from decoder
        logits = model.decoder(tokens_tensor, audio_features, training=False)
        
        # Get next token (greedy: argmax of last position)
        next_token_logits = logits[0, -1, :]  # [vocab_size]
        next_token = tf.argmax(next_token_logits, axis=-1, output_type=tf.int32)
        next_token_value = int(next_token.numpy())
        
        # Check for end of text
        if next_token_value == eot:
            print(f"    Generated {i+1} tokens (stopped at EOT)")
            break
        
        # Append token
        tokens.append(next_token_value)
        tokens_tensor = tf.constant([tokens], dtype=tf.int32)
        
        # Progress
        if (i + 1) % 10 == 0:
            print(f"    Generated {i+1} tokens...")
    
    # Decode to text
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    
    return text


def test_inference(audio_path: str, model_name: str = "base", language: str = "vi"):
    """
    Test inference on audio file
    
    Args:
        audio_path: Path to audio file
        model_name: Model size ("tiny", "base", "small", "medium", "large")
        language: Language code (default: "vi")
    """
    # Check if audio file exists
    if not os.path.exists(audio_path):
        print(f"\nError: Audio file not found: {audio_path}")
        print("\nPlease provide a valid audio file path (.wav, .mp3, etc.)")
        sys.exit(1)
    
    # Load student model with OpenAI pretrained weights
    print(f"\n[Step 1] Loading Student Model ({model_name})")
    print("-" * 70)
    student = WhisperStudentTensorFlow(
        model_name=model_name,
        load_openai_weights=True
    )
    model = student.model
    
    # Load tokenizer
    print(f"\n[Step 2] Loading Tokenizer")
    print("-" * 70)
    try:
        from transformers import WhisperProcessor
        processor = WhisperProcessor.from_pretrained("vinai/PhoWhisper-large")
        tokenizer = processor.tokenizer
        print(f"    Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Install with: pip install transformers")
        sys.exit(1)
    
    # Load and preprocess audio
    print(f"\n[Step 3] Loading and Preprocessing Audio")
    print("-" * 70)
    mel = load_audio(audio_path)
    
    # Encode audio
    print(f"\n[3] Encoding audio...")
    audio_features = model.encoder(mel, training=False)
    print(f"    Audio features shape: {audio_features.shape}")
    
    # Decode
    print(f"\n[Step 4] Decoding (Greedy)")
    print("-" * 70)
    transcription = greedy_decode(
        model,
        audio_features,
        tokenizer,
        max_length=224,
        language=language
    )
    
    # Print result
    print(f"\n" + "="*70)
    print(f"TRANSCRIPTION RESULT")
    print("="*70)
    print(f"\nAudio: {audio_path}")
    print(f"Model: {model_name}")
    print(f"Language: {language}")
    print(f"\nText: {transcription}")
    print("\n" + "="*70)
    
    return transcription


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test student model inference")
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file (.wav, .mp3, etc.)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Model size (default: base)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="vi",
        help="Language code (default: vi for Vietnamese)"
    )
    
    args = parser.parse_args()
    
    # Run inference
    test_inference(
        audio_path=args.audio,
        model_name=args.model,
        language=args.language
    )
