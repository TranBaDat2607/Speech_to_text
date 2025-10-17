"""
Test script to verify audio preprocessing pipeline
"""

import os
import tensorflow as tf
import numpy as np
from pathlib import Path

# Force TensorFlow to use CPU only to avoid CUDA issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

from audio_processor import load_audio, log_mel_spectrogram, process_single_audio, pad_or_trim
from audio_constants import *


def test_audio_loading():
    """Test audio loading functionality"""
    print("Testing audio loading...")
    
    # Find a sample WAV file
    base_dir = Path(__file__).parent.parent
    dataset_dir = base_dir / "crawl_data" / "datasets" / "sachnoivietnam15" / "dataset"
    
    wav_files = list(dataset_dir.glob("*.wav"))
    if not wav_files:
        print("No WAV files found for testing")
        return False
    
    sample_wav = wav_files[0]
    print(f"Testing with: {sample_wav.name}")
    
    try:
        # Load audio
        audio = load_audio(str(sample_wav))
        print(f"Audio shape: {audio.shape}")
        print(f"Audio dtype: {audio.dtype}")
        print(f"Audio range: [{tf.reduce_min(audio):.3f}, {tf.reduce_max(audio):.3f}]")
        
        # Expected: 30 seconds at 16kHz = 480000 samples
        expected_samples = N_SAMPLES
        if audio.shape[0] == expected_samples:
            print(f"Audio length correct: {expected_samples} samples (30s at 16kHz)")
        else:
            print(f"Audio length: {audio.shape[0]} samples (expected {expected_samples})")
        
        return True
        
    except Exception as e:
        print(f"Audio loading failed: {str(e)}")
        return False


def test_mel_spectrogram():
    """Test mel spectrogram conversion"""
    print("\nTesting mel spectrogram conversion...")
    
    # Create dummy audio signal (sine wave)
    t = tf.linspace(0.0, CHUNK_LENGTH, N_SAMPLES)
    frequency = 440.0  # A4 note
    dummy_audio = 0.1 * tf.sin(2 * np.pi * frequency * t)
    
    try:
        # Follow OpenAI order: pad_or_trim first, then mel spectrogram
        dummy_audio = pad_or_trim(dummy_audio)
        mel_spec = log_mel_spectrogram(dummy_audio)
        print(f"Mel spectrogram shape: {mel_spec.shape}")
        print(f"Expected shape: ({N_MELS}, {N_FRAMES})")
        print(f"Mel spectrogram dtype: {mel_spec.dtype}")
        print(f"Mel spectrogram range: [{tf.reduce_min(mel_spec):.3f}, {tf.reduce_max(mel_spec):.3f}]")
        
        # Verify shape
        if mel_spec.shape == (N_MELS, N_FRAMES):
            print("Mel spectrogram shape correct!")
            return True
        else:
            print(f"Shape mismatch! Got {mel_spec.shape}, expected ({N_MELS}, {N_FRAMES})")
            return False
            
    except Exception as e:
        print(f"Mel spectrogram conversion failed: {str(e)}")
        return False


def test_full_pipeline():
    """Test complete preprocessing pipeline"""
    print("\nTesting full preprocessing pipeline...")
    
    # Find a sample file pair
    base_dir = Path(__file__).parent.parent
    dataset_dir = base_dir / "crawl_data" / "datasets" / "sachnoivietnam15" / "dataset"
    
    wav_files = list(dataset_dir.glob("*.wav"))
    if not wav_files:
        print("No WAV files found for testing")
        return False
    
    # Find matching JSON file
    for wav_file in wav_files:
        json_file = wav_file.with_suffix('.json')
        if json_file.exists():
            print(f"Testing with: {wav_file.name} + {json_file.name}")
            
            try:
                mel_spec, json_content = process_single_audio(str(wav_file), str(json_file))
                
                print(f"Mel spectrogram shape: {mel_spec.shape}")
                print(f"Expected shape: (80, 3000)")
                
                # Verify shape
                if mel_spec.shape == (80, 3000):
                    print("Mel spectrogram shape correct!")
                else:
                    print(f"Shape mismatch! Got {mel_spec.shape}, expected (80, 3000)")
                    return False
                
                # Parse JSON to verify
                import json
                json_str = json_content.numpy().decode('utf-8')
                transcript_data = json.loads(json_str)
                print(f"Sample transcript: {transcript_data['text'][:100]}...")
                
                return True
                
            except Exception as e:
                print(f"Full pipeline test failed: {str(e)}")
                return False
    
    print("No matching WAV+JSON pairs found")
    return False


def main():
    print('TensorFlow version:', tf.__version__)
    print('GPU devices:', tf.config.list_physical_devices('GPU'))
    """Run all tests"""
    print("=" * 60)
    print("Whisper Audio Preprocessing Test Suite")
    print("=" * 60)
    
    print(f"Configuration:")
    print(f"- Sample rate: {SAMPLE_RATE} Hz")
    print(f"- Chunk length: {CHUNK_LENGTH} seconds")
    print(f"- N_FFT: {N_FFT}")
    print(f"- Hop length: {HOP_LENGTH}")
    print(f"- Mel bins: {N_MELS}")
    print(f"- Expected samples: {N_SAMPLES}")
    print(f"- Expected frames: {N_FRAMES}")
    
    tests = [
        ("Audio Loading", test_audio_loading),
        ("Mel Spectrogram", test_mel_spectrogram),
        ("Full Pipeline", test_full_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        print("\n" + "-" * 40)
        success = test_func()
        results.append((test_name, success))
        print(f"{test_name}: {'PASS' if success else 'FAIL'}")
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {test_name:20s}: {status}")
    
    all_passed = all(result[1] for result in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    if all_passed:
        print("\nPreprocessing pipeline is ready!")
        print("You can now run: python preprocess_dataset.py")
    else:
        print("\nPlease check the failed tests before proceeding.")


if __name__ == "__main__":
    main()
