"""
TensorFlow implementation of Whisper audio preprocessing
"""

import tensorflow as tf
import numpy as np
from audio_constants import *
from mel_filters import get_mel_filters


def load_audio(file_path, sr=SAMPLE_RATE):
    """
    Load audio file matching OpenAI Whisper exactly
    OpenAI returns: np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    
    Args:
        file_path: Path to WAV file
        sr: Sample rate (default: 16000)
        
    Returns:
        tf.Tensor: Audio waveform normalized to [-1, 1], float32 dtype
    """
    # Load audio file as int16 first (matching OpenAI processing)
    audio_binary = tf.io.read_file(file_path)
    audio, original_sr = tf.audio.decode_wav(audio_binary, desired_channels=1, desired_samples=-1)
    audio = tf.squeeze(audio, axis=-1)
    
    # Convert back to int16 range then normalize like OpenAI: / 32768.0
    # tf.audio.decode_wav already normalizes to [-1, 1], so we need to denormalize then renormalize
    # to match OpenAI's int16 -> float32 conversion
    audio_int16 = tf.cast(audio * 32767.0, tf.int16)
    audio_float32 = tf.cast(audio_int16, tf.float32) / 32768.0
    
    # Note: We assume WAV files are already at correct sample rate (16kHz)
    # OpenAI uses FFmpeg for resampling, but dataset is already processed
    
    return audio_float32


def pad_or_trim(audio, length=N_SAMPLES, axis=-1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    Matching OpenAI Whisper implementation exactly.
    
    Args:
        audio: Input audio tensor
        length: Target length (default: 480000 for 30 seconds) 
        axis: Axis to pad/trim along (default: -1)
    
    Returns:
        tf.Tensor: Audio with exact target length
    """
    # Handle axis parameter like OpenAI
    if axis == -1:
        axis = len(audio.shape) - 1
    
    current_length = tf.shape(audio)[axis]
    
    if current_length > length:
        # Trim audio (matching OpenAI logic)
        indices = tf.range(length)
        audio = tf.gather(audio, indices, axis=axis)
    elif current_length < length:
        # Pad with zeros (matching OpenAI pad_widths logic)
        pad_length = length - current_length
        
        # Create pad_widths for tf.pad
        ndim = len(audio.shape)
        pad_widths = [[0, 0]] * ndim
        pad_widths[axis] = [0, pad_length]
        
        audio = tf.pad(audio, pad_widths, mode='CONSTANT')
    
    return audio


def log_mel_spectrogram(audio, n_mels=N_MELS, padding=0):
    """
    Convert audio to log mel spectrogram matching OpenAI Whisper exactly
    
    Args:
        audio: Audio waveform tensor [N_SAMPLES]
        n_mels: Number of mel frequency bins (default: 80)
        padding: Number of zero samples to pad to the right
        
    Returns:
        tf.Tensor: Log mel spectrogram [n_mels, n_frames]
    """
    # Ensure audio has correct length (don't pad/trim here like OpenAI)
    # OpenAI does pad_or_trim outside of this function
    
    # Add padding if specified (matching OpenAI)
    if padding > 0:
        audio = tf.pad(audio, [[0, padding]], mode='CONSTANT')
    
    # Create Hann window (matching OpenAI exactly)
    window = tf.signal.hann_window(N_FFT, dtype=audio.dtype)
    
    # Compute STFT (matching OpenAI parameters exactly)
    # OpenAI: stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    stft = tf.signal.stft(
        audio,
        frame_length=N_FFT,
        frame_step=HOP_LENGTH,
        fft_length=N_FFT,
        window_fn=lambda frame_length, dtype: tf.cast(window, dtype),
        pad_end=True  # This should match PyTorch default behavior
    )
    
    # Get magnitude squared and remove last frequency bin (matching OpenAI exactly)
    # OpenAI: magnitudes = stft[..., :-1].abs() ** 2
    # TensorFlow STFT shape: [time, freq], PyTorch: [freq, time]
    
    # First compute magnitudes, then remove last frequency bin
    magnitudes = tf.abs(stft) ** 2  # Shape: [time, freq]
    
    # Remove last frequency bin: stft[..., :-1] equivalent
    magnitudes = magnitudes[:, :-1]  # Shape: [time, freq-1]
    
    # Transpose to match OpenAI format [freq-1, time] for matrix multiplication  
    magnitudes = tf.transpose(magnitudes)  # Shape: [freq-1, time]
    
    # Apply mel filter bank (matching OpenAI: filters @ magnitudes)
    mel_filters = get_mel_filters()
    mel_spec = tf.matmul(mel_filters, magnitudes)
    
    # Convert to log scale with clamping (matching OpenAI exactly)
    # OpenAI: log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    # Use tf.math.log with base conversion: log10(x) = log(x) / log(10)
    log_spec = tf.math.log(tf.maximum(mel_spec, 1e-10)) / tf.math.log(10.0)
    
    # Dynamic range normalization (matching OpenAI exactly)
    # OpenAI: log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - 8.0)
    
    # Final normalization to [0, 1] range (matching OpenAI exactly)
    # OpenAI: log_spec = (log_spec + 4.0) / 4.0
    log_spec = (log_spec + 4.0) / 4.0
    
    return log_spec


def process_single_audio(wav_path, json_path):
    """
    Process a single audio file and its transcript
    
    Args:
        wav_path: Path to WAV file
        json_path: Path to JSON transcript file
    
    Returns:
        tuple: (log_mel_spectrogram, transcript_text)
    """
    # Load and process audio - follow OpenAI order: load -> pad_or_trim -> log_mel
    audio = load_audio(wav_path)
    audio = pad_or_trim(audio)  # Ensure exact 30s length like OpenAI
    mel_spec = log_mel_spectrogram(audio)
    
    # Load transcript
    json_content = tf.io.read_file(json_path)
    
    return mel_spec, json_content


def batch_process_dataset(dataset_dir, output_dir):
    """
    Process entire dataset directory
    
    Args:
        dataset_dir: Input dataset directory path
        output_dir: Output directory for processed files
    """
    import os
    import json
    from pathlib import Path
    
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all WAV files
    wav_files = list(dataset_path.glob("*.wav"))
    
    print(f"Found {len(wav_files)} audio files to process")
    
    processed_count = 0
    
    for wav_file in wav_files:
        json_file = wav_file.with_suffix('.json')
        
        if not json_file.exists():
            print(f"Warning: No transcript found for {wav_file.name}")
            continue
            
        try:
            # Process audio - follow OpenAI Whisper order: load -> pad_or_trim -> log_mel
            audio = load_audio(str(wav_file))
            audio = pad_or_trim(audio)  # Ensure exact 30s length like OpenAI
            mel_spec = log_mel_spectrogram(audio)
            
            # Load transcript
            with open(json_file, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
            
            # Save processed mel spectrogram
            output_mel_path = output_path / f"{wav_file.stem}_mel.npy"
            mel_spec_numpy = mel_spec.numpy()
            np.save(output_mel_path, mel_spec_numpy)
            
            # Save transcript metadata
            output_json_path = output_path / f"{wav_file.stem}_processed.json"
            processed_data = {
                'original_audio_file': str(wav_file),
                'mel_spectrogram_file': str(output_mel_path),
                'mel_shape': mel_spec_numpy.shape,
                'transcript': transcript_data['text'],
                'duration': transcript_data.get('duration', 30.0),
                'sample_rate': SAMPLE_RATE,
                'n_mels': N_MELS,
                'n_frames': N_FRAMES
            }
            
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"Processed {processed_count}/{len(wav_files)} files")
                
        except Exception as e:
            print(f"Error processing {wav_file.name}: {str(e)}")
            continue
    
    print(f"Processing complete: {processed_count}/{len(wav_files)} files processed")
    print(f"Output saved to: {output_dir}")
    
    # Create summary file
    summary = {
        'total_files': len(wav_files),
        'processed_files': processed_count,
        'failed_files': len(wav_files) - processed_count,
        'output_directory': str(output_path),
        'mel_spectrogram_shape': [N_MELS, N_FRAMES],
        'audio_config': {
            'sample_rate': SAMPLE_RATE,
            'n_fft': N_FFT,
            'hop_length': HOP_LENGTH,
            'n_mels': N_MELS,
            'chunk_length': CHUNK_LENGTH
        }
    }
    
    summary_path = output_path / 'processing_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")
    
    return processed_count
