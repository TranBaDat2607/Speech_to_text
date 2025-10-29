"""
Utility functions for teacher model operations
"""

import os
import torch
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path


def estimate_logits_storage(
    num_samples: int,
    avg_sequence_length: int = 448,
    vocab_size: int = 51864,
    dtype: str = "float32"
) -> Dict[str, float]:
    """
    Estimate storage requirements for teacher logits
    
    Args:
        num_samples: Number of audio samples
        avg_sequence_length: Average sequence length in tokens
        vocab_size: Vocabulary size
        dtype: Data type ("float32" or "float16")
        
    Returns:
        Dictionary with storage estimates
    """
    
    bytes_per_value = 4 if dtype == "float32" else 2
    
    bytes_per_sample = avg_sequence_length * vocab_size * bytes_per_value
    
    total_bytes = num_samples * bytes_per_sample
    
    return {
        "num_samples": num_samples,
        "bytes_per_sample": bytes_per_sample,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024**2),
        "total_gb": total_bytes / (1024**3),
        "dtype": dtype
    }


def calculate_dataset_hours(
    audio_dir: str,
    extensions: List[str] = [".wav", ".mp3", ".flac"]
) -> Dict[str, float]:
    """
    Calculate total hours in audio dataset
    
    Args:
        audio_dir: Directory containing audio files
        extensions: List of audio file extensions
        
    Returns:
        Dictionary with duration statistics
    """
    
    try:
        import librosa
        import soundfile as sf
    except ImportError:
        print("Warning: librosa or soundfile not installed")
        return {"error": "Missing dependencies"}
    
    audio_files = []
    for ext in extensions:
        audio_files.extend(Path(audio_dir).rglob(f"*{ext}"))
    
    if len(audio_files) == 0:
        return {
            "num_files": 0,
            "total_hours": 0,
            "error": "No audio files found"
        }
    
    total_duration = 0
    durations = []
    
    print(f"Calculating duration for {len(audio_files)} files...")
    
    for audio_path in audio_files:
        try:
            info = sf.info(str(audio_path))
            duration = info.duration
            total_duration += duration
            durations.append(duration)
        except Exception as e:
            print(f"Warning: Could not read {audio_path.name}: {e}")
            continue
    
    return {
        "num_files": len(durations),
        "total_seconds": total_duration,
        "total_minutes": total_duration / 60,
        "total_hours": total_duration / 3600,
        "avg_duration": np.mean(durations) if durations else 0,
        "min_duration": np.min(durations) if durations else 0,
        "max_duration": np.max(durations) if durations else 0,
    }


def get_optimal_batch_size(
    gpu_memory_gb: float,
    model_size: str = "large"
) -> int:
    """
    Estimate optimal batch size for teacher inference
    
    Args:
        gpu_memory_gb: Available GPU memory in GB
        model_size: Whisper model size
        
    Returns:
        Recommended batch size
    """
    
    model_memory = {
        "large": 6.0,
        "medium": 3.0,
        "small": 1.5,
        "base": 0.5,
        "tiny": 0.3
    }
    
    model_mem = model_memory.get(model_size, 6.0)
    
    available_mem = gpu_memory_gb - model_mem
    
    mem_per_sample = 0.5
    
    batch_size = max(1, int(available_mem / mem_per_sample))
    
    return min(batch_size, 16)


def save_logits_efficient(
    logits: np.ndarray,
    save_path: str,
    compress: bool = True,
    dtype: str = "float16"
):
    """
    Save logits with optional compression
    
    Args:
        logits: Logits array [seq_len, vocab_size]
        save_path: Path to save file
        compress: Use compression
        dtype: Data type for saving
    """
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if dtype == "float16":
        logits = logits.astype(np.float16)
    
    if compress:
        np.savez_compressed(save_path, logits=logits)
    else:
        np.save(save_path, logits)


def load_logits_efficient(
    load_path: str
) -> np.ndarray:
    """
    Load logits from file
    
    Args:
        load_path: Path to logits file
        
    Returns:
        Logits array
    """
    
    if load_path.endswith('.npz'):
        data = np.load(load_path)
        return data['logits']
    else:
        return np.load(load_path)


def verify_gpu_availability() -> Dict[str, any]:
    """
    Check GPU availability and memory
    
    Returns:
        Dictionary with GPU information
    """
    
    cuda_available = torch.cuda.is_available()
    
    info = {
        "cuda_available": cuda_available,
        "device_count": 0,
        "devices": []
    }
    
    if cuda_available:
        info["device_count"] = torch.cuda.device_count()
        
        for i in range(info["device_count"]):
            device_props = torch.cuda.get_device_properties(i)
            device_info = {
                "id": i,
                "name": device_props.name,
                "total_memory_gb": device_props.total_memory / (1024**3),
                "compute_capability": f"{device_props.major}.{device_props.minor}"
            }
            
            if torch.cuda.is_available():
                torch.cuda.set_device(i)
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                device_info["allocated_memory_gb"] = allocated
                device_info["reserved_memory_gb"] = reserved
                device_info["free_memory_gb"] = device_info["total_memory_gb"] - reserved
            
            info["devices"].append(device_info)
    
    return info


def print_gpu_info():
    """Print formatted GPU information"""
    
    info = verify_gpu_availability()
    
    print(f"\n{'='*60}")
    print("GPU INFORMATION")
    print(f"{'='*60}")
    
    if not info["cuda_available"]:
        print("ERROR: CUDA not available")
        print("   Teacher inference will run on CPU (very slow)")
        print("   Consider using a GPU for distillation")
    else:
        print(f"OK: CUDA available: {info['device_count']} device(s)")
        
        for device in info["devices"]:
            print(f"\n  Device {device['id']}: {device['name']}")
            print(f"    Total Memory: {device['total_memory_gb']:.2f} GB")
            if 'free_memory_gb' in device:
                print(f"    Free Memory: {device['free_memory_gb']:.2f} GB")
            print(f"    Compute Capability: {device['compute_capability']}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print_gpu_info()
    
    print("\nStorage estimation for 100h dataset:")
    print("  Assuming ~3600 samples (100s average)")
    storage = estimate_logits_storage(
        num_samples=3600,
        avg_sequence_length=448,
        vocab_size=51864,
        dtype="float32"
    )
    print(f"  Total storage: {storage['total_gb']:.2f} GB")
    print(f"  Per sample: {storage['bytes_per_sample'] / (1024**2):.2f} MB")
    
    storage_fp16 = estimate_logits_storage(
        num_samples=3600,
        avg_sequence_length=448,
        vocab_size=51864,
        dtype="float16"
    )
    print(f"\n  With float16 compression:")
    print(f"  Total storage: {storage_fp16['total_gb']:.2f} GB")
    print(f"  Saved: {storage['total_gb'] - storage_fp16['total_gb']:.2f} GB")
