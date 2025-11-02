"""
Memory management utilities for mini-batch distillation
Handles GPU memory cleanup and logits file cleanup
"""

import os
import gc
import shutil
from pathlib import Path
from typing import Optional, List
import torch
import tensorflow as tf


def unload_teacher_from_gpu(teacher_model) -> None:
    """
    Unload teacher model from GPU and free VRAM
    
    Args:
        teacher_model: PyTorch teacher model instance
    """
    if teacher_model is not None:
        del teacher_model
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    gc.collect()


def unload_student_from_gpu(student_model) -> None:
    """
    Unload student model from GPU and free VRAM
    
    Args:
        student_model: TensorFlow student model instance
    """
    if student_model is not None:
        del student_model
    
    tf.keras.backend.clear_session()
    
    gc.collect()


def cleanup_batch_logits(
    logits_dir: str,
    start_idx: int,
    end_idx: int,
    keep_metadata: bool = False
) -> int:
    """
    Delete logits files for a specific batch range
    
    Args:
        logits_dir: Directory containing logits files
        start_idx: Starting sample index
        end_idx: Ending sample index (exclusive)
        keep_metadata: If True, keep metadata.json file
        
    Returns:
        Number of files deleted
    """
    logits_dir = Path(logits_dir)
    if not logits_dir.exists():
        return 0
    
    deleted_count = 0
    
    for idx in range(start_idx, end_idx):
        logits_file = logits_dir / f"logits_{idx:06d}.npy"
        if logits_file.exists():
            try:
                logits_file.unlink()
                deleted_count += 1
            except Exception:
                pass
    
    if not keep_metadata:
        metadata_file = logits_dir / "logits_metadata.json"
        if metadata_file.exists():
            try:
                metadata_file.unlink()
            except Exception:
                pass
    
    return deleted_count


def cleanup_batch_directory(batch_dir: str) -> bool:
    """
    Remove entire batch directory
    
    Args:
        batch_dir: Path to batch directory
        
    Returns:
        True if successful, False otherwise
    """
    batch_dir = Path(batch_dir)
    if not batch_dir.exists():
        return True
    
    try:
        shutil.rmtree(batch_dir)
        return True
    except Exception:
        return False


def get_gpu_memory_usage() -> dict:
    """
    Get current GPU memory usage
    
    Returns:
        Dictionary with memory info in MB
    """
    memory_info = {
        'pytorch_allocated': 0,
        'pytorch_reserved': 0,
        'tensorflow_allocated': 0,
        'total_available': 0
    }
    
    if torch.cuda.is_available():
        memory_info['pytorch_allocated'] = torch.cuda.memory_allocated() / 1024**2
        memory_info['pytorch_reserved'] = torch.cuda.memory_reserved() / 1024**2
    
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            memory_info['total_available'] = 8192
    except Exception:
        pass
    
    return memory_info


def estimate_batch_disk_usage(
    num_samples: int,
    avg_logits_size_mb: float = 5.0
) -> float:
    """
    Estimate disk space needed for a batch of logits
    
    Args:
        num_samples: Number of samples in batch
        avg_logits_size_mb: Average size per logits file in MB
        
    Returns:
        Estimated disk usage in MB
    """
    return num_samples * avg_logits_size_mb


def check_disk_space(path: str, required_mb: float) -> bool:
    """
    Check if enough disk space is available
    
    Args:
        path: Path to check
        required_mb: Required space in MB
        
    Returns:
        True if enough space available
    """
    try:
        stat = shutil.disk_usage(path)
        available_mb = stat.free / 1024**2
        return available_mb >= required_mb
    except Exception:
        return True


def get_batch_range(
    total_samples: int,
    batch_size: int
) -> List[tuple]:
    """
    Generate batch ranges for mini-batch processing
    
    Args:
        total_samples: Total number of samples
        batch_size: Size of each batch
        
    Returns:
        List of (start_idx, end_idx) tuples
    """
    ranges = []
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        ranges.append((start_idx, end_idx))
    return ranges
