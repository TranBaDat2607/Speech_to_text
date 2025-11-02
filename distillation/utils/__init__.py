"""Utility functions for distillation"""

from .batch_manager import BatchManager, split_dataset_indices
from .memory_utils import (
    unload_teacher_from_gpu,
    unload_student_from_gpu,
    cleanup_batch_logits,
    cleanup_batch_directory,
    get_gpu_memory_usage,
    estimate_batch_disk_usage,
    check_disk_space,
    get_batch_range
)

__all__ = [
    'BatchManager',
    'split_dataset_indices',
    'unload_teacher_from_gpu',
    'unload_student_from_gpu',
    'cleanup_batch_logits',
    'cleanup_batch_directory',
    'get_gpu_memory_usage',
    'estimate_batch_disk_usage',
    'check_disk_space',
    'get_batch_range'
]
