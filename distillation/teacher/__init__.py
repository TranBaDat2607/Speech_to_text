"""Teacher model module for knowledge distillation"""

from .load_teacher import WhisperTeacher
from .teacher_utils import (
    estimate_logits_storage,
    calculate_dataset_hours,
    get_optimal_batch_size,
    verify_gpu_availability,
    print_gpu_info
)

__all__ = [
    'WhisperTeacher',
    'estimate_logits_storage',
    'calculate_dataset_hours',
    'get_optimal_batch_size',
    'verify_gpu_availability',
    'print_gpu_info'
]