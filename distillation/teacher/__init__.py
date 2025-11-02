"""Teacher model module for knowledge distillation"""

from .load_teacher_pytorch import WhisperTeacherPyTorch
from .teacher_utils import (
    estimate_logits_storage,
    calculate_dataset_hours,
    get_optimal_batch_size,
    verify_gpu_availability,
    print_gpu_info
)

__all__ = [
    'WhisperTeacherPyTorch',
    'estimate_logits_storage',
    'calculate_dataset_hours',
    'get_optimal_batch_size',
    'verify_gpu_availability',
    'print_gpu_info'
]