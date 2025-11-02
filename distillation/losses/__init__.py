"""Loss functions for knowledge distillation"""

from .distillation_loss import DistillationLoss, compute_distillation_loss

__all__ = ['DistillationLoss', 'compute_distillation_loss']
