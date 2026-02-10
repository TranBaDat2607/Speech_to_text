"""
Data module for distillation
"""

from .distillation_dataset import DistillationDataset, collate_fn_distillation

__all__ = ['DistillationDataset', 'collate_fn_distillation']
