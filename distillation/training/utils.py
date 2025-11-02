"""
Training Utility Functions
Learning rate schedules, metrics, logging helpers
"""

import tensorflow as tf
import numpy as np
from typing import List, Dict, Optional
import time
from pathlib import Path


class LearningRateScheduler:
    """
    Learning rate scheduler with warmup and cosine annealing
    """
    
    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        schedule_type: str = "cosine"
    ):
        """
        Initialize LR scheduler
        
        Args:
            base_lr: Peak learning rate after warmup
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr: Minimum learning rate
            schedule_type: "cosine" or "linear"
        """
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.schedule_type = schedule_type
    
    def get_lr(self, step: int) -> float:
        """Get learning rate for current step"""
        # Constant LR (no decay)
        if self.schedule_type == "constant":
            if step < self.warmup_steps:
                return self.base_lr * (step / self.warmup_steps)
            return self.base_lr
        
        # Standard warmup
        if step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (step / self.warmup_steps)
        
        # After warmup
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        
        if self.schedule_type == "cosine":
            # Cosine annealing
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
                1 + np.cos(np.pi * progress)
            )
        else:
            # Linear decay
            lr = self.base_lr - (self.base_lr - self.min_lr) * progress
        
        return float(lr)


class MetricsTracker:
    """
    Track training metrics over time
    """
    
    def __init__(self):
        self.metrics = {}
        self.history = {}
    
    def update(self, metric_name: str, value: float):
        """Update metric with new value"""
        self.metrics[metric_name] = value
        
        if metric_name not in self.history:
            self.history[metric_name] = []
        self.history[metric_name].append(value)
    
    def get_average(self, metric_name: str, last_n: Optional[int] = None) -> float:
        """Get average of last N values"""
        if metric_name not in self.history:
            return 0.0
        
        values = self.history[metric_name]
        if last_n:
            values = values[-last_n:]
        
        return np.mean(values) if values else 0.0
    
    def reset(self):
        """Reset current metrics"""
        self.metrics = {}
    
    def get_current(self) -> Dict[str, float]:
        """Get current metrics"""
        return self.metrics.copy()


class Timer:
    """Simple timer for tracking training time"""
    
    def __init__(self):
        self.start_time = None
        self.elapsed = 0.0
    
    def start(self):
        """Start timer"""
        self.start_time = time.time()
    
    def stop(self) -> float:
        """Stop timer and return elapsed time"""
        if self.start_time is None:
            return 0.0
        
        self.elapsed = time.time() - self.start_time
        self.start_time = None
        return self.elapsed
    
    def get_elapsed(self) -> float:
        """Get elapsed time without stopping"""
        if self.start_time is None:
            return self.elapsed
        return time.time() - self.start_time


class CheckpointManager:
    """
    Manage model checkpoints
    Keep only the best N checkpoints
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_to_keep: int = 5,
        save_best_only: bool = False,
        metric_name: str = "val_loss",
        mode: str = "min"
    ):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_to_keep: Maximum number of checkpoints to keep
            save_best_only: If True, only save when metric improves
            metric_name: Metric to track for best checkpoint
            mode: "min" or "max" for metric comparison
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_to_keep = max_to_keep
        self.save_best_only = save_best_only
        self.metric_name = metric_name
        self.mode = mode
        
        self.checkpoints = []  # List of (step, metric_value, path)
        self.best_metric = float('inf') if mode == "min" else float('-inf')
    
    def should_save(self, metric_value: float) -> bool:
        """Check if should save checkpoint based on metric"""
        if not self.save_best_only:
            return True
        
        if self.mode == "min":
            return metric_value < self.best_metric
        else:
            return metric_value > self.best_metric
    
    def save_checkpoint(
        self,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        step: int,
        epoch: int,
        metric_value: float,
        additional_info: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Save checkpoint if conditions are met
        
        Returns:
            Path to saved checkpoint or None
        """
        if not self.should_save(metric_value):
            return None
        
        # Update best metric
        if self.mode == "min":
            if metric_value < self.best_metric:
                self.best_metric = metric_value
        else:
            if metric_value > self.best_metric:
                self.best_metric = metric_value
        
        # Create checkpoint path
        checkpoint_name = f"checkpoint_epoch_{epoch}_step_{step}.weights.h5"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save model weights
        model.save_weights(str(checkpoint_path))
        
        # Save optimizer state and metadata
        metadata = {
            'step': step,
            'epoch': epoch,
            'metric_name': self.metric_name,
            'metric_value': metric_value,
            'optimizer_config': optimizer.get_config(),
        }
        
        if additional_info:
            metadata.update(additional_info)
        
        metadata_path = checkpoint_path.parent / f"{checkpoint_path.stem}_meta.npy"
        np.save(metadata_path, metadata, allow_pickle=True)
        
        # Track checkpoint
        self.checkpoints.append((step, metric_value, str(checkpoint_path)))
        
        # Remove old checkpoints
        self._cleanup_old_checkpoints()
        
        print(f"  Checkpoint saved: {checkpoint_name}")
        print(f"  {self.metric_name}: {metric_value:.4f}")
        
        return str(checkpoint_path)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keep only max_to_keep"""
        if len(self.checkpoints) <= self.max_to_keep:
            return
        
        # Sort by metric value
        if self.mode == "min":
            self.checkpoints.sort(key=lambda x: x[1])  # Lower is better
        else:
            self.checkpoints.sort(key=lambda x: -x[1])  # Higher is better
        
        # Remove worst checkpoints
        checkpoints_to_remove = self.checkpoints[self.max_to_keep:]
        self.checkpoints = self.checkpoints[:self.max_to_keep]
        
        for _, _, ckpt_path in checkpoints_to_remove:
            try:
                # Remove weights file
                Path(ckpt_path).unlink(missing_ok=True)
                # Remove metadata
                meta_path = Path(ckpt_path).parent / f"{Path(ckpt_path).stem}_meta.npy"
                meta_path.unlink(missing_ok=True)
            except Exception as e:
                print(f"Warning: Could not remove checkpoint {ckpt_path}: {e}")
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint"""
        if not self.checkpoints:
            return None
        
        if self.mode == "min":
            best_ckpt = min(self.checkpoints, key=lambda x: x[1])
        else:
            best_ckpt = max(self.checkpoints, key=lambda x: x[1])
        
        return best_ckpt[2]


def format_time(seconds: float) -> str:
    """Format seconds to human-readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def compute_gradient_norm(gradients: List[tf.Tensor]) -> float:
    """Compute global gradient norm"""
    total_norm = 0.0
    for grad in gradients:
        if grad is not None:
            total_norm += tf.reduce_sum(tf.square(grad))
    return tf.sqrt(total_norm).numpy()
