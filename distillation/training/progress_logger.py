"""
Simple tqdm-style Progress Bar for Training
"""

import sys
import time
from typing import Dict


class SimpleProgbar:
    """
    Simplified progress bar using tqdm-like display
    """
    
    def __init__(self, total: int, desc: str = "Training", unit: str = "step"):
        """
        Initialize simple progress bar
        
        Args:
            total: Total number of iterations
            desc: Description prefix
            unit: Unit name (e.g., 'step', 'batch')
        """
        self.total = total
        self.desc = desc
        self.unit = unit
        self.current = 0
        self.start_time = time.time()
        self.metrics = {}
    
    def update(self, n: int = 1, **metrics):
        """Update progress by n steps and set metrics"""
        self.current = min(self.current + n, self.total)
        self.metrics.update(metrics)
        self._display()
    
    def set_postfix(self, **kwargs):
        """Set postfix metrics"""
        self.metrics.update(kwargs)
    
    def _display(self):
        """Display current progress"""
        elapsed = time.time() - self.start_time
        progress = self.current / self.total if self.total > 0 else 0
        
        # Progress bar (ASCII for single-line update)
        bar_len = 30
        filled = int(bar_len * progress)
        bar = '=' * filled + '.' * (bar_len - filled)
        
        # Rate
        rate = self.current / elapsed if elapsed > 0 else 0
        
        # ETA
        if self.current > 0 and self.current < self.total:
            eta = (self.total - self.current) / rate if rate > 0 else 0
            eta_str = f" ETA:{int(eta)}s"
        else:
            eta_str = ""
        
        # Metrics (compact format)
        metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                 for k, v in self.metrics.items() if k != 'applied_gradients'])
        if metrics_str:
            metrics_str = " | " + metrics_str
        
        # Build line with padding to clear previous content
        line_content = f"{self.desc}: {self.current}/{self.total} [{bar}] {int(progress*100)}% {int(elapsed)}s {rate:.1f}{self.unit}/s{eta_str}{metrics_str}"
        
        # Add padding to clear any leftover characters from previous line
        terminal_width = 120  # Assume standard width
        padded_line = line_content.ljust(terminal_width)
        
        # Write with carriage return to overwrite same line
        sys.stdout.write(f"\r{padded_line}")
        sys.stdout.flush()
        
        if self.current >= self.total:
            print()  # New line when complete
    
    def close(self):
        """Close progress bar"""
        if self.current < self.total:
            self.current = self.total
            self._display()
