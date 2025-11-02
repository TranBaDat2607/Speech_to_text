"""
Metrics logging for training visualization
Save metrics to JSON for later plotting
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class MetricsLogger:
    """
    Log training metrics to file for visualization
    Supports both JSON and CSV formats
    """
    
    def __init__(self, log_dir: str = "./logs", experiment_name: str = None):
        """
        Initialize metrics logger
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment (default: timestamp)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.experiment_name = experiment_name
        self.metrics_history = []
        
        # File paths
        self.json_path = self.log_dir / f"{experiment_name}_metrics.json"
        self.csv_path = self.log_dir / f"{experiment_name}_metrics.csv"
        
        print(f"Metrics logger initialized:")
        print(f"  JSON: {self.json_path}")
        print(f"  CSV:  {self.csv_path}")
    
    def log_step(self, step: int, metrics: Dict[str, float], phase: str = "train"):
        """
        Log metrics for a single step
        
        Args:
            step: Global step number
            metrics: Dictionary of metric_name -> value
            phase: "train" or "val"
        """
        entry = {
            "step": step,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        
        self.metrics_history.append(entry)
    
    def log_batch(self, batch_num: int, metrics: Dict[str, float]):
        """
        Log metrics for a batch
        
        Args:
            batch_num: Batch number
            metrics: Dictionary of metrics
        """
        entry = {
            "batch": batch_num,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        
        self.metrics_history.append(entry)
        
        # Save after each batch (in case training crashes)
        self.save()
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """
        Log metrics for an epoch
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        entry = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        
        self.metrics_history.append(entry)
        self.save()
    
    def save(self):
        """Save metrics to both JSON and CSV"""
        self._save_json()
        self._save_csv()
    
    def _save_json(self):
        """Save to JSON format"""
        with open(self.json_path, 'w') as f:
            json.dump({
                'experiment_name': self.experiment_name,
                'metrics': self.metrics_history
            }, f, indent=2)
    
    def _save_csv(self):
        """Save to CSV format"""
        if not self.metrics_history:
            return
        
        # Get all unique keys
        all_keys = set()
        for entry in self.metrics_history:
            all_keys.update(entry.keys())
        
        fieldnames = sorted(all_keys)
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metrics_history)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get full metrics history"""
        return self.metrics_history
    
    def get_metric_values(self, metric_name: str) -> List[float]:
        """
        Get all values for a specific metric
        
        Args:
            metric_name: Name of metric
            
        Returns:
            List of values
        """
        return [
            entry[metric_name] 
            for entry in self.metrics_history 
            if metric_name in entry
        ]
    
    def print_summary(self):
        """Print summary of logged metrics"""
        if not self.metrics_history:
            print("No metrics logged yet")
            return
        
        print(f"\n{'='*60}")
        print(f"Metrics Summary: {self.experiment_name}")
        print(f"{'='*60}")
        print(f"Total entries: {len(self.metrics_history)}")
        
        # Get metric names (excluding metadata)
        metric_names = set()
        for entry in self.metrics_history:
            metric_names.update(
                k for k in entry.keys() 
                if k not in ['step', 'epoch', 'batch', 'phase', 'timestamp']
            )
        
        print(f"Metrics tracked: {', '.join(sorted(metric_names))}")
        print(f"Saved to:")
        print(f"  - {self.json_path}")
        print(f"  - {self.csv_path}")


def load_metrics(log_file: str) -> Dict[str, Any]:
    """
    Load metrics from JSON log file
    
    Args:
        log_file: Path to JSON log file
        
    Returns:
        Dictionary with experiment info and metrics
    """
    with open(log_file, 'r') as f:
        return json.load(f)
