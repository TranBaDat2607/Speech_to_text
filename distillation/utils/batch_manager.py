"""
Batch Manager for Mini-batch Distillation
Handles dataset splitting, batch processing, and cleanup
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import json


class BatchManager:
    """Manager for mini-batch distillation workflow"""
    
    def __init__(self, total_samples: int, mini_batch_size: int):
        """
        Initialize batch manager
        
        Args:
            total_samples: Total number of samples in dataset
            mini_batch_size: Number of samples per mini-batch
        """
        self.total_samples = total_samples
        self.mini_batch_size = mini_batch_size
        self.num_batches = (total_samples + mini_batch_size - 1) // mini_batch_size
        
    def get_batch_ranges(self) -> List[Tuple[int, int]]:
        """
        Get list of (start_idx, end_idx) for each batch
        
        Returns:
            List of tuples: [(0, 1000), (1000, 2000), ...]
        """
        batch_ranges = []
        for i in range(self.num_batches):
            start_idx = i * self.mini_batch_size
            end_idx = min((i + 1) * self.mini_batch_size, self.total_samples)
            batch_ranges.append((start_idx, end_idx))
        return batch_ranges
    
    def get_batch_info(self, batch_idx: int) -> Dict:
        """
        Get information about a specific batch
        
        Args:
            batch_idx: Batch index (0-based)
            
        Returns:
            Dictionary with batch info
        """
        start_idx = batch_idx * self.mini_batch_size
        end_idx = min((batch_idx + 1) * self.mini_batch_size, self.total_samples)
        
        return {
            "batch_idx": batch_idx,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "num_samples": end_idx - start_idx,
            "total_batches": self.num_batches
        }
    
    @staticmethod
    def cleanup_logits_folder(folder_path: str, keep_metadata: bool = False):
        """
        Delete logits folder to free up disk space
        
        Args:
            folder_path: Path to logits folder
            keep_metadata: If True, keep logits_metadata.json
        """
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            print(f"Folder does not exist: {folder_path}")
            return
        
        if keep_metadata:
            # Delete only .npy files
            for file in folder_path.glob("*.npy"):
                file.unlink()
            print(f"Deleted {len(list(folder_path.glob('*.npy')))} .npy files")
        else:
            # Delete entire folder
            shutil.rmtree(folder_path)
            print(f"Deleted folder: {folder_path}")
    
    @staticmethod
    def save_progress(progress_file: str, batch_idx: int, status: str = "completed"):
        """
        Save progress to file for resume capability
        
        Args:
            progress_file: Path to progress JSON file
            batch_idx: Current batch index
            status: Status string (e.g., "completed", "training", "generating")
        """
        progress_data = {
            "last_completed_batch": batch_idx,
            "status": status
        }
        
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    @staticmethod
    def load_progress(progress_file: str) -> Dict:
        """
        Load progress from file
        
        Args:
            progress_file: Path to progress JSON file
            
        Returns:
            Progress dictionary or None if file doesn't exist
        """
        if not os.path.exists(progress_file):
            return {"last_completed_batch": -1, "status": "not_started"}
        
        with open(progress_file, 'r') as f:
            return json.load(f)
    
    def print_batch_summary(self):
        """Print summary of batching plan"""
        print(f"\n{'='*60}")
        print(f"Mini-Batch Distillation Plan")
        print(f"{'='*60}")
        print(f"  Total samples: {self.total_samples}")
        print(f"  Mini-batch size: {self.mini_batch_size}")
        print(f"  Number of batches: {self.num_batches}")
        print(f"  Estimated disk usage: ~{self.mini_batch_size * 30 / 1024:.1f} GB per batch")
        print(f"{'='*60}\n")


def split_dataset_indices(total_samples: int, mini_batch_size: int) -> List[Tuple[int, int]]:
    """
    Split dataset indices into mini-batches
    
    Args:
        total_samples: Total number of samples
        mini_batch_size: Size of each mini-batch
        
    Returns:
        List of (start_idx, end_idx) tuples
    """
    batches = []
    for i in range(0, total_samples, mini_batch_size):
        start = i
        end = min(i + mini_batch_size, total_samples)
        batches.append((start, end))
    return batches