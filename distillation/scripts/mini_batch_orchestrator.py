"""
Mini-Batch Orchestrator with Separate Processes
Coordinates teacher logits generation and student training using subprocess
"""

import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class MiniBatchOrchestrator:
    """
    Orchestrator for mini-batch distillation with separate processes
    Each phase runs in isolated subprocess to avoid PyTorch-TensorFlow conflicts
    """
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        
        import yaml
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.batch_size = self.config['distillation']['mini_batch_size']
        self.auto_cleanup = self.config['distillation']['auto_cleanup_logits']
        self.epochs_per_batch = self.config['distillation'].get('epochs', 1)
        
        self.data_dir = self.config['paths']['preprocessed_dataset']
        self.checkpoint_dir = self.config['paths']['checkpoints_dir']
        self.temp_logits_dir = Path("./temp_logits")
        
        self.temp_logits_dir.mkdir(exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        self.state_file = Path("./mini_batch_state.json")
        
        self.state = {
            'batches_completed': 0,
            'total_samples_trained': 0,
            'start_time': 0,
            'batch_losses': []
        }
    
    def get_total_samples(self):
        """Get total number of samples from dataset"""
        metadata_file = Path(self.data_dir) / "metadata.json"
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        total_samples = len(metadata['samples'])
        train_ratio = self.config['data']['train_split']
        train_samples = int(total_samples * train_ratio)
        
        return train_samples
    
    def save_state(self):
        """Save orchestrator state"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def load_state(self):
        """Load orchestrator state"""
        if not self.state_file.exists():
            return False
        
        with open(self.state_file, 'r') as f:
            self.state = json.load(f)
        
        return True
    
    def generate_batch_logits(self, batch_num: int, start_idx: int, end_idx: int):
        """
        Generate teacher logits using separate subprocess
        
        Args:
            batch_num: Batch number
            start_idx: Starting sample index
            end_idx: Ending sample index
        """
        output_dir = self.temp_logits_dir / f"batch_{batch_num}"
        output_dir.mkdir(exist_ok=True)
        
        cmd = [
            sys.executable,
            "scripts/generate_batch_logits_standalone.py",
            "--config", str(self.config_path),
            "--batch_num", str(batch_num),
            "--start_idx", str(start_idx),
            "--end_idx", str(end_idx),
            "--output_dir", str(output_dir),
            "--data_dir", self.data_dir,
            "--temperature", str(self.config['distillation']['temperature'])
        ]
        
        result = subprocess.run(cmd, check=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Logits generation failed for batch {batch_num}")
        
        return str(output_dir)
    
    def train_on_batch(self, batch_num: int, start_idx: int, end_idx: int, logits_dir: str):
        """
        Train student using separate subprocess
        
        Args:
            batch_num: Batch number
            start_idx: Starting sample index
            end_idx: Ending sample index
            logits_dir: Directory containing logits
        """
        cmd = [
            sys.executable,
            "scripts/train_batch_standalone.py",
            "--config", str(self.config_path),
            "--batch_num", str(batch_num),
            "--start_idx", str(start_idx),
            "--end_idx", str(end_idx),
            "--logits_dir", logits_dir,
            "--checkpoint_dir", self.checkpoint_dir,
            "--num_epochs", str(self.epochs_per_batch)
        ]
        
        # Load checkpoint from previous batch if it exists
        if batch_num > 0:
            prev_checkpoint = Path(self.checkpoint_dir) / f"batch_{batch_num - 1}.weights.h5"
            if prev_checkpoint.exists():
                cmd.extend(["--resume_from", str(prev_checkpoint)])
                print(f"  Will resume from: {prev_checkpoint.name}")
        
        result = subprocess.run(cmd, check=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Training failed for batch {batch_num}")
    
    def cleanup_batch(self, batch_num: int, logits_dir: str):
        """
        Cleanup batch logits
        
        Args:
            batch_num: Batch number
            logits_dir: Directory to cleanup
        """
        if not self.auto_cleanup:
            return
        
        logits_path = Path(logits_dir)
        if logits_path.exists():
            try:
                shutil.rmtree(logits_path)
            except Exception as e:
                print(f"Warning: Failed to delete {logits_path}: {e}")
    
    def run(self, resume: bool = False):
        """
        Run mini-batch distillation pipeline
        
        Args:
            resume: Resume from saved state
        """
        print("\nMini-Batch Distillation")
        
        if resume and self.load_state():
            print(f"\nResuming from batch {self.state['batches_completed']}")
            start_batch = self.state['batches_completed']
        else:
            start_batch = 0
            self.state['start_time'] = time.time()
        
        total_samples = self.get_total_samples()
        
        print(f"\nDataset: {total_samples} samples")
        print(f"Batch size: {self.batch_size}")
        
        num_batches = (total_samples + self.batch_size - 1) // self.batch_size
        print(f"Total batches: {num_batches}")
        
        try:
            for batch_num in range(start_batch, num_batches):
                start_idx = batch_num * self.batch_size
                end_idx = min(start_idx + self.batch_size, total_samples)
                
                print(f"\nBatch {batch_num + 1}/{num_batches} ({(batch_num / num_batches) * 100:.0f}%): samples {start_idx}-{end_idx}")
                
                logits_dir = self.generate_batch_logits(batch_num, start_idx, end_idx)
                
                self.train_on_batch(batch_num, start_idx, end_idx, logits_dir)
                
                self.cleanup_batch(batch_num, logits_dir)
                
                self.state['batches_completed'] = batch_num + 1
                self.state['total_samples_trained'] += (end_idx - start_idx)
                
                self.save_state()
                
                elapsed = time.time() - self.state['start_time']
                print(f"Batch {batch_num + 1} done ({elapsed/60:.1f}m) - {self.state['total_samples_trained']}/{total_samples} samples")
            
            total_time = time.time() - self.state['start_time']
            print(f"\nComplete! Time: {total_time/3600:.1f}h, Samples: {self.state['total_samples_trained']}")
            print(f"Checkpoint: {self.checkpoint_dir}/batch_{num_batches-1}.weights.h5")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Resume with --resume")
            self.save_state()
        
        except Exception as e:
            print(f"\n\nError: {e}")
            import traceback
            traceback.print_exc()
            self.save_state()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Mini-batch orchestrator with separate processes")
    parser.add_argument("--config", type=str, default="config/distillation_config.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from saved state")
    
    args = parser.parse_args()
    
    orchestrator = MiniBatchOrchestrator(args.config)
    orchestrator.run(resume=args.resume)


if __name__ == "__main__":
    main()
