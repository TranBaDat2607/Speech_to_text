"""
Standalone script to train student on a batch
TensorFlow only - no PyTorch imports
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import load_config
from training.trainer import DistillationTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--batch_num", type=int, required=True)
    parser.add_argument("--start_idx", type=int, required=True)
    parser.add_argument("--end_idx", type=int, required=True)
    parser.add_argument("--logits_dir", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint from previous batch")
    
    args = parser.parse_args()
    
    print(f"\nTraining batch {args.batch_num}: samples {args.start_idx}-{args.end_idx}")
    
    config = load_config(args.config)
    
    # Determine whether to load OpenAI pretrained weights
    # Only load OpenAI weights for batch 0 (first batch)
    # Subsequent batches will load from checkpoint
    load_openai = (args.batch_num == 0 and args.resume_from is None)
    
    if load_openai:
        print("\n[INFO] Batch 0: Will initialize from OpenAI pretrained weights")
    elif args.resume_from and Path(args.resume_from).exists():
        print(f"\n[INFO] Batch {args.batch_num}: Will resume from checkpoint: {Path(args.resume_from).name}")
    else:
        print(f"\n[WARN] Batch {args.batch_num}: No checkpoint found, will use OpenAI pretrained")
        load_openai = True
    
    # Create trainer with appropriate weight loading strategy
    trainer = DistillationTrainer(config, load_openai_weights=load_openai)
    
    # Load checkpoint from previous batch if exists
    if args.resume_from and Path(args.resume_from).exists():
        print(f"Loading checkpoint from previous batch: {args.resume_from}")
        try:
            trainer.model.load_weights(args.resume_from)
            print("[INFO] Checkpoint loaded successfully")
        except Exception as e:
            print(f"[WARN] Warning: Could not load checkpoint: {e}")
            print("       Falling back to current model weights")
    
    # Train on batch
    metrics = trainer.train_on_batch_range(
        logits_dir=args.logits_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        num_epochs=args.num_epochs
    )
    
    # Save weights
    checkpoint_path = Path(args.checkpoint_dir) / f"batch_{args.batch_num}.weights.h5"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.model.save_weights(str(checkpoint_path))
    
    # Save metadata
    metadata_path = checkpoint_path.parent / f"batch_{args.batch_num}_metadata.json"
    metadata = {
        'batch_num': args.batch_num,
        'start_idx': args.start_idx,
        'end_idx': args.end_idx,
        'num_samples': args.end_idx - args.start_idx,
        'num_epochs': args.num_epochs,
        'loss': metrics.get('loss', 0.0),
        'kl_loss': metrics.get('kl_loss', 0.0),
        'ce_loss': metrics.get('ce_loss', 0.0),
        'checkpoint_file': checkpoint_path.name,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Batch {args.batch_num} done. Loss: {metrics.get('loss', 0.0):.4f}")
    print(f"  Checkpoint: {checkpoint_path.name}")
    print(f"  Metadata: {metadata_path.name}")


if __name__ == "__main__":
    main()
