"""
Train Student Model with Knowledge Distillation
Entry point for training pipeline
"""

import os
import sys
from pathlib import Path
import argparse
import tensorflow as tf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import load_config
from training.trainer import DistillationTrainer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train Whisper student model with knowledge distillation"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/distillation_config.yaml",
        help="Path to config YAML file"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate"
    )
    
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Override checkpoint directory"
    )
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    print("\nWhisper Distillation Training")
    print(f"Config: {args.config}")
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.checkpoint_dir is not None:
        config.paths.checkpoints_dir = args.checkpoint_dir
    
    # Print configuration summary
    config.print_summary()
    
    # Initialize trainer
    trainer = DistillationTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from: {args.resume}")
        # TODO: Implement checkpoint loading
        # trainer.load_checkpoint(args.resume)
    
    # Start training
    try:
        trainer.train()
        
        print("\nTraining complete!")
        best_checkpoint = trainer.checkpoint_manager.get_best_checkpoint()
        if best_checkpoint:
            print(f"Best: {best_checkpoint}")
        print(f"Logs: {config.paths.logs_dir}")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted. Resume with --resume <checkpoint_path>")
    
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
