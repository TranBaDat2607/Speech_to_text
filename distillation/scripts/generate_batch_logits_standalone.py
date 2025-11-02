"""
Standalone script to generate teacher logits for a batch
PyTorch only - no TensorFlow imports
"""

import os
import sys
import argparse
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from teacher.load_teacher_pytorch import WhisperTeacherPyTorch
from data.dataset_loader import AudioDataset
from scripts.generate_teacher_logits import generate_teacher_logits_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--batch_num", type=int, required=True)
    parser.add_argument("--start_idx", type=int, required=True)
    parser.add_argument("--end_idx", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=3.0)
    
    args = parser.parse_args()
    
    print(f"\nBatch {args.batch_num}: samples {args.start_idx}-{args.end_idx}")
    
    import yaml
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    teacher_model_name = config_dict['teacher']['model_name']
    
    teacher = WhisperTeacherPyTorch(
        model_name=teacher_model_name,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    dataset = AudioDataset(
        data_dir=args.data_dir,
        config=config_dict,
        split='train',
        max_samples=None
    )
    
    generate_teacher_logits_batch(
        teacher_model=teacher,
        dataset=dataset,
        output_dir=args.output_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        temperature=args.temperature,
        batch_name=f"batch_{args.batch_num}"
    )
    


if __name__ == "__main__":
    main()
