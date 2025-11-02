"""
Generate teacher logits for knowledge distillation
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import time

sys.path.append(str(Path(__file__).parent.parent))

from teacher.load_teacher_pytorch import WhisperTeacherPyTorch
from data.dataset_loader import load_config, AudioDataset
from utils.batch_manager import BatchManager


def generate_teacher_logits_batch(
    teacher_model,
    dataset,
    output_dir: str,
    start_idx: int,
    end_idx: int,
    temperature: float = 3.0,
    batch_name: str = "batch"
):
    """
    Generate and save teacher logits for a specific batch of samples
    
    Args:
        teacher_model: Teacher model instance
        dataset: Audio dataset
        output_dir: Directory to save logits
        start_idx: Starting index (inclusive)
        end_idx: Ending index (exclusive)
        temperature: Temperature for soft labels
        batch_name: Name for this batch (for logging)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    num_samples = end_idx - start_idx
    
    print(f"\nGenerating {batch_name}: samples {start_idx}-{end_idx} ({num_samples} samples)")
    
    metadata = []
    successful = 0
    failed = 0
    
    start_time = time.time()
    
    for idx in tqdm(range(start_idx, end_idx), desc=f"Generating {batch_name}"):
        try:
            sample = dataset[idx]
            audio = sample['audio']
            text = sample['text']
            audio_path = sample['audio_path']
            
            inputs = teacher_model.processor(
                text=text,
                return_tensors="pt",
                add_special_tokens=True
            )
            decoder_input_ids = inputs.input_ids
            
            logits = teacher_model.generate_logits_from_audio(
                audio_array=audio,
                decoder_input_ids=decoder_input_ids,
                temperature=temperature,
                sampling_rate=16000
            )
            
            logits_np = logits.cpu().numpy()
            
            # Keep original index in filename
            logits_filename = f"logits_{idx:06d}.npy"
            logits_path = os.path.join(output_dir, logits_filename)
            np.save(logits_path, logits_np)
            
            metadata.append({
                "id": idx,
                "logits_file": logits_filename,
                "audio_path": audio_path,
                "text": text,
                "logits_shape": list(logits_np.shape),
                "duration": sample['duration']
            })
            
            successful += 1
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            failed += 1
            continue
    
    elapsed = time.time() - start_time
    
    metadata_path = os.path.join(output_dir, "logits_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump({
            "batch_name": batch_name,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "total_samples": successful,
            "failed_samples": failed,
            "temperature": temperature,
            "generation_time_seconds": elapsed,
            "samples": metadata
        }, f, indent=2, ensure_ascii=False)
    
    print(f"{batch_name} complete: {successful}/{num_samples} samples ({elapsed/60:.1f}m, {successful/(elapsed/60):.0f} samples/min)")
    
    return successful, failed


def generate_teacher_logits(
    teacher_model,
    dataset,
    output_dir: str,
    temperature: float = 3.0,
    batch_size: int = 1,
    max_samples: int = None
):
    """
    Generate and save teacher logits for entire dataset (legacy function)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    print(f"\nGenerating logits for {num_samples} samples...")
    
    metadata = []
    successful = 0
    failed = 0
    
    start_time = time.time()
    
    for idx in tqdm(range(num_samples), desc="Generating logits"):
        try:
            sample = dataset[idx]
            audio = sample['audio']
            text = sample['text']
            audio_path = sample['audio_path']
            
            inputs = teacher_model.processor(
                text=text,
                return_tensors="pt",
                add_special_tokens=True
            )
            decoder_input_ids = inputs.input_ids
            
            logits = teacher_model.generate_logits_from_audio(
                audio_array=audio,
                decoder_input_ids=decoder_input_ids,
                temperature=temperature,
                sampling_rate=16000
            )
            
            logits_np = logits.cpu().numpy()
            
            logits_filename = f"logits_{idx:06d}.npy"
            logits_path = os.path.join(output_dir, logits_filename)
            np.save(logits_path, logits_np)
            
            metadata.append({
                "id": idx,
                "logits_file": logits_filename,
                "audio_path": audio_path,
                "text": text,
                "logits_shape": list(logits_np.shape),
                "duration": sample['duration']
            })
            
            successful += 1
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            failed += 1
            continue
    
    elapsed = time.time() - start_time
    
    metadata_path = os.path.join(output_dir, "logits_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total_samples": successful,
            "failed_samples": failed,
            "temperature": temperature,
            "generation_time_seconds": elapsed,
            "samples": metadata
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Complete: {successful}/{num_samples} samples ({elapsed/60:.1f}m, {successful/(elapsed/60):.0f} samples/min)")


def main():
    print("\nTeacher Logits Generation")
    
    config = load_config("config/distillation_config.yaml")
    
    teacher = WhisperTeacherPyTorch(
        model_name="vinai/PhoWhisper-large",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    train_dataset = AudioDataset(
        data_dir=config['paths']['preprocessed_dataset'],
        config=config,
        split='train',
        max_samples=config['data'].get('max_samples', None)
    )
    
    val_dataset = AudioDataset(
        data_dir=config['paths']['preprocessed_dataset'],
        config=config,
        split='validation',
        max_samples=config['data'].get('max_samples', None)
    )
    
    print("\nGenerating train logits...")
    train_output_dir = os.path.join(config['paths']['teacher_logits_dir'], 'train')
    generate_teacher_logits(
        teacher_model=teacher,
        dataset=train_dataset,
        output_dir=train_output_dir,
        temperature=config['teacher']['temperature'],
        batch_size=1
    )
    
    print("\nGenerating validation logits...")
    val_output_dir = os.path.join(config['paths']['teacher_logits_dir'], 'val')
    generate_teacher_logits(
        teacher_model=teacher,
        dataset=val_dataset,
        output_dir=val_output_dir,
        temperature=config['teacher']['temperature'],
        batch_size=1
    )
    
    print("\nComplete!")
    print(f"Train: {train_output_dir}")
    print(f"Val: {val_output_dir}")


if __name__ == "__main__":
    main()
