"""
Download PhoAudioBook dataset with streaming (direct mode only)
Saves audio files directly to disk without caching full dataset
"""

import os
import sys
from datasets import load_dataset
from tqdm import tqdm
import json
import soundfile as sf
import numpy as np
from dotenv import load_dotenv
import shutil
import time


def clear_huggingface_cache():
    """Clear HuggingFace dataset cache to free disk space"""
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets/thivux___phoaudiobook")
    
    if os.path.exists(cache_dir):
        print(f"\nClearing cache: {cache_dir}")
        try:
            shutil.rmtree(cache_dir)
            print("OK: Cache cleared successfully")
            # Wait a bit for filesystem to release
            time.sleep(2)
            return True
        except Exception as e:
            print(f"Warning: Could not clear cache: {e}")
            return False
    return True


def download_phowhisper_100h(
    output_dir: str = "../preprocessing_data/phoaudiobook_100h",
    target_hours: float = 100.0,
    split: str = "train",
    batch_hours: float = 25.0  # Download in batches to save disk space
):
    """
    Download PhoAudioBook dataset in batches with automatic cache cleanup
    
    Args:
        output_dir: Directory to save audio files and metadata
        target_hours: Total hours to download (100.0)
        split: Dataset split ('train', 'test')
        batch_hours: Hours per batch (25.0) - smaller = less disk usage
    """
    
    print("\n" + "="*60)
    print("BATCH DOWNLOAD MODE - AUTO CACHE CLEANUP")
    print("="*60)
    print(f"  Total target: {target_hours}h")
    print(f"  Batch size: {batch_hours}h")
    print(f"  Batches needed: {int(target_hours / batch_hours) + 1}")
    
    # Create output directories
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Target: {target_hours} hours")
    print(f"Split: {split}")
    
    # Load HuggingFace token from .env file
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("\nError: HF_TOKEN not found in .env file")
        print("\nPlease create a .env file with:")
        print("  HF_TOKEN=your_huggingface_token_here")
        print("\nOr run: huggingface-cli login")
        raise ValueError("HF_TOKEN not found")
    
    # Track overall progress
    all_samples = []
    total_duration_collected = 0.0
    start_idx = 0
    batch_num = 1
    
    # Calculate number of batches
    num_batches = int(target_hours / batch_hours) + (1 if target_hours % batch_hours > 0 else 0)
    
    print(f"\nStarting batch download...")
    print(f"  Cache location: C:\\Users\\Admin\\.cache\\huggingface\\datasets")
    print(f"  Audio output: {audio_dir}")
    print(f"  Token: from .env file\n")
    
    # Process in batches
    while total_duration_collected < target_hours * 3600:
        remaining_hours = target_hours - (total_duration_collected / 3600)
        current_batch_hours = min(batch_hours, remaining_hours)
        
        print(f"\n{'='*60}")
        print(f"BATCH {batch_num}/{num_batches}")
        print(f"  Target for this batch: {current_batch_hours:.1f}h")
        print(f"  Overall progress: {total_duration_collected/3600:.1f}h / {target_hours}h")
        print(f"{'='*60}\n")
        
        try:
            # Load dataset
            print(f"Loading dataset batch {batch_num}...")
            dataset = load_dataset(
                "thivux/phoaudiobook",
                split=split,
                token=hf_token
            )
            print(f"OK: Dataset loaded: {len(dataset)} samples")
            
        except Exception as e:
            print(f"\nError loading dataset: {e}")
            raise
    
        # Process samples for this batch
        batch_samples = []
        batch_duration = 0.0
        batch_target_duration = current_batch_hours * 3600
        
        print(f"\nProcessing batch {batch_num}...")
        
        for idx, sample in enumerate(tqdm(dataset, desc=f"Batch {batch_num}", unit="sample", position=0)):
            # Skip samples we already processed
            if idx < start_idx:
                continue
            
            try:
                # Extract audio
                if "audio" not in sample:
                    continue
                    
                audio = sample["audio"]
                audio_array = audio["array"] if isinstance(audio, dict) else audio.array
                sampling_rate = audio["sampling_rate"] if isinstance(audio, dict) else audio.sampling_rate
                
                # Get transcription
                transcription = sample.get("transcription", sample.get("sentence", sample.get("text", "")))
                
            except Exception as e:
                print(f"\nWARNING: Skip sample {idx}: {e}")
                continue
            
            # Calculate duration
            duration = len(audio_array) / sampling_rate
            
            # Check if batch target reached
            if batch_duration + duration > batch_target_duration:
                print(f"\nBatch target reached: {batch_duration / 3600:.2f}h")
                start_idx = idx
                break
            
            # Save audio file
            global_idx = len(all_samples) + len(batch_samples)
            audio_filename = f"audio_{global_idx:06d}.wav"
            audio_path = os.path.join(audio_dir, audio_filename)
            sf.write(audio_path, audio_array, sampling_rate)
            
            # Collect metadata
            sample_info = {
                "id": global_idx,
                "audio_path": audio_filename,
                "transcription": transcription,
                "duration": duration,
                "sampling_rate": sampling_rate,
                "speaker_id": sample.get("speaker_id", "unknown")
            }
            batch_samples.append(sample_info)
            batch_duration += duration
            
            # Progress update every 50 samples
            if (len(batch_samples)) % 50 == 0:
                batch_hours_collected = batch_duration / 3600
                print(f"\n  Batch progress: {batch_hours_collected:.2f}h / {current_batch_hours:.1f}h")
        
        # Update totals
        all_samples.extend(batch_samples)
        total_duration_collected += batch_duration
        
        print(f"\nOK: Batch {batch_num} completed: {batch_duration/3600:.2f}h ({len(batch_samples)} samples)")
        print(f"  Overall progress: {total_duration_collected/3600:.2f}h / {target_hours}h")
        
        # Clear cache after this batch
        print(f"\nCleaning up cache...")
        clear_huggingface_cache()
        
        # Check if we've reached overall target
        if total_duration_collected >= target_hours * 3600:
            print(f"\nOK: Overall target reached: {total_duration_collected/3600:.2f}h")
            break
        
        batch_num += 1
        
        # Small delay between batches
        time.sleep(3)
    
    # Save final metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": len(all_samples),
            "total_duration_seconds": total_duration_collected,
            "total_duration_hours": total_duration_collected / 3600,
            "target_hours": target_hours,
            "split": split,
            "batch_hours": batch_hours,
            "num_batches": batch_num,
            "samples": all_samples
        }, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETED!")
    print("="*60)
    print(f"\nStatistics:")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Total duration: {total_duration_collected / 3600:.2f} hours")
    print(f"  Average duration: {total_duration_collected / len(all_samples):.2f} seconds")
    print(f"  Batches processed: {batch_num}")
    print(f"\nFiles saved to:")
    print(f"  Audio: {audio_dir}/")
    print(f"  Metadata: {metadata_path}")
    print("\n" + "="*60)
    
    # Final cache cleanup
    print("\nFinal cache cleanup...")
    clear_huggingface_cache()
    
    return all_samples, total_duration_collected


def verify_dataset(output_dir: str = "../preprocessing_data/phoaudiobook_100h"):
    """Verify downloaded dataset"""
    
    print("\n" + "="*60)
    print("VERIFYING DATASET")
    print("="*60)
    
    # Load metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata not found at {metadata_path}")
        return False
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    # Verify audio files exist
    audio_dir = os.path.join(output_dir, "audio")
    missing_files = []
    
    print(f"\nChecking {len(metadata['samples'])} audio files...")
    
    for sample in tqdm(metadata["samples"]):
        audio_path = os.path.join(audio_dir, sample["audio_path"])
        if not os.path.exists(audio_path):
            missing_files.append(sample["audio_path"])
    
    if missing_files:
        print(f"\nMissing {len(missing_files)} files:")
        for f in missing_files[:10]:
            print(f"  - {f}")
        return False
    
    print("\nOK: All audio files present")
    print(f"\nDataset Summary:")
    print(f"  Total samples: {metadata['total_samples']}")
    print(f"  Total duration: {metadata['total_duration_hours']:.2f} hours")
    print(f"  Split: {metadata['split']}")
    
    print("\n" + "="*60)
    print("VERIFICATION PASSED!")
    print("="*60)
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download PhoAudioBook dataset (streaming mode - no cache)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../preprocessing_data/phoaudiobook_100h",
        help="Output directory for audio files"
    )
    parser.add_argument(
        "--target_hours",
        type=float,
        default=100,
        help="Total hours to download"
    )
    parser.add_argument(
        "--batch_hours",
        type=float,
        default=25.0,
        help="Hours per batch (smaller = less disk usage)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing dataset"
    )
    
    args = parser.parse_args()
    
    if args.verify:
        verify_dataset(args.output_dir)
    else:
        download_phowhisper_100h(
            output_dir=args.output_dir,
            target_hours=args.target_hours,
            split=args.split,
            batch_hours=args.batch_hours
        )
        
        # Auto verify after download
        verify_dataset(args.output_dir)
