"""
Download first 100 hours of PhoWhisper dataset for distillation
"""

import os
import sys
from datasets import load_dataset
from tqdm import tqdm
import json
import soundfile as sf
import numpy as np


def download_phowhisper_100h(
    output_dir: str = "../preprocessing_data/phoaudiobook_100h",
    target_hours: float = 100.0,
    split: str = "train"
):
    """
    Download first 100 hours from PhoAudioBook dataset
    
    Args:
        output_dir: Directory to save audio files and metadata
        target_hours: Target hours to download (100.0)
        split: Dataset split ('train', 'test')
    """
    
    print("\n" + "="*60)
    print("DOWNLOADING PHOAUDIOBOOK - FIRST 100 HOURS")
    print("="*60)
    
    # Create output directories
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Target: {target_hours} hours")
    print(f"Split: {split}")
    
    # Load PhoAudioBook dataset (requires authentication)
    print(f"\nLoading PhoAudioBook dataset from HuggingFace...")
    print(f"  Note: This dataset requires authentication")
    print(f"  Make sure you've run: huggingface-cli login")
    
    try:
        # Use non-streaming mode to avoid torchcodec requirement
        dataset = load_dataset(
            "thivux/phoaudiobook",
            split=split
        )
        print(f"✓ Dataset loaded successfully")
        print(f"  Total samples in {split}: {len(dataset)}")
    except Exception as e:
        print(f"\nError: {e}")
        print(f"\nPossible issues:")
        print(f"  1. Not logged in - Run: huggingface-cli login")
        print(f"  2. Access not approved yet - Check: https://huggingface.co/datasets/thivux/phoaudiobook")
        print(f"  3. Invalid token")
        raise
    
    # Collect samples until we reach 100 hours
    samples = []
    total_duration = 0.0
    target_duration = target_hours * 3600  # Convert to seconds
    
    print(f"\nCollecting samples (target: {target_duration:.0f} seconds = {target_hours}h)...")
    
    for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        # Get audio info
        try:
            audio = sample["audio"]
            audio_array = audio["array"]
            sampling_rate = audio["sampling_rate"]
            # Try different transcription field names
            transcription = sample.get("transcription", sample.get("sentence", sample.get("text", "")))
        except KeyError as e:
            print(f"\nWarning: Error processing sample {idx}: {e}")
            continue
        
        # Calculate duration
        duration = len(audio_array) / sampling_rate
        
        # Check if we've reached target
        if total_duration + duration > target_duration:
            print(f"\n✓ Reached target: {total_duration / 3600:.2f}h")
            break
        
        # Save audio file
        audio_filename = f"audio_{idx:06d}.wav"
        audio_path = os.path.join(audio_dir, audio_filename)
        
        sf.write(audio_path, audio_array, sampling_rate)
        
        # Collect metadata
        sample_info = {
            "id": idx,
            "audio_path": audio_filename,
            "transcription": transcription,
            "duration": duration,
            "sampling_rate": sampling_rate,
            "speaker_id": sample.get("speaker_id", "unknown")
        }
        samples.append(sample_info)
        
        total_duration += duration
        
        # Progress update every 500 samples
        if (idx + 1) % 500 == 0:
            hours_collected = total_duration / 3600
            print(f"\n  Collected: {hours_collected:.2f}h ({len(samples)} samples)")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": len(samples),
            "total_duration_seconds": total_duration,
            "total_duration_hours": total_duration / 3600,
            "target_hours": target_hours,
            "split": split,
            "samples": samples
        }, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETED!")
    print("="*60)
    print(f"\nStatistics:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Total duration: {total_duration / 3600:.2f} hours")
    print(f"  Average duration: {total_duration / len(samples):.2f} seconds")
    print(f"\nFiles saved to:")
    print(f"  Audio: {audio_dir}/")
    print(f"  Metadata: {metadata_path}")
    print("\n" + "="*60)
    
    return samples, total_duration


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
        print(f"\n❌ Missing {len(missing_files)} files:")
        for f in missing_files[:10]:
            print(f"  - {f}")
        return False
    
    print("\n✓ All audio files present")
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
    
    parser = argparse.ArgumentParser(description="Download PhoWhisper dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../preprocessing_data/phoaudiobook_100h",
        help="Output directory"
    )
    parser.add_argument(
        "--target_hours",
        type=float,
        default=100.0,
        help="Target hours to download"
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
            target_hours=args.target_hours
        )
        
        # Auto verify after download
        verify_dataset(args.output_dir)
