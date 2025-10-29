"""
Verify audio dataset - check if audio files match metadata
"""

import os
import json
import glob
import soundfile as sf
from tqdm import tqdm


def verify_audio_dataset(dataset_dir="../preprocessing_data/phoaudiobook_100h"):
    """
    Verify audio dataset by checking actual files
    """
    
    audio_dir = os.path.join(dataset_dir, "audio")
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    
    print("\n" + "="*60)
    print("VERIFY AUDIO DATASET")
    print("="*60)
    
    # Check if directories exist
    if not os.path.exists(audio_dir):
        print(f"\n❌ Audio directory not found: {audio_dir}")
        return
    
    if not os.path.exists(metadata_path):
        print(f"\n❌ Metadata file not found: {metadata_path}")
        return
    
    print(f"\nAudio dir: {audio_dir}")
    print(f"Metadata: {metadata_path}")
    
    # Load metadata
    print("\nLoading metadata...")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    metadata_samples = metadata.get("total_samples", 0)
    metadata_duration_hours = metadata.get("total_duration_hours", 0)
    
    print(f"  Metadata claims: {metadata_samples} samples, {metadata_duration_hours:.2f} hours")
    
    # Count actual audio files
    print("\nCounting actual audio files...")
    audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    actual_count = len(audio_files)
    
    print(f"  Found: {actual_count} audio files")
    
    if actual_count == 0:
        print("\n❌ NO AUDIO FILES FOUND!")
        print("   The audio directory is empty.")
        return
    
    # Calculate actual duration by reading files
    print("\nCalculating actual duration from audio files...")
    total_duration = 0.0
    errors = 0
    
    for audio_file in tqdm(audio_files, desc="Reading"):
        try:
            info = sf.info(audio_file)
            duration = info.duration
            total_duration += duration
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error reading {os.path.basename(audio_file)}: {e}")
    
    total_hours = total_duration / 3600
    avg_duration = total_duration / actual_count if actual_count > 0 else 0
    
    # Compare results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print(f"\nMetadata:")
    print(f"  Samples: {metadata_samples}")
    print(f"  Duration: {metadata_duration_hours:.2f} hours")
    
    print(f"\nActual Files:")
    print(f"  Samples: {actual_count}")
    print(f"  Duration: {total_hours:.2f} hours")
    print(f"  Average: {avg_duration:.2f} seconds")
    
    if errors > 0:
        print(f"\n⚠️  Errors: {errors} files could not be read")
    
    # Verification
    print("\n" + "="*60)
    
    if actual_count == 0:
        print("❌ FAILED: No audio files found")
    elif actual_count != metadata_samples:
        print(f"⚠️  WARNING: File count mismatch!")
        print(f"   Expected: {metadata_samples}")
        print(f"   Found: {actual_count}")
        print(f"   Missing: {metadata_samples - actual_count}")
    else:
        print("✓ File count matches metadata")
    
    if abs(total_hours - metadata_duration_hours) > 0.1:
        print(f"⚠️  WARNING: Duration mismatch!")
        print(f"   Difference: {abs(total_hours - metadata_duration_hours):.2f} hours")
    else:
        print("✓ Duration matches metadata")
    
    if total_hours >= 100.0:
        print(f"\n✓ Target reached: {total_hours:.2f} hours >= 100 hours")
    else:
        print(f"\n❌ Target NOT reached: {total_hours:.2f} hours < 100 hours")
        print(f"   Missing: {100.0 - total_hours:.2f} hours")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify audio dataset")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="../preprocessing_data/phoaudiobook_100h",
        help="Dataset directory"
    )
    
    args = parser.parse_args()
    verify_audio_dataset(args.dataset_dir)
