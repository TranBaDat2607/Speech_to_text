"""
Process cached parquet files using datasets library
Extract audio and delete cache files one by one
"""

import os
import glob
import json
import soundfile as sf
from datasets import Dataset
from tqdm import tqdm
import shutil


def find_cache_directory():
    """Find HuggingFace cache directory"""
    
    possible_dirs = [
        os.path.expanduser("~/.cache/huggingface/hub/datasets--thivux--phoaudiobook/snapshots"),
        os.path.expanduser("~/.cache/huggingface/datasets/thivux___phoaudiobook"),
    ]
    
    for d in possible_dirs:
        if os.path.exists(d):
            # Find the actual snapshot directory
            if "snapshots" in d:
                subdirs = [os.path.join(d, x) for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]
                if subdirs:
                    return subdirs[0]  # Return first snapshot
            return d
    
    return None


def find_parquet_files(cache_dir):
    """Find all train parquet files"""
    
    parquet_pattern = os.path.join(cache_dir, "**", "train-*.parquet")
    parquet_files = glob.glob(parquet_pattern, recursive=True)
    parquet_files.sort()
    
    return parquet_files


def process_single_parquet_with_datasets(parquet_path, audio_dir, start_idx=0):
    """
    Load single parquet file using datasets library
    Extract audio files properly decoded
    """
    
    print(f"\nLoading: {os.path.basename(parquet_path)}")
    
    try:
        # Load parquet table directly to avoid audio auto-decode
        import pyarrow.parquet as pq
        table = pq.read_table(parquet_path)
        print(f"  Loaded: {len(table)} samples")
    except Exception as e:
        print(f"  Error loading: {e}")
        return [], 0
    
    samples = []
    audio_count = 0
    
    # Process each sample from raw table
    for idx in tqdm(range(len(table)), desc="Extracting", leave=False):
        try:
            # Get row as dict without decoding
            sample = {col: table[col][idx].as_py() for col in table.column_names}
            
            # Extract audio
            if "audio" not in sample:
                continue
            
            audio = sample["audio"]
            if audio is None:
                continue
            
            # Decode audio manually from bytes
            import io
            import numpy as np
            
            if isinstance(audio, dict):
                # Check if already decoded
                if "array" in audio:
                    audio_array = audio["array"]
                    sampling_rate = audio.get("sampling_rate", 16000)
                # Decode from bytes
                elif "bytes" in audio:
                    audio_bytes = audio["bytes"]
                    audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
                elif "path" in audio:
                    audio_array, sampling_rate = sf.read(audio["path"])
                else:
                    continue
            else:
                # Try to get array/bytes from object
                if hasattr(audio, 'array'):
                    audio_array = audio.array
                    sampling_rate = audio.sampling_rate if hasattr(audio, 'sampling_rate') else 16000
                elif hasattr(audio, 'bytes'):
                    audio_bytes = audio.bytes
                    audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
                else:
                    continue
            
            # Ensure audio is numpy array
            if not isinstance(audio_array, np.ndarray):
                audio_array = np.array(audio_array)
            
            if audio_array is None:
                continue
            
            # Get transcription
            transcription = sample.get("transcription", sample.get("sentence", sample.get("text", "")))
            
            # Calculate duration
            duration = len(audio_array) / sampling_rate
            
            # Save audio file
            global_idx = start_idx + audio_count
            audio_filename = f"audio_{global_idx:06d}.wav"
            audio_path = os.path.join(audio_dir, audio_filename)
            
            try:
                sf.write(audio_path, audio_array, sampling_rate)
            except Exception as e:
                print(f"Error saving audio {audio_filename}: {e}")
                continue
            
            # Collect metadata (only if audio saved successfully)
            sample_info = {
                "id": global_idx,
                "audio_path": audio_filename,
                "transcription": transcription,
                "duration": duration,
                "sampling_rate": sampling_rate,
                "speaker_id": sample.get("speaker_id", "unknown")
            }
            samples.append(sample_info)
            audio_count += 1
            
        except Exception as e:
            # Skip problematic samples
            if idx % 100 == 0:  # Print occasional errors
                print(f"Error processing sample {idx}: {e}")
            continue
    
    return samples, audio_count


def process_cached_with_datasets(
    output_dir="../preprocessing_data/phoaudiobook_100h",
    target_hours=100.0,
    delete_after_process=True
):
    """
    Process cached parquet files using datasets library
    Delete each file after processing
    """
    
    print("\n" + "="*60)
    print("PROCESS CACHED PARQUET - DATASETS LIBRARY")
    print("="*60)
    print(f"Target: {target_hours} hours")
    print(f"Auto-delete: {delete_after_process}")
    
    # Find cache directory
    cache_dir = find_cache_directory()
    if cache_dir is None:
        print("\nError: Could not find cache directory")
        return
    
    print(f"\nCache: {cache_dir}")
    
    # Create output directory
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    print(f"Output: {audio_dir}")
    
    # Find parquet files
    parquet_files = find_parquet_files(cache_dir)
    print(f"\nFound {len(parquet_files)} parquet files")
    
    if not parquet_files:
        print("No parquet files found!")
        return
    
    # Process files
    all_samples = []
    total_duration = 0.0
    target_duration = target_hours * 3600
    start_idx = 0
    
    print(f"\n{'='*60}")
    print(f"Processing {len(parquet_files)} files...")
    print(f"{'='*60}")
    
    for file_idx, parquet_path in enumerate(parquet_files, 1):
        print(f"\n[{file_idx}/{len(parquet_files)}] {os.path.basename(parquet_path)}")
        
        # Get file size
        file_size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
        print(f"  Size: {file_size_mb:.1f} MB")
        
        # Process this file
        samples, count = process_single_parquet_with_datasets(parquet_path, audio_dir, start_idx)
        
        if count == 0:
            print(f"  No valid samples extracted")
            if delete_after_process:
                os.remove(parquet_path)
                print(f"  Deleted ({file_size_mb:.1f} MB freed)")
            continue
        
        # Update totals
        all_samples.extend(samples)
        start_idx += count
        
        # Calculate duration
        batch_duration = sum(s['duration'] for s in samples)
        total_duration += batch_duration
        
        print(f"  Extracted: {count} files ({batch_duration/3600:.2f}h)")
        print(f"  Progress: {total_duration/3600:.2f}h / {target_hours}h ({len(all_samples)} samples)")
        
        # Delete parquet to free space
        if delete_after_process:
            os.remove(parquet_path)
            print(f"  Deleted ({file_size_mb:.1f} MB freed)")
        
        # Check if target reached
        if total_duration >= target_duration:
            print(f"\n✓ Target reached!")
            break
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": len(all_samples),
            "total_duration_seconds": total_duration,
            "total_duration_hours": total_duration / 3600,
            "target_hours": target_hours,
            "split": "train",
            "samples": all_samples
        }, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("COMPLETED!")
    print("="*60)
    print(f"\nStatistics:")
    print(f"  Files processed: {file_idx}/{len(parquet_files)}")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Total duration: {total_duration / 3600:.2f} hours")
    if len(all_samples) > 0:
        print(f"  Average duration: {total_duration / len(all_samples):.2f} seconds")
    print(f"\nOutput:")
    print(f"  Audio: {audio_dir}/")
    print(f"  Metadata: {metadata_path}")
    print("\n" + "="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process cached parquet with datasets library"
    )
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
        help="Target hours"
    )
    parser.add_argument(
        "--no_delete",
        action="store_true",
        help="Keep parquet files (don't delete)"
    )
    
    args = parser.parse_args()
    
    process_cached_with_datasets(
        output_dir=args.output_dir,
        target_hours=args.target_hours,
        delete_after_process=not args.no_delete
    )
