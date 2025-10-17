"""
Main script to preprocess Whisper dataset
Convert WAV files to log mel spectrograms for TensorFlow Whisper training
"""

import os
import sys
import argparse
from pathlib import Path

# Force TensorFlow to use CPU only to avoid CUDA issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from audio_processor import batch_process_dataset


def main():
    parser = argparse.ArgumentParser(description='Preprocess Whisper dataset for TensorFlow training')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input dataset directory containing WAV and JSON files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed mel spectrograms')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    # Validate input directory
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_path}")
        sys.exit(1)
    
    if not input_path.is_dir():
        print(f"Error: Input path is not a directory: {input_path}")
        sys.exit(1)
    
    # Check if there are WAV files
    wav_files = list(input_path.glob("*.wav"))
    if not wav_files:
        print(f"Error: No WAV files found in {input_path}")
        sys.exit(1)
    
    print(f"Input directory: {input_path}")
    print(f"Output directory: {output_path}")
    print(f"Found {len(wav_files)} WAV files to process")
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process the dataset
    try:
        processed_count = batch_process_dataset(str(input_path), str(output_path))
        
        print(f"\nPreprocessing completed successfully!")
        print(f"Processed {processed_count} files")
        print(f"Output files saved to: {output_path}")
        
        print(f"\nOutput structure:")
        print(f"- *_mel.npy: Mel spectrograms (shape: 80x3000)")
        print(f"- *_processed.json: Metadata and transcripts")
        print(f"- processing_summary.json: Processing summary")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        sys.exit(1)


def process_sachnoivietnam15():
    """Helper function to process the specific dataset"""
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "crawl_data" / "datasets" / "sachnoivietnam15" / "dataset"
    output_dir = base_dir / "preprocessing_data" / "processed_dataset"
    
    print("Processing sachnoivietnam15 dataset...")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    if not input_dir.exists():
        print(f"Error: Dataset directory not found: {input_dir}")
        return False
    
    try:
        processed_count = batch_process_dataset(str(input_dir), str(output_dir))
        print(f"\nProcessing completed: {processed_count} files processed")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


if __name__ == "__main__":
    # If no arguments provided, try to process the default dataset
    if len(sys.argv) == 1:
        print("No arguments provided. Processing default dataset...")
        if process_sachnoivietnam15():
            print("\nSuccess! Dataset preprocessed.")
        else:
            print("\nFailed to process dataset.")
    else:
        main()
