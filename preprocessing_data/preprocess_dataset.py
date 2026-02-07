"""
Main script to preprocess Whisper dataset - Safe version with memory fix
Convert WAV files to log mel spectrograms for TensorFlow Whisper training
"""

import os
import sys
import argparse
from pathlib import Path

# Fix TensorFlow memory issues BEFORE importing TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN to prevent double free
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Limit OpenBLAS threads
os.environ['MKL_NUM_THREADS'] = '1'  # Limit MKL threads

import tensorflow as tf

# Configure TensorFlow
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

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
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
