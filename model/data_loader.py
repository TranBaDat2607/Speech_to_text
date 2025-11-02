"""
Data loading utilities for Whisper model training
"""

import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple, Dict, Optional


class WhisperDataLoader:
    """
    DataLoader for Whisper training data
    
    Loads mel spectrograms and corresponding transcripts from processed dataset
    """
    
    def __init__(self, 
                 dataset_dir: str,
                 batch_size: int = 2,
                 max_samples: Optional[int] = None):
        """
        Initialize data loader
        
        Args:
            dataset_dir: Path to processed dataset directory
            batch_size: Batch size for training
            max_samples: Maximum number of samples to load (None for all)
        """
        self.dataset_dir = Path(dataset_dir)
        self.batch_size = batch_size
        self.max_samples = max_samples
        
        # Find all mel files
        self.mel_files = list(self.dataset_dir.glob("*_mel.npy"))
        
        if self.max_samples is not None:
            self.mel_files = self.mel_files[:self.max_samples]
        
        if len(self.mel_files) == 0:
            raise ValueError(f"No mel files found in {dataset_dir}")
    
    def load_sample(self, mel_file_path: Path) -> Tuple[np.ndarray, str]:
        """
        Load a single sample (mel spectrogram + transcript)
        
        Args:
            mel_file_path: Path to mel file
            
        Returns:
            Tuple of (mel_spectrogram, transcript)
        """
        # Load mel spectrogram
        mel_spec = np.load(mel_file_path)
        
        # Load corresponding JSON metadata
        json_path = str(mel_file_path).replace('_mel.npy', '_processed.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        transcript = metadata['transcript']
        
        return mel_spec.astype(np.float32), transcript
    
    def create_tf_dataset(self) -> tf.data.Dataset:
        """
        Create TensorFlow dataset from mel files
        
        Returns:
            tf.data.Dataset yielding (mel_spec, transcript) pairs
        """
        def data_generator():
            """Generator function for tf.data.Dataset"""
            for mel_file in self.mel_files:
                try:
                    mel_spec, transcript = self.load_sample(mel_file)
                    yield mel_spec, transcript
                except Exception as e:
                    print(f"Error loading {mel_file}: {e}")
                    continue
        
        # Create dataset
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                tf.TensorSpec(shape=[80, 3000], dtype=tf.float32),  # Mel spectrogram
                tf.TensorSpec(shape=(), dtype=tf.string)            # Transcript text
            )
        )
        
        return dataset
    
    def get_batched_dataset(self) -> tf.data.Dataset:
        """
        Get batched dataset for training
        
        Returns:
            Batched tf.data.Dataset
        """
        dataset = self.create_tf_dataset()
        
        # Batch and prefetch for performance
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
