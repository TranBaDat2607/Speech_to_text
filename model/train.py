"""
Training script for Whisper model
"""

import os
import time
import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple

# Import our custom modules
from model import create_whisper_model
from data_loader import WhisperDataLoader  
from text_processor import WhisperTextProcessor


class WhisperTrainer:
    """
    Trainer class for Whisper model
    """
    
    def __init__(self, 
                 model_name: str = "tiny",
                 learning_rate: float = 1e-4,
                 language: str = "vi",
                 device: str = "CPU"):
        """
        Initialize trainer
        
        Args:
            model_name: Whisper model size ("tiny", "base", "small", etc.)
            learning_rate: Learning rate for optimizer
            language: Language code for tokenizer
            device: Device to use ("CPU" or "GPU")
        """
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.language = language
        self.device = device
        
        print(f"Initializing WhisperTrainer:")
        print(f"  Model: {model_name}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Language: {language}")
        print(f"  Device: {device}")
        
        # Force CPU usage if specified
        if device == "CPU":
            tf.config.experimental.set_visible_devices([], 'GPU')
            print("  Forced CPU usage")
        
        # Initialize model
        self.model = create_whisper_model(model_name)
        
        # Initialize text processor
        self.text_processor = WhisperTextProcessor(
            language=language,
            task="transcribe"
        )
        
        # Initialize optimizer
        self.optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-6
        )
        
        # Initialize loss function
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE  # We'll handle reduction manually
        )
        
        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        
        print("Trainer initialized successfully!")
    
    def compute_loss(self, 
                     logits: tf.Tensor, 
                     targets: tf.Tensor, 
                     ignore_index: int = 50257) -> tf.Tensor:
        """
        Compute cross-entropy loss with masking
        
        Args:
            logits: Model predictions [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
            ignore_index: Token ID to ignore in loss (typically EOT token)
        
        Returns:
            Scalar loss value
        """
        # Create mask to ignore padding tokens (EOT tokens)
        mask = tf.not_equal(targets, ignore_index)
        mask = tf.cast(mask, tf.float32)
        
        # Compute loss for each token
        token_losses = self.loss_fn(targets, logits)
        
        # Apply mask
        masked_losses = token_losses * mask
        
        # Average over valid tokens only
        total_loss = tf.reduce_sum(masked_losses)
        total_tokens = tf.reduce_sum(mask)
        
        # Avoid division by zero
        loss = tf.cond(
            total_tokens > 0,
            lambda: total_loss / total_tokens,
            lambda: 0.0
        )
        
        return loss
    
    @tf.function
    def train_step(self, 
                   mel_spectrograms: tf.Tensor, 
                   decoder_inputs: tf.Tensor, 
                   decoder_targets: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Single training step
        
        Args:
            mel_spectrograms: Input mel spectrograms [batch_size, 80, 3000]
            decoder_inputs: Decoder input tokens [batch_size, seq_len]
            decoder_targets: Target tokens [batch_size, seq_len]
        
        Returns:
            Dictionary with loss and accuracy
        """
        with tf.GradientTape() as tape:
            # Forward pass
            logits = self.model(mel_spectrograms, decoder_inputs, training=True)
            
            # Compute loss
            loss = self.compute_loss(logits, decoder_targets)
        
        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Clip gradients to prevent exploding gradients
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update metrics
        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(decoder_targets, logits)
        
        return {
            'loss': loss,
            'accuracy': self.train_accuracy.result()
        }
    
    def train_epoch(self, dataset: tf.data.Dataset, epoch: int, total_epochs: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            dataset: Training dataset
            epoch: Current epoch number
            total_epochs: Total number of epochs
        
        Returns:
            Dictionary with epoch metrics
        """
        print(f"\n=== Epoch {epoch + 1}/{total_epochs} ===")
        
        # Reset metrics
        self.train_loss.reset_state()
        self.train_accuracy.reset_state()
        
        num_batches = 0
        epoch_start_time = time.time()
        
        # Training loop
        for batch_idx, (batch_mel, batch_text) in enumerate(dataset):
            # Convert text to tokens
            batch_text_str = [text.numpy().decode('utf-8') for text in batch_text]
            decoder_inputs, decoder_targets = self.text_processor.process_batch_texts(batch_text_str)
            
            # Training step
            step_metrics = self.train_step(batch_mel, decoder_inputs, decoder_targets)
            
            num_batches += 1
            
            # Print progress
            if (batch_idx + 1) % 1 == 0:  # Print every batch since we have small dataset
                print(f"  Batch {batch_idx + 1}: "
                      f"Loss = {step_metrics['loss']:.4f}, "
                      f"Accuracy = {step_metrics['accuracy']:.4f}")
        
        epoch_time = time.time() - epoch_start_time
        
        # Compute epoch metrics
        epoch_metrics = {
            'loss': float(self.train_loss.result()),
            'accuracy': float(self.train_accuracy.result()),
            'time': epoch_time,
            'batches': num_batches
        }
        
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Average Loss: {epoch_metrics['loss']:.4f}")
        print(f"  Average Accuracy: {epoch_metrics['accuracy']:.4f}")
        print(f"  Time: {epoch_metrics['time']:.2f}s")
        print(f"  Batches: {epoch_metrics['batches']}")
        
        return epoch_metrics
    
    def train(self, 
              dataset_dir: str,
              epochs: int = 2,
              batch_size: int = 2,
              max_samples: int = 10):
        """
        Main training function
        
        Args:
            dataset_dir: Path to processed dataset
            epochs: Number of training epochs
            batch_size: Batch size
            max_samples: Maximum number of samples to use
        """
        print(f"\nStarting Whisper Training")
        print(f"Dataset: {dataset_dir}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Max samples: {max_samples}")
        
        # Create data loader
        data_loader = WhisperDataLoader(
            dataset_dir=dataset_dir,
            batch_size=batch_size,
            max_samples=max_samples
        )
        
        # Get dataset
        train_dataset = data_loader.get_batched_dataset()
        
        # Training loop
        training_history = []
        
        for epoch in range(epochs):
            epoch_metrics = self.train_epoch(train_dataset, epoch, epochs)
            training_history.append(epoch_metrics)
        
        print(f"\nTraining Complete!")
        print(f"Final Loss: {training_history[-1]['loss']:.4f}")
        print(f"Final Accuracy: {training_history[-1]['accuracy']:.4f}")
        
        return training_history
    
    def test_model_output(self, dataset_dir: str):
        """
        Test model output with a single sample
        """
        print("\n=== Testing Model Output ===")
        
        # Load one sample
        data_loader = WhisperDataLoader(dataset_dir, batch_size=1, max_samples=1)
        dataset = data_loader.get_batched_dataset()
        
        for mel_batch, text_batch in dataset.take(1):
            print(f"Input mel shape: {mel_batch.shape}")
            print(f"Input text: {text_batch[0].numpy().decode('utf-8')[:100]}...")
            
            # Process text
            text_str = text_batch[0].numpy().decode('utf-8')
            decoder_inputs, decoder_targets = self.text_processor.process_batch_texts([text_str])
            
            print(f"Decoder inputs shape: {decoder_inputs.shape}")
            print(f"Decoder targets shape: {decoder_targets.shape}")
            
            # Forward pass
            logits = self.model(mel_batch, decoder_inputs, training=False)
            print(f"Model output shape: {logits.shape}")
            
            # Compute loss
            loss = self.compute_loss(logits, decoder_targets)
            print(f"Loss: {loss:.4f}")
            
            print("Model forward pass successful!")
            break


def main():
    """Main training function"""
    
    # Configuration
    CONFIG = {
        'dataset_dir': r"c:\Users\Admin\Desktop\dat301m\Speech_to_text\preprocessing_data\processed_dataset",
        'model_name': 'tiny',  # Start with tiny model for testing
        'epochs': 2,
        'batch_size': 2,
        'max_samples': 10,
        'learning_rate': 1e-4,
        'language': 'vi',
        'device': 'CPU'  # Use CPU as requested
    }
    
    print("Whisper Training Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Initialize trainer
    trainer = WhisperTrainer(
        model_name=CONFIG['model_name'],
        learning_rate=CONFIG['learning_rate'],
        language=CONFIG['language'],
        device=CONFIG['device']
    )
    
    # Test model output first
    trainer.test_model_output(CONFIG['dataset_dir'])
    
    # Start training
    training_history = trainer.train(
        dataset_dir=CONFIG['dataset_dir'],
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        max_samples=CONFIG['max_samples']
    )
    
    print(f"\nTraining History:")
    for i, metrics in enumerate(training_history):
        print(f"Epoch {i+1}: Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.4f}, Time={metrics['time']:.1f}s")


if __name__ == "__main__":
    main()
