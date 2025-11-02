"""
Distillation Trainer
Main training loop for knowledge distillation
"""

import os
import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
from typing import Optional, Dict, Tuple
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import Config
from training.utils import (
    LearningRateScheduler,
    MetricsTracker,
    Timer,
    CheckpointManager,
    format_time,
    compute_gradient_norm
)
from training.metrics_logger import MetricsLogger
from losses.distillation_loss import DistillationLoss
from student.load_student_tensorflow import WhisperStudentTensorFlow
from data.distillation_dataset import DistillationDataset, collate_fn_distillation

# Add model directory for tokenizer
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "model"))
from tokenizer import get_tokenizer


class DistillationTrainer:
    """
    Knowledge Distillation Trainer for Whisper
    Trains student model using teacher logits
    """
    
    def __init__(self, config: Config, load_openai_weights: bool = True):
        """
        Initialize trainer
        
        Args:
            config: Training configuration
            load_openai_weights: Whether to load OpenAI pretrained weights (default: True for batch 0)
        """
        self.config = config
        self.load_openai_weights = load_openai_weights
        
        print("\nInitializing trainer...")
        
        # Setup
        self._setup_device()
        self._setup_model()
        self._setup_loss()
        self._setup_optimizer()
        self._setup_data()
        
        # Training state
        self.current_step = 0
        self.current_epoch = 0
        self.best_val_metric = float('inf')
        self.patience_counter = 0
        
        # Metrics tracking
        self.metrics_tracker = MetricsTracker()
        self.timer = Timer()
        
        # Metrics logger for visualization
        self.metrics_logger = MetricsLogger(
            log_dir=str(self.config.paths.logs_dir) if hasattr(self.config.paths, 'logs_dir') else "./logs",
            experiment_name=self.config.experiment_name if hasattr(self.config, 'experiment_name') else None
        )
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.paths.checkpoints_dir,
            max_to_keep=self.config.training.keep_last_n_checkpoints,
            save_best_only=True,
            metric_name="val_loss",
            mode="min"
        )
        
        print("Trainer ready.\n")
    
    def _setup_device(self):
        """Setup GPU/CPU device"""
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus and self.config.hardware.device == "GPU":
            try:
                # Enable memory growth
                if self.config.hardware.allow_growth:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                
                # Select GPU
                tf.config.set_visible_devices(gpus[self.config.hardware.gpu_id], 'GPU')
                
            except RuntimeError as e:
                print(f"GPU setup error: {e}")
        
        # Setup mixed precision
        if self.config.training.mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
    
    def _setup_model(self):
        """Load student model"""
        self.student = WhisperStudentTensorFlow(
            model_name=self.config.model.student_model_name,
            freeze_encoder=self.config.model.freeze_encoder,
            load_openai_weights=self.load_openai_weights
        )
        
        self.model = self.student.model
        
        # Get tokenizer
        self.tokenizer = get_tokenizer(
            multilingual=True,
            num_languages=self.model.dims.n_vocab,
            language=self.config.data.language,
            task="transcribe"
        )
    
    def _setup_loss(self):
        """Setup distillation loss function"""
        self.loss_fn = DistillationLoss(
            alpha=self.config.distillation.soft_loss_weight,
            temperature=self.config.distillation.temperature,
            ignore_index=self.config.distillation.ignore_index
        )
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        
        # Calculate total steps
        # Note: Will be updated after dataset is loaded
        self.total_steps = 10000  # Placeholder
        
        # Learning rate scheduler
        self.lr_scheduler = LearningRateScheduler(
            base_lr=self.config.training.learning_rate,
            warmup_steps=self.config.training.warmup_steps,
            total_steps=self.total_steps,
            min_lr=self.config.training.min_learning_rate,
            schedule_type=self.config.training.lr_schedule
        )
        
        # Optimizer
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-6
        )
        
        # Wrap with LossScaleOptimizer for mixed precision
        if self.config.training.mixed_precision:
            self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        else:
            self.optimizer = optimizer
    
    def _setup_data(self):
        """Setup datasets and data loaders"""
        # Check if teacher logits exist
        logits_dir = Path(self.config.paths.teacher_logits_dir)
        if not logits_dir.exists() or not (logits_dir / "logits_metadata.json").exists():
            print(f"WARNING: No teacher logits at {logits_dir}")
            self.train_dataset = None
            self.val_dataset = None
            return
        
        # Load dataset
        try:
            audio_dir = Path(self.config.paths.preprocessed_dataset) / "audio"
            
            self.train_dataset = DistillationDataset(
                audio_dir=str(audio_dir),
                logits_dir=str(logits_dir),
                sample_rate=self.config.data.sample_rate,
                max_audio_length=self.config.data.max_audio_length
            )
            
            # Split train/val
            dataset_size = len(self.train_dataset)
            train_size = int(dataset_size * self.config.data.train_split)
            
            print(f"  Dataset: {dataset_size} samples")
            print(f"  Train: {train_size} samples")
            print(f"  Validation: {dataset_size - train_size} samples")
            
            # Update total steps
            steps_per_epoch = train_size // self.config.training.effective_batch_size
            self.total_steps = steps_per_epoch * self.config.training.epochs
            
            # Update LR scheduler with correct total steps
            self.lr_scheduler.total_steps = self.total_steps
            
            print(f"  Steps per epoch: {steps_per_epoch}")
            print(f"  Total steps: {self.total_steps}")
            
        except Exception as e:
            print(f"  Error loading dataset: {e}")
            self.train_dataset = None
            self.val_dataset = None
    
    def _audio_to_mel(self, audio: np.ndarray) -> tf.Tensor:
        """
        Convert audio waveform to mel spectrogram
        
        Args:
            audio: Audio waveform (samples,)
            
        Returns:
            Mel spectrogram (n_mels, frames) padded to model's expected length
        """
        import librosa
        
        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.config.data.sample_rate,
            n_mels=self.config.data.n_mels,
            n_fft=self.config.data.n_fft,
            hop_length=self.config.data.hop_length
        )
        
        # Convert to log scale
        mel = librosa.power_to_db(mel, ref=np.max)
        
        # Normalize
        mel = (mel + 80) / 80
        
        # Model has fixed positional embedding size (n_audio_ctx)
        # Conv layers downsample by factor of 2, so input needs 2x frames
        # base model: n_audio_ctx=1500 → requires 3000 input frames
        target_frames = self.model.dims.n_audio_ctx * 2
        
        # Pad or truncate to target length
        if mel.shape[1] < target_frames:
            pad_width = target_frames - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        elif mel.shape[1] > target_frames:
            mel = mel[:, :target_frames]
        
        return tf.constant(mel, dtype=tf.float32)
    
    def _preprocess_batch(self, batch: Dict) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Preprocess batch data
        
        Args:
            batch: Dictionary with audio, teacher_logits, text
            
        Returns:
            mel_inputs: Mel spectrogram (batch, n_mels, frames)
            teacher_logits: Teacher soft labels (batch, seq_len, vocab_size)
            decoder_input_ids: Token IDs for decoder (batch, seq_len)
        """
        # Convert audio to mel spectrogram
        # Note: This is simplified - in practice use proper mel conversion
        audio = batch['audio']  # (batch, audio_samples)
        
        # Placeholder mel conversion (replace with actual implementation)
        # In real training, use librosa or tf.signal
        batch_size = audio.shape[0]
        mel_frames = 3000  # Placeholder
        mel_inputs = tf.random.normal([batch_size, 80, mel_frames])  # Placeholder
        
        # Get teacher logits
        teacher_logits = batch['teacher_logits']  # (batch, seq_len, vocab_size)
        
        # Tokenize text
        texts = batch['text']
        decoder_input_ids = []
        
        for text in texts:
            tokens = self.tokenizer.encode(text)
            # Pad/truncate to max length
            if len(tokens) > self.config.data.max_text_length:
                tokens = tokens[:self.config.data.max_text_length]
            else:
                tokens = tokens + [self.tokenizer.eot] * (self.config.data.max_text_length - len(tokens))
            decoder_input_ids.append(tokens)
        
        decoder_input_ids = tf.constant(decoder_input_ids, dtype=tf.int32)
        
        return mel_inputs, teacher_logits, decoder_input_ids
    
    def train_step(
        self,
        mel_inputs: tf.Tensor,
        teacher_logits: tf.Tensor,
        decoder_input_ids: tf.Tensor
    ) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            mel_inputs: Mel spectrogram
            teacher_logits: Teacher soft labels
            decoder_input_ids: Ground truth tokens
            
        Returns:
            Dictionary with loss values
        """
        with tf.GradientTape() as tape:
            # Forward pass
            student_logits = self.model(mel_inputs, decoder_input_ids, training=True)
            
            # Debug: log shapes on first call
            if self.current_step == 0:
                print(f"      student_logits (after forward): {student_logits.shape}")
            
            # Compute loss
            total_loss, loss_dict = self.loss_fn(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=decoder_input_ids
            )
        
        # Compute gradients
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        
        # Clip gradients
        gradients, global_norm = tf.clip_by_global_norm(
            gradients,
            self.config.training.max_gradient_norm
        )
        
        # Update weights
        # LossScaleOptimizer automatically handles loss scaling and gradient unscaling
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Return metrics
        return {
            'loss': float(total_loss.numpy()),
            'kl_loss': float(loss_dict['kl_loss'].numpy()),
            'ce_loss': float(loss_dict['ce_loss'].numpy()),
            'grad_norm': float(global_norm.numpy())
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with epoch metrics
        """
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{self.config.training.epochs}")
        print(f"{'='*60}")
        
        self.model.training = True
        epoch_metrics = MetricsTracker()
        
        # Placeholder: Create data loader (simplified)
        # In real implementation, use proper data pipeline
        num_batches = 100  # Placeholder
        
        progress_bar = tqdm(range(num_batches), desc=f"Training")
        
        for batch_idx in progress_bar:
            # Update learning rate
            current_lr = self.lr_scheduler.get_lr(self.current_step)
            self.optimizer.learning_rate.assign(current_lr)
            
            # Placeholder batch (replace with real data loading)
            batch_size = self.config.training.batch_size
            mel_inputs = tf.random.normal([batch_size, 80, 3000])
            teacher_logits = tf.random.normal([batch_size, 10, 51864])
            decoder_input_ids = tf.random.uniform([batch_size, 10], 0, 1000, dtype=tf.int32)
            
            # Training step
            step_metrics = self.train_step(mel_inputs, teacher_logits, decoder_input_ids)
            
            # Update metrics
            for key, value in step_metrics.items():
                epoch_metrics.update(key, value)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{step_metrics['loss']:.4f}",
                'lr': f"{current_lr:.2e}"
            })
            
            # Increment step
            self.current_step += 1
            
            # Validation & checkpointing
            if self.current_step % self.config.training.eval_every_n_steps == 0:
                self._validate_and_save(epoch)
        
        # Epoch summary
        avg_metrics = {k: epoch_metrics.get_average(k) for k in ['loss', 'kl_loss', 'ce_loss']}
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Avg Loss: {avg_metrics['loss']:.4f}")
        print(f"  Avg KL Loss: {avg_metrics['kl_loss']:.4f}")
        print(f"  Avg CE Loss: {avg_metrics['ce_loss']:.4f}")
        
        return avg_metrics
    
    def _validate_and_save(self, epoch: int):
        """Run validation and save checkpoint"""
        # Placeholder validation
        val_loss = np.random.random() * 2.0  # Replace with real validation
        
        print(f"\n  Validation at step {self.current_step}:")
        print(f"    Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            step=self.current_step,
            epoch=epoch,
            metric_value=val_loss,
            additional_info={'lr': float(self.optimizer.learning_rate.numpy())}
        )
    
    def train(self):
        """Main training loop"""
        if self.train_dataset is None:
            print("\nERROR: Cannot start training - dataset not loaded!")
            print("Please run Phase 2 to generate teacher logits first.")
            return
        
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        self.timer.start()
        
        try:
            for epoch in range(self.config.training.epochs):
                self.current_epoch = epoch
                
                # Train one epoch
                epoch_metrics = self.train_epoch(epoch)
                
                # Early stopping check
                if self._check_early_stopping():
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break
            
            print("\n" + "="*60)
            print("Training Completed!")
            print("="*60)
            print(f"Total time: {format_time(self.timer.get_elapsed())}")
            print(f"Best checkpoint: {self.checkpoint_manager.get_best_checkpoint()}")
            
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            print("Saving checkpoint...")
            self._validate_and_save(self.current_epoch)
        
        except Exception as e:
            print(f"\nError during training: {e}")
            import traceback
            traceback.print_exc()
    
    def _check_early_stopping(self) -> bool:
        """Check if should stop training early"""
        # Placeholder - implement with real validation metrics
        return False
    
    def _create_tf_dataset(self, batch_dataset, batch_size: int, num_epochs: int = 1):
        """
        Create TensorFlow data pipeline with prefetching for efficient data loading
        
        Args:
            batch_dataset: DistillationDataset instance
            batch_size: Batch size for training
            num_epochs: Number of epochs to repeat dataset
            
        Returns:
            tf.data.Dataset with prefetching enabled
        """
        num_samples = len(batch_dataset)
        
        def data_generator():
            """Generator function to yield preprocessed batches"""
            for epoch in range(num_epochs):
                for step_idx in range(0, num_samples, batch_size):
                    # Load batch samples
                    start_idx = step_idx
                    end_idx = min(start_idx + batch_size, num_samples)
                    batch_samples = [batch_dataset[i] for i in range(start_idx, end_idx)]
                    
                    # Preprocess batch (audio -> mel, tokenize, padding)
                    mel_inputs_list = []
                    teacher_logits_list = []
                    decoder_input_ids_list = []
                    
                    for sample in batch_samples:
                        # Convert audio to mel
                        audio = sample['audio']
                        mel = self._audio_to_mel(audio)
                        mel_inputs_list.append(mel)
                        
                        # Teacher logits
                        teacher_logits_np = sample['teacher_logits']
                        
                        # Squeeze batch dim if present
                        if len(teacher_logits_np.shape) == 3 and teacher_logits_np.shape[0] == 1:
                            teacher_logits_np = teacher_logits_np[0]
                        
                        # Match vocab size
                        if teacher_logits_np.shape[-1] > 51864:
                            teacher_logits_np = teacher_logits_np[..., :51864]
                        
                        # Tokenize text
                        text = sample['text']
                        tokens = self.tokenizer.encode(text)
                        if len(tokens) > self.config.data.max_text_length:
                            tokens = tokens[:self.config.data.max_text_length]
                        
                        # Pad teacher logits to match decoder length
                        teacher_seq_len = teacher_logits_np.shape[0]
                        decoder_seq_len = len(tokens)
                        
                        if teacher_seq_len < decoder_seq_len:
                            pad_len = decoder_seq_len - teacher_seq_len
                            teacher_logits_np = np.pad(
                                teacher_logits_np,
                                ((0, pad_len), (0, 0)),
                                mode='constant',
                                constant_values=-100
                            )
                        elif teacher_seq_len > decoder_seq_len:
                            teacher_logits_np = teacher_logits_np[:decoder_seq_len, :]
                        
                        teacher_logits_list.append(teacher_logits_np)
                        decoder_input_ids_list.append(tokens)
                    
                    # Pad sequences to max length in batch
                    max_seq_len = max(len(tokens) for tokens in decoder_input_ids_list)
                    
                    # Pad decoder_input_ids
                    padded_decoder_ids = []
                    for tokens in decoder_input_ids_list:
                        if len(tokens) < max_seq_len:
                            tokens = tokens + [0] * (max_seq_len - len(tokens))
                        padded_decoder_ids.append(tokens)
                    
                    # Pad teacher_logits
                    padded_teacher_logits = []
                    for logits in teacher_logits_list:
                        if logits.shape[0] < max_seq_len:
                            pad_len = max_seq_len - logits.shape[0]
                            logits = np.pad(
                                logits,
                                ((0, pad_len), (0, 0)),
                                mode='constant',
                                constant_values=-100
                            )
                        padded_teacher_logits.append(logits)
                    
                    # Stack into batches
                    mel_inputs = np.stack(mel_inputs_list)
                    teacher_logits = np.stack(padded_teacher_logits)
                    decoder_input_ids = np.array(padded_decoder_ids, dtype=np.int32)
                    
                    yield mel_inputs, teacher_logits, decoder_input_ids
        
        # Create TensorFlow dataset from generator
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                tf.TensorSpec(shape=(None, 80, None), dtype=tf.float32),  # mel_inputs
                tf.TensorSpec(shape=(None, None, 51864), dtype=tf.float32),  # teacher_logits
                tf.TensorSpec(shape=(None, None), dtype=tf.int32)  # decoder_input_ids
            )
        )
        
        # Apply prefetching for parallel data loading
        prefetch_size = getattr(self.config.hardware, 'prefetch_size', 2)
        dataset = dataset.prefetch(buffer_size=prefetch_size)
        
        return dataset
    
    def train_on_batch_range(
        self,
        logits_dir: str,
        start_idx: int,
        end_idx: int,
        num_epochs: int = 1
    ) -> Dict[str, float]:
        """
        Train on a specific batch range of samples
        
        Args:
            logits_dir: Directory containing teacher logits for this batch
            start_idx: Starting sample index
            end_idx: Ending sample index (exclusive)
            num_epochs: Number of epochs to train on this batch
            
        Returns:
            Dictionary with training metrics
        """
        num_samples = end_idx - start_idx
        
        print(f"\nTraining on batch: samples {start_idx}-{end_idx} ({num_samples} samples)")
        print(f"Logits dir: {logits_dir}")
        
        batch_dataset = DistillationDataset(
            audio_dir=str(Path(self.config.paths.preprocessed_dataset) / "audio"),
            logits_dir=logits_dir,
            sample_rate=self.config.data.sample_rate,
            max_audio_length=self.config.data.max_audio_length
        )
        
        if len(batch_dataset) == 0:
            print("Warning: No samples found in batch")
            return {'loss': 0.0}
        
        self.model.training = True
        epoch_metrics = MetricsTracker()
        
        # Calculate batch number for logging
        batch_num = start_idx // self.config.distillation.mini_batch_size if hasattr(self.config.distillation, 'mini_batch_size') else 0
        
        # Calculate steps based on batch size
        batch_size = self.config.training.batch_size
        steps_per_epoch = (num_samples + batch_size - 1) // batch_size
        
        print(f"Loaded DistillationDataset: {len(batch_dataset)} samples")
        print(f"Batch size: {batch_size}, Steps per epoch: {steps_per_epoch}")
        print(f"Using TensorFlow data pipeline with prefetching (buffer_size={getattr(self.config.hardware, 'prefetch_size', 2)})")
        
        # Create TensorFlow dataset with prefetching
        dataset = self._create_tf_dataset(batch_dataset, batch_size, num_epochs)
        
        # Training loop using prefetched data
        step_idx = 0
        epoch = 0
        first_step = True
        
        for mel_inputs, teacher_logits, decoder_input_ids in dataset:
            # Convert numpy arrays to TensorFlow tensors if needed
            if not isinstance(mel_inputs, tf.Tensor):
                mel_inputs = tf.constant(mel_inputs, dtype=tf.float32)
            if not isinstance(teacher_logits, tf.Tensor):
                teacher_logits = tf.constant(teacher_logits, dtype=tf.float32)
            if not isinstance(decoder_input_ids, tf.Tensor):
                decoder_input_ids = tf.constant(decoder_input_ids, dtype=tf.int32)
            
            # Update learning rate
            current_lr = self.lr_scheduler.get_lr(self.current_step)
            self.optimizer.learning_rate.assign(current_lr)
            
            # Print progress
            step_in_epoch = step_idx % steps_per_epoch
            if step_in_epoch == 0:
                epoch = step_idx // steps_per_epoch
                print(f"  Epoch {epoch + 1}/{num_epochs}")
            
            if step_in_epoch % 10 == 0:
                print(f"    Step {step_in_epoch + 1}/{steps_per_epoch}", end='\r')
            
            # Debug info for first step
            if first_step:
                print(f"    Running training step (compiling graph, this may take time)...")
                print(f"    Debug shapes:")
                print(f"      mel_inputs: {mel_inputs.shape}")
                print(f"      teacher_logits: {teacher_logits.shape}")
                print(f"      decoder_input_ids: {decoder_input_ids.shape}")
            
            # Training step (data is already prefetched!)
            step_metrics = self.train_step(mel_inputs, teacher_logits, decoder_input_ids)
            
            if first_step:
                print(f"    First step complete! Loss: {step_metrics['loss']:.4f}")
                print(f"    Subsequent steps will be faster with prefetching...")
                first_step = False
            
            # Update metrics tracker
            for key, value in step_metrics.items():
                epoch_metrics.update(key, value)
            
            # Log step-level metrics (if enabled)
            if hasattr(self.config, 'logging') and getattr(self.config.logging, 'log_step_metrics', False):
                save_every = getattr(self.config.logging, 'save_step_metrics_every', 1)
                if step_in_epoch % save_every == 0:
                    step_metrics_dict = {k: float(v) for k, v in step_metrics.items()}
                    step_metrics_dict['batch'] = batch_num
                    step_metrics_dict['epoch'] = epoch + 1
                    step_metrics_dict['step_in_epoch'] = step_in_epoch
                    step_metrics_dict['global_step'] = self.current_step
                    self.metrics_logger.log_step(self.current_step, step_metrics_dict, phase='train')
            
            self.current_step += 1
            step_idx += 1
            
            # New line after each epoch
            if step_in_epoch == steps_per_epoch - 1:
                print()  # New line after progress
        
        avg_metrics = {k: epoch_metrics.get_average(k) for k in ['loss', 'kl_loss', 'ce_loss']}
        
        print(f"  Batch training complete - Avg loss: {avg_metrics['loss']:.4f}")
        
        # Log batch-level metrics for visualization
        self.metrics_logger.log_batch(batch_num, avg_metrics)
        
        return avg_metrics
