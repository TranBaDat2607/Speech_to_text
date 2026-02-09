"""Optimized Offline Distillation Training for Whisper"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import tensorflow as tf
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "model"))
sys.path.insert(0, str(Path(__file__).parent / "distillation"))

from model import Whisper
from model_dimensions import get_whisper_dimensions
from losses.distillation_loss import DistillationLoss
from data.distillation_dataset import DistillationDataset
from training.config import Config
from transformers import WhisperProcessor


class OptimizedDistillationTrainer:
    """Memory-optimized trainer for offline knowledge distillation"""

    def __init__(self, config_path: str = "distillation/config/config.yaml"):
        self.config = Config.from_yaml(config_path)
        self.current_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.gradient_accumulation_steps = getattr(
            self.config.training, 'gradient_accumulation_steps', 1
        )
        self.accumulated_gradients = None
        self.accumulation_counter = 0

        self._setup_gpu()
        self._setup_model()
        self._setup_loss()
        self._setup_optimizer()
        self._setup_data()

    def _setup_gpu(self):
        """Setup GPU with memory optimizations"""
        gpus = tf.config.list_physical_devices('GPU')

        if gpus and self.config.hardware.device == "GPU":
            try:
                if self.config.hardware.allow_growth:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.set_visible_devices(gpus[self.config.hardware.gpu_id], 'GPU')
            except RuntimeError as e:
                print(f"GPU setup warning: {e}")
        else:
            print("No GPU found, using CPU")

        if self.config.training.mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)

    def _setup_model(self):
        """Load student model"""
        self.dims = get_whisper_dimensions(self.config.model.student_model_name)
        self.model = Whisper(self.dims)
        dummy_mel = tf.random.normal([1, self.dims.n_mels, 3000])
        dummy_tokens = tf.random.uniform([1, 10], 0, self.dims.n_vocab, dtype=tf.int32)
        _ = self.model(dummy_mel, dummy_tokens, training=False)

        if self.config.model.student_pretrained:
            try:
                from student.load_student_tensorflow import WhisperStudentTensorFlow
                student = WhisperStudentTensorFlow(
                    model_name=self.config.model.student_model_name,
                    load_openai_weights=True
                )
                self.model = student.model
                print("OpenAI weights loaded")
            except Exception as e:
                print(f"Could not load OpenAI weights: {e}")

        if self.config.model.freeze_encoder:
            self.model.encoder.trainable = False

        self.processor = WhisperProcessor.from_pretrained("vinai/PhoWhisper-large")
        self.tokenizer = self.processor.tokenizer

    def _setup_loss(self):
        """Setup distillation loss function"""
        self.loss_fn = DistillationLoss(
            alpha=self.config.distillation.soft_loss_weight,
            temperature=self.config.distillation.temperature,
            ignore_index=self.config.distillation.ignore_index
        )

    def _setup_optimizer(self):
        """Setup optimizer with learning rate scheduling"""
        self.total_steps = 10000

        initial_lr = self.config.training.learning_rate
        warmup_steps = self.config.training.warmup_steps
        min_lr = self.config.training.min_learning_rate

        if self.config.training.lr_schedule == "cosine":
            class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
                def __init__(self, initial_lr, warmup_steps, total_steps, min_lr):
                    self.initial_lr = initial_lr
                    self.warmup_steps = warmup_steps
                    self.total_steps = total_steps
                    self.min_lr = min_lr

                def __call__(self, step):
                    step = tf.cast(step, tf.float32)
                    warmup_steps = tf.cast(self.warmup_steps, tf.float32)
                    total_steps = tf.cast(self.total_steps, tf.float32)

                    warmup_lr = self.initial_lr * step / warmup_steps
                    decay_steps = total_steps - warmup_steps
                    decay_progress = (step - warmup_steps) / decay_steps
                    cosine_decay = 0.5 * (1 + tf.cos(np.pi * decay_progress))
                    decay_lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay

                    return tf.cond(
                        step < warmup_steps,
                        lambda: warmup_lr,
                        lambda: decay_lr
                    )

            lr_schedule = WarmupCosineDecay(initial_lr, warmup_steps, self.total_steps, min_lr)
        else:
            lr_schedule = initial_lr

        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=self.config.training.weight_decay,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-6
        )

        if self.config.training.mixed_precision:
            self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        else:
            self.optimizer = optimizer

    def _setup_data(self):
        """Setup data pipeline"""
        logits_dir = Path(self.config.paths.teacher_logits_dir)
        if not logits_dir.exists():
            self.train_dataset = None
            return

        try:
            audio_dir = Path(self.config.paths.preprocessed_dataset) / "audio"
            mel_dir = Path(self.config.paths.preprocessed_dataset) / "mels"
            use_precomputed_mels = mel_dir.exists() and len(list(mel_dir.glob("*.npy"))) > 0

            if use_precomputed_mels:
                print(f"Using pre-computed mels from {mel_dir}")
            else:
                print(f"WARNING: Pre-computed mels not found. Run precompute_mels.py first")

            self.train_dataset = DistillationDataset(
                audio_dir=str(audio_dir),
                logits_dir=str(logits_dir),
                sample_rate=self.config.data.sample_rate,
                max_audio_length=self.config.data.max_audio_length,
                use_precomputed_mels=use_precomputed_mels,
                mel_dir=str(mel_dir) if use_precomputed_mels else None
            )

            dataset_size = len(self.train_dataset)
            steps_per_epoch = dataset_size // (
                self.config.training.batch_size * self.gradient_accumulation_steps
            )
            self.total_steps = steps_per_epoch * self.config.training.epochs
            print(f"Steps/epoch: {steps_per_epoch}, Total: {self.total_steps}")

        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.train_dataset = None

    def _audio_to_mel(self, audio: np.ndarray) -> tf.Tensor:
        """Convert audio to mel spectrogram"""
        from whisper.audio import log_mel_spectrogram, pad_or_trim
        audio_padded = pad_or_trim(audio)
        mel = log_mel_spectrogram(audio_padded, n_mels=self.config.data.n_mels)
        mel = mel.cpu().numpy()
        return tf.constant(mel, dtype=tf.float32)

    def train_step(
        self,
        mel_inputs: tf.Tensor,
        teacher_logits: tf.Tensor,
        decoder_input_ids: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """Single training step with gradient accumulation"""
        with tf.GradientTape() as tape:
            decoder_input = decoder_input_ids[:, :-1]
            labels = decoder_input_ids[:, 1:]
            student_logits = self.model(mel_inputs, decoder_input, training=True)

            target_len = tf.shape(decoder_input)[1]
            teacher_len = tf.shape(teacher_logits)[1]
            min_len = tf.minimum(teacher_len, target_len)
            teacher_logits_truncated = teacher_logits[:, :min_len, :]

            pad_needed = target_len - min_len
            teacher_logits_shifted = tf.cond(
                pad_needed > 0,
                lambda: tf.pad(
                    teacher_logits_truncated,
                    [[0, 0], [0, pad_needed], [0, 0]],
                    constant_values=-100.0
                ),
                lambda: teacher_logits_truncated
            )

            total_loss, loss_dict = self.loss_fn(
                student_logits=student_logits,
                teacher_logits=teacher_logits_shifted,
                labels=labels
            )

            scaled_loss = total_loss / tf.cast(self.gradient_accumulation_steps, tf.float32)

        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)

        if self.accumulated_gradients is None:
            self.accumulated_gradients = [None for _ in gradients]

        for i, grad in enumerate(gradients):
            if grad is not None:
                if self.accumulated_gradients[i] is None:
                    self.accumulated_gradients[i] = grad
                else:
                    self.accumulated_gradients[i] = self.accumulated_gradients[i] + grad

        self.accumulation_counter += 1
        should_apply = (self.accumulation_counter >= self.gradient_accumulation_steps)

        if should_apply:
            clipped_gradients, global_norm = tf.clip_by_global_norm(
                self.accumulated_gradients,
                self.config.training.max_gradient_norm
            )

            self.optimizer.apply_gradients(
                [(g, v) for g, v in zip(clipped_gradients, self.model.trainable_variables) if g is not None]
            )

            self.accumulated_gradients = [None for _ in self.accumulated_gradients]
            self.accumulation_counter = 0
        else:
            global_norm = tf.constant(0.0)

        return {
            'loss': total_loss,
            'kl_loss': loss_dict['kl_loss'],
            'ce_loss': loss_dict['ce_loss'],
            'grad_norm': global_norm,
            'applied_gradients': should_apply
        }

    def train(self):
        """Main training loop"""
        if self.train_dataset is None:
            return
        start_time = time.time()

        try:
            checkpoint_dir = Path(self.config.paths.checkpoints_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            for epoch in range(self.config.training.epochs):
                self.current_epoch = epoch
                epoch_metrics = self._train_epoch(epoch)
                self._save_checkpoint(epoch, epoch_metrics)

            elapsed_time = time.time() - start_time

        except KeyboardInterrupt:
            self._save_checkpoint(self.current_epoch, {'avg_loss': 0.0})
        except Exception as e:
            import traceback
            traceback.print_exc()

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        batch_size = self.config.training.batch_size
        dataset_size = len(self.train_dataset)
        num_batches = dataset_size // batch_size

        epoch_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_ce_loss = 0.0
        num_updates = 0

        pbar = tqdm(range(num_batches), desc=f"Training")

        for batch_idx in pbar:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, dataset_size)
            batch_samples = [self.train_dataset[i] for i in range(start_idx, end_idx)]

            mel_inputs_list = []
            teacher_logits_list = []
            decoder_ids_list = []

            for sample in batch_samples:
                if 'mel' in sample:
                    mel = tf.constant(sample['mel'], dtype=tf.float32)
                else:
                    mel = self._audio_to_mel(sample['audio'])
                mel_inputs_list.append(mel)
                teacher_logits_list.append(sample['teacher_logits'])

                if sample.get('tokens') is not None:
                    tokens = sample['tokens']
                else:
                    inputs = self.processor(text=sample['text'], return_tensors="pt")
                    tokens = inputs.input_ids[0].numpy()
                decoder_ids_list.append(tokens)

            mel_inputs = tf.stack(mel_inputs_list)
            teacher_logits = tf.constant(np.stack(teacher_logits_list), dtype=tf.float32)

            max_len = max(len(t) for t in decoder_ids_list)
            padded_tokens = []
            for tokens in decoder_ids_list:
                if len(tokens) < max_len:
                    tokens = np.pad(tokens, (0, max_len - len(tokens)), constant_values=0)
                padded_tokens.append(tokens)
            decoder_input_ids = tf.constant(padded_tokens, dtype=tf.int32)

            if hasattr(self.optimizer.learning_rate, '__call__'):
                current_lr = self.optimizer.learning_rate(self.current_step)
            else:
                current_lr = self.optimizer.learning_rate

            step_metrics = self.train_step(mel_inputs, teacher_logits, decoder_input_ids)

            epoch_loss += float(step_metrics['loss'])
            epoch_kl_loss += float(step_metrics['kl_loss'])
            epoch_ce_loss += float(step_metrics['ce_loss'])

            if step_metrics['applied_gradients']:
                num_updates += 1

            pbar.set_postfix({
                'loss': f"{float(step_metrics['loss']):.4f}",
                'lr': f"{float(current_lr):.2e}"
            })

            self.current_step += 1

        return {
            'avg_loss': epoch_loss / num_batches,
            'avg_kl_loss': epoch_kl_loss / num_batches,
            'avg_ce_loss': epoch_ce_loss / num_batches,
        }

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.paths.checkpoints_dir)
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.weights.h5"
        self.model.save_weights(str(checkpoint_path))

        metadata = {
            'epoch': epoch + 1,
            'step': self.current_step,
            'loss': metrics.get('avg_loss', 0.0),
            'model_name': self.config.model.student_model_name
        }

        metadata_path = checkpoint_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    """Main entry point"""
    config_path = "distillation/config/config.yaml"

    if not Path(config_path).exists():
        return

    trainer = OptimizedDistillationTrainer(config_path=config_path)
    trainer.train()


if __name__ == "__main__":
    main()
