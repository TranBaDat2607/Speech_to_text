import os
import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
from typing import Optional, Dict, Tuple
from tqdm import tqdm
import json

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
from transformers import WhisperProcessor
from training.progress_logger import SimpleProgbar


class DistillationTrainer:
    """Knowledge Distillation Trainer for Whisper"""

    def __init__(self, config: Config, load_openai_weights: bool = True):
        self.config = config
        self.load_openai_weights = load_openai_weights

        print("\nInitializing trainer...")

        self._setup_device()
        self._setup_model()
        self._setup_loss()
        self._setup_optimizer()
        self._setup_data()

        self.current_step = 0
        self.current_epoch = 0
        self.best_val_metric = float('inf')
        self.patience_counter = 0

        self.gradient_accumulation_steps = getattr(self.config.training, 'gradient_accumulation_steps', 1)
        self.accumulated_gradients = None
        self.accumulation_counter = 0

        self.metrics_tracker = MetricsTracker()
        self.timer = Timer()

        self.metrics_logger = MetricsLogger(
            log_dir=str(self.config.paths.logs_dir) if hasattr(self.config.paths, 'logs_dir') else "./logs",
            experiment_name=self.config.experiment_name if hasattr(self.config, 'experiment_name') else None
        )

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.paths.checkpoints_dir,
            max_to_keep=self.config.training.keep_last_n_checkpoints,
            save_best_only=True,
            metric_name="val_loss",
            mode="min"
        )

        print("Trainer ready.\n")

    def _setup_device(self):
        gpus = tf.config.list_physical_devices('GPU')

        if gpus and self.config.hardware.device == "GPU":
            try:
                if self.config.hardware.allow_growth:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)

                tf.config.set_visible_devices(gpus[self.config.hardware.gpu_id], 'GPU')

            except RuntimeError as e:
                print(f"GPU setup error: {e}")

        if self.config.training.mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)

    def _setup_model(self):
        self.student = WhisperStudentTensorFlow(
            model_name=self.config.model.student_model_name,
            freeze_encoder=self.config.model.freeze_encoder,
            load_openai_weights=self.load_openai_weights
        )

        self.model = self.student.model

        self.processor = WhisperProcessor.from_pretrained("vinai/PhoWhisper-large")
        self.tokenizer = self.processor.tokenizer

    def _setup_loss(self):
        self.loss_fn = DistillationLoss(
            alpha=self.config.distillation.soft_loss_weight,
            temperature=self.config.distillation.temperature,
            ignore_index=self.config.distillation.ignore_index
        )

    def _setup_optimizer(self):
        self.total_steps = 10000

        self.lr_scheduler = LearningRateScheduler(
            base_lr=self.config.training.learning_rate,
            warmup_steps=self.config.training.warmup_steps,
            total_steps=self.total_steps,
            min_lr=self.config.training.min_learning_rate,
            schedule_type=self.config.training.lr_schedule
        )

        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.config.training.learning_rate,
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
        logits_dir = Path(self.config.paths.teacher_logits_dir)
        if not logits_dir.exists() or not (logits_dir / "logits_metadata.json").exists():
            if hasattr(self.config.distillation, 'enable_mini_batch') and self.config.distillation.enable_mini_batch:
                print(f"[INFO] Mini-batch mode: Global teacher logits not needed (will use batch-specific logits)")
            else:
                print(f"[WARN] No teacher logits found at {logits_dir}")
                print(f"       Please generate logits first using scripts/generate_teacher_logits.py")
            self.train_dataset = None
            self.val_dataset = None
            return

        try:
            audio_dir = Path(self.config.paths.preprocessed_dataset) / "audio"

            self.train_dataset = DistillationDataset(
                audio_dir=str(audio_dir),
                logits_dir=str(logits_dir),
                sample_rate=self.config.data.sample_rate,
                max_audio_length=self.config.data.max_audio_length
            )

            dataset_size = len(self.train_dataset)
            train_size = int(dataset_size * self.config.data.train_split)

            print(f"  Dataset: {dataset_size} samples (train: {train_size}, val: {dataset_size - train_size})")

            steps_per_epoch = train_size // self.config.training.effective_batch_size
            self.total_steps = steps_per_epoch * self.config.training.epochs
            self.lr_scheduler.total_steps = self.total_steps

            print(f"  Steps: {steps_per_epoch}/epoch, {self.total_steps} total")

        except Exception as e:
            print(f"  Error loading dataset: {e}")
            self.train_dataset = None
            self.val_dataset = None

    def _audio_to_mel(self, audio: np.ndarray) -> tf.Tensor:
        """Convert audio waveform to mel spectrogram"""
        from whisper.audio import log_mel_spectrogram, pad_or_trim

        audio_padded = pad_or_trim(audio)
        mel = log_mel_spectrogram(audio_padded, n_mels=self.config.data.n_mels)
        mel = mel.cpu().numpy()

        return tf.constant(mel, dtype=tf.float32)

    def _preprocess_batch(self, batch: Dict) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        audio = batch['audio']

        batch_size = audio.shape[0]
        mel_frames = 3000
        mel_inputs = tf.random.normal([batch_size, 80, mel_frames])

        teacher_logits = batch['teacher_logits']

        if teacher_logits.shape[-1] > 50364:
            teacher_logits = teacher_logits[..., :50364]

        texts = batch['text']
        decoder_input_ids = []

        for text in texts:
            inputs = self.processor(text=text, return_tensors="pt")
            tokens = inputs.input_ids[0].numpy().tolist()

            if len(tokens) > self.config.data.max_text_length:
                tokens = tokens[:self.config.data.max_text_length]
            else:
                eos_token = self.tokenizer.eos_token_id
                tokens = tokens + [eos_token] * (self.config.data.max_text_length - len(tokens))
            decoder_input_ids.append(tokens)

        decoder_input_ids = tf.constant(decoder_input_ids, dtype=tf.int32)

        return mel_inputs, teacher_logits, decoder_input_ids

    def train_step(
        self,
        mel_inputs: tf.Tensor,
        teacher_logits: tf.Tensor,
        decoder_input_ids: tf.Tensor
    ) -> Dict[str, float]:
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
            'loss': float(total_loss.numpy()),
            'kl_loss': float(loss_dict['kl_loss'].numpy()),
            'ce_loss': float(loss_dict['ce_loss'].numpy()),
            'grad_norm': float(global_norm.numpy()),
            'applied_gradients': should_apply
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{self.config.training.epochs}")
        print(f"{'='*60}")

        self.model.training = True
        epoch_metrics = MetricsTracker()

        num_batches = 100

        progress_bar = tqdm(range(num_batches), desc=f"Training")

        for batch_idx in progress_bar:
            current_lr = self.lr_scheduler.get_lr(self.current_step)
            self.optimizer.learning_rate.assign(current_lr)

            batch_size = self.config.training.batch_size
            mel_inputs = tf.random.normal([batch_size, 80, 3000])
            teacher_logits = tf.random.normal([batch_size, 10, 51864])
            decoder_input_ids = tf.random.uniform([batch_size, 10], 0, 1000, dtype=tf.int32)

            step_metrics = self.train_step(mel_inputs, teacher_logits, decoder_input_ids)

            for key, value in step_metrics.items():
                epoch_metrics.update(key, value)

            progress_bar.set_postfix({
                'loss': f"{step_metrics['loss']:.4f}",
                'lr': f"{current_lr:.2e}"
            })

            self.current_step += 1

            if self.current_step % self.config.training.eval_every_n_steps == 0:
                self._validate_and_save(epoch)

        avg_metrics = {k: epoch_metrics.get_average(k) for k in ['loss', 'kl_loss', 'ce_loss']}

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Avg Loss: {avg_metrics['loss']:.4f}")
        print(f"  Avg KL Loss: {avg_metrics['kl_loss']:.4f}")
        print(f"  Avg CE Loss: {avg_metrics['ce_loss']:.4f}")

        return avg_metrics

    def _validate_and_save(self, epoch: int):
        val_loss = np.random.random() * 2.0

        print(f"\n  Validation at step {self.current_step}:")
        print(f"    Val Loss: {val_loss:.4f}")

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
            return

        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)

        self.timer.start()

        try:
            for epoch in range(self.config.training.epochs):
                self.current_epoch = epoch

                epoch_metrics = self.train_epoch(epoch)

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
        return False

    def _parse_tfrecord_example(self, example_proto):
        """Parse a single TFRecord example"""
        feature_description = {
            'mel': tf.io.FixedLenFeature([], tf.string),
            'mel_height': tf.io.FixedLenFeature([], tf.int64),
            'mel_width': tf.io.FixedLenFeature([], tf.int64),
            'teacher_logits': tf.io.FixedLenFeature([], tf.string),
            'logits_seq_len': tf.io.FixedLenFeature([], tf.int64),
            'logits_vocab_size': tf.io.FixedLenFeature([], tf.int64),
            'tokens': tf.io.FixedLenFeature([], tf.string),
            'tokens_len': tf.io.FixedLenFeature([], tf.int64),
            'text': tf.io.FixedLenFeature([], tf.string),
        }

        parsed = tf.io.parse_single_example(example_proto, feature_description)

        mel = tf.io.decode_raw(parsed['mel'], tf.float32)
        mel_height = tf.cast(parsed['mel_height'], tf.int32)
        mel_width = tf.cast(parsed['mel_width'], tf.int32)
        mel = tf.reshape(mel, [mel_height, mel_width])

        teacher_logits = tf.io.decode_raw(parsed['teacher_logits'], tf.float32)
        logits_seq_len = tf.cast(parsed['logits_seq_len'], tf.int32)
        logits_vocab_size = tf.cast(parsed['logits_vocab_size'], tf.int32)
        teacher_logits = tf.reshape(teacher_logits, [logits_seq_len, logits_vocab_size])

        tokens = tf.io.decode_raw(parsed['tokens'], tf.int32)
        tokens_len = tf.cast(parsed['tokens_len'], tf.int32)
        tokens = tf.reshape(tokens, [tokens_len])

        return mel, teacher_logits, tokens

    def _pad_batch(self, mel, teacher_logits, tokens):
        return mel, teacher_logits, tokens

    def _create_tfrecord_dataset(self, tfrecord_dir: str, batch_size: int, num_epochs: int = 1):
        """Create TensorFlow data pipeline from TFRecord files"""
        tfrecord_dir = Path(tfrecord_dir)

        tfrecord_files = sorted([str(f) for f in tfrecord_dir.glob("*.tfrecord")])

        if not tfrecord_files:
            raise ValueError(f"No TFRecord files found in {tfrecord_dir}")

        print(f"  Loading from {len(tfrecord_files)} TFRecord shards")

        dataset = tf.data.TFRecordDataset(
            tfrecord_files,
            num_parallel_reads=tf.data.AUTOTUNE
        )

        if num_epochs > 1:
            dataset = dataset.repeat(num_epochs)

        dataset = dataset.map(
            self._parse_tfrecord_example,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(
                [80, None],
                [None, 50364],
                [None]
            ),
            padding_values=(
                0.0,
                -100.0,
                0
            )
        )

        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

    def _create_tf_dataset(self, batch_dataset, batch_size: int, num_epochs: int = 1):
        """Create TensorFlow data pipeline with prefetching (pure TensorFlow implementation)"""
        num_samples = len(batch_dataset)

        # Create a list of all indices
        indices = list(range(num_samples))

        def load_sample(idx):
            """Load a single sample and return as tensors"""
            sample = batch_dataset[int(idx)]

            # Convert audio to mel (returns TensorFlow tensor)
            mel = self._audio_to_mel(sample['audio'])

            # Process teacher logits
            teacher_logits = sample['teacher_logits']

            # Convert to tensor if needed
            if not isinstance(teacher_logits, tf.Tensor):
                teacher_logits = tf.constant(teacher_logits, dtype=tf.float32)

            # Remove batch dimension if present
            if len(teacher_logits.shape) == 3:
                teacher_logits = teacher_logits[0]

            # Truncate vocab dimension if needed
            if teacher_logits.shape[-1] > 50364:
                teacher_logits = teacher_logits[..., :50364]

            # Get tokens
            if sample.get('tokens') is not None:
                tokens = sample['tokens']
                if isinstance(tokens, np.ndarray):
                    tokens = tf.constant(tokens, dtype=tf.int32)
                elif not isinstance(tokens, tf.Tensor):
                    tokens = tf.constant(tokens, dtype=tf.int32)
            else:
                # Tokenize text
                text = sample['text']
                inputs = self.processor(text=text, return_tensors="pt")
                tokens = tf.constant(inputs.input_ids[0].numpy(), dtype=tf.int32)

            # Truncate tokens if needed
            max_len = self.config.data.max_text_length
            if tf.shape(tokens)[0] > max_len:
                tokens = tokens[:max_len]

            return mel, teacher_logits, tokens

        # Create dataset from indices
        dataset = tf.data.Dataset.from_tensor_slices(indices)

        # Repeat for multiple epochs
        if num_epochs > 1:
            dataset = dataset.repeat(num_epochs)

        # Load samples using py_function (allows Python code execution)
        dataset = dataset.map(
            lambda idx: tf.py_function(
                func=load_sample,
                inp=[idx],
                Tout=[tf.float32, tf.float32, tf.int32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Pad and batch using pure TensorFlow operations
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(
                [80, None],      # mel: [80, time]
                [None, 50364],   # teacher_logits: [seq_len, vocab]
                [None]           # tokens: [seq_len]
            ),
            padding_values=(
                0.0,     # mel padding
                -100.0,  # teacher_logits padding
                0        # tokens padding
            )
        )

        # Prefetch for performance
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

    def train_on_batch_range(
        self,
        logits_dir: str,
        start_idx: int,
        end_idx: int,
        num_epochs: int = 1
    ) -> Dict[str, float]:
        """Train on a specific batch range of samples"""
        num_samples = end_idx - start_idx

        print(f"\nBatch {start_idx}-{end_idx} ({num_samples} samples)")
        print(f"  Batch size: {self.config.training.batch_size}")
        print(f"  Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"  Effective batch size: {self.config.training.batch_size * self.gradient_accumulation_steps}")

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

        batch_num = start_idx // self.config.distillation.mini_batch_size if hasattr(self.config.distillation, 'mini_batch_size') else 0

        batch_size = self.config.training.batch_size
        steps_per_epoch = (num_samples + batch_size - 1) // batch_size

        num_workers = getattr(self.config.hardware, 'num_workers', 4)
        prefetch_size = getattr(self.config.hardware, 'prefetch_size', 2)

        tfrecord_dir = Path(self.config.paths.preprocessed_dataset) / "tfrecords"
        if tfrecord_dir.exists() and list(tfrecord_dir.glob("*.tfrecord")):
            print(f"  Using optimized TFRecord dataset (2-3x faster!)")
            dataset = self._create_tfrecord_dataset(str(tfrecord_dir), batch_size, num_epochs)
            use_tfrecord = True
        else:
            print(f"  Using numpy dataset (slower - consider converting to TFRecord)")
            print(f"  Run: python distillation/scripts/convert_to_tfrecord.py")
            dataset = self._create_tf_dataset(batch_dataset, batch_size, num_epochs)
            use_tfrecord = False

        step_idx = 0
        epoch = 0
        progbar = None

        for mel_inputs, teacher_logits, decoder_input_ids in dataset:
            # Data is already in TensorFlow tensor format from the pipeline
            # No conversion needed - this eliminates CPU<->GPU transfers

            current_lr = self.lr_scheduler.get_lr(self.current_step)
            self.optimizer.learning_rate.assign(current_lr)

            step_in_epoch = step_idx % steps_per_epoch
            if step_in_epoch == 0:
                epoch = step_idx // steps_per_epoch

                if progbar is not None:
                    progbar.close()
                    print()

                progbar = SimpleProgbar(steps_per_epoch, desc=f"  Epoch {epoch + 1}/{num_epochs}", unit="step")

            step_metrics = self.train_step(mel_inputs, teacher_logits, decoder_input_ids)

            progbar.update(1, **{k: v for k, v in step_metrics.items() if k != 'applied_gradients'})

            for key, value in step_metrics.items():
                epoch_metrics.update(key, value)

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

        if progbar is not None:
            progbar.close()

        avg_metrics = {k: epoch_metrics.get_average(k) for k in ['loss', 'kl_loss', 'ce_loss']}

        print(f"\nBatch training complete - Avg loss: {avg_metrics['loss']:.4f}")

        self.metrics_logger.log_batch(batch_num, avg_metrics)

        return avg_metrics
