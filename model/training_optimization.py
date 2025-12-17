"""
Training Optimization for RTX 5060 8GB

Techniques to maximize training speed on limited GPU memory:
1. Mixed Precision (FP16) - 2x speedup, 50% less memory
2. Gradient Accumulation - simulate larger batches
3. Optimized data pipeline - eliminate bottlenecks
4. XLA compilation - faster kernels
5. Dynamic batch sizing - maximize GPU utilization
"""

import tensorflow as tf
from typing import Optional
import os


class GPUOptimizer:
    """
    Optimize TensorFlow for RTX 5060 8GB training
    """

    @staticmethod
    def configure_gpu():
        """
        Configure GPU for optimal performance

        RTX 5060 8GB optimization:
        - Enable mixed precision (FP16)
        - Set memory growth
        - Enable XLA compilation
        """
        print("Configuring GPU for RTX 5060 8GB...")

        # 1. Enable memory growth (prevent OOM)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✓ Found {len(gpus)} GPU(s)")
                print(f"✓ Memory growth enabled (prevents OOM)")
            except RuntimeError as e:
                print(e)

        # 2. Enable mixed precision (FP16) for 2x speedup
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"✓ Mixed precision enabled: {policy.name}")
        print(f"  Compute: float16 (2x faster)")
        print(f"  Variables: float32 (numerical stability)")

        # 3. Enable XLA compilation for faster kernels
        tf.config.optimizer.set_jit(True)
        print(f"✓ XLA compilation enabled")

        # 4. Set environment variables for optimal performance
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        os.environ['TF_GPU_THREAD_COUNT'] = '2'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

        print("\nGPU configuration complete!")

    @staticmethod
    def estimate_batch_size(model_params_M: float, gpu_memory_GB: float = 8) -> dict:
        """
        Estimate optimal batch size for your GPU

        Args:
            model_params_M: Model size in millions of parameters
            gpu_memory_GB: Available GPU memory in GB

        Returns:
            Dict with recommended batch sizes
        """
        print(f"\nEstimating batch size for:")
        print(f"  Model: {model_params_M:.1f}M parameters")
        print(f"  GPU: {gpu_memory_GB}GB memory")

        # Rule of thumb for Whisper-like models:
        # FP32: ~200MB per batch element
        # FP16: ~100MB per batch element

        available_mb = gpu_memory_GB * 1024 * 0.8  # Use 80% of available memory

        # FP32 estimation
        fp32_batch = int(available_mb / 200)

        # FP16 estimation (mixed precision)
        fp16_batch = int(available_mb / 100)

        recommendations = {
            'fp32_batch_size': max(1, fp32_batch),
            'fp16_batch_size': max(1, fp16_batch),
            'recommended': max(1, fp16_batch),
            'with_gradient_accumulation': max(1, fp16_batch) * 4
        }

        print(f"\nRecommended batch sizes:")
        print(f"  FP32: {recommendations['fp32_batch_size']} (not recommended, slow)")
        print(f"  FP16 (mixed precision): {recommendations['fp16_batch_size']} (recommended)")
        print(f"  FP16 + gradient accumulation: {recommendations['with_gradient_accumulation']} (effective)")

        return recommendations


class GradientAccumulator:
    """
    Gradient Accumulation for simulating larger batch sizes

    With 8GB GPU:
    - Direct batch size: 4-6
    - With accumulation: Effective batch size 16-32
    """

    def __init__(self, model, optimizer, accumulation_steps: int = 4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.gradient_accumulation = [
            tf.Variable(tf.zeros_like(var), trainable=False)
            for var in model.trainable_variables
        ]

    @tf.function
    def train_step(self, mel, tokens, targets):
        """Single training step with gradient accumulation"""
        with tf.GradientTape() as tape:
            logits = self.model(mel, tokens, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                targets, logits, from_logits=True
            )
            loss = tf.reduce_mean(loss)

            # Scale loss for gradient accumulation
            scaled_loss = loss / self.accumulation_steps

        # Compute gradients
        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)

        # Accumulate gradients
        for i, grad in enumerate(gradients):
            if grad is not None:
                self.gradient_accumulation[i].assign_add(grad)

        return loss

    def apply_gradients(self):
        """Apply accumulated gradients"""
        self.optimizer.apply_gradients(
            zip(self.gradient_accumulation, self.model.trainable_variables)
        )

        # Reset accumulation
        for grad_var in self.gradient_accumulation:
            grad_var.assign(tf.zeros_like(grad_var))


class OptimizedDataPipeline:
    """
    Optimized data pipeline to prevent GPU starvation

    Key optimizations:
    - Parallel data loading
    - Prefetching
    - Caching (if dataset fits in RAM)
    - Optimized preprocessing
    """

    @staticmethod
    def create_optimized_dataset(dataset_dir: str, batch_size: int = 4):
        """
        Create optimized TensorFlow dataset

        Args:
            dataset_dir: Path to dataset
            batch_size: Batch size for training

        Returns:
            Optimized tf.data.Dataset
        """
        from data_loader import WhisperDataLoader

        print("Creating optimized data pipeline...")

        # Load dataset
        loader = WhisperDataLoader(dataset_dir, batch_size=batch_size)
        dataset = loader.create_tf_dataset()

        # Optimization 1: Parallel preprocessing
        dataset = dataset.map(
            lambda mel, text: (mel, text),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Optimization 2: Cache if dataset fits in RAM
        # Uncomment if your dataset is small (<16GB)
        # dataset = dataset.cache()

        # Optimization 3: Shuffle
        dataset = dataset.shuffle(buffer_size=1000)

        # Optimization 4: Batch
        dataset = dataset.batch(batch_size)

        # Optimization 5: Prefetch (critical for GPU utilization!)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        print("✓ Data pipeline optimized:")
        print(f"  - Parallel preprocessing: {tf.data.AUTOTUNE}")
        print(f"  - Prefetching: {tf.data.AUTOTUNE}")
        print(f"  - Batch size: {batch_size}")

        return dataset


def create_optimized_training_config(gpu_memory_gb: float = 8):
    """
    Create optimal training configuration for RTX 5060

    Args:
        gpu_memory_gb: Available GPU memory

    Returns:
        Dict with training configuration
    """
    print("="*60)
    print("Creating Optimized Training Configuration")
    print("="*60)

    config = {
        # Mixed precision settings
        'use_mixed_precision': True,
        'loss_scale': 'dynamic',  # Prevent underflow in FP16

        # Batch size (for 8GB GPU with improved Whisper ~46M params)
        'batch_size': 6,  # FP16 batch size
        'gradient_accumulation_steps': 4,  # Effective batch = 24
        'effective_batch_size': 24,

        # Learning rate
        'initial_learning_rate': 1e-4,
        'warmup_steps': 1000,
        'total_steps': 50000,

        # Memory optimization
        'gradient_checkpointing': False,  # Not needed for 8GB with this model

        # XLA compilation
        'use_xla': True,

        # Data pipeline
        'prefetch_buffer': tf.data.AUTOTUNE,
        'num_parallel_calls': tf.data.AUTOTUNE,

        # Expected training speed
        'estimated_speed': '~200-250 samples/sec with optimizations',
        'estimated_time': '~8-12 hours for 50k steps'
    }

    print("\nOptimized configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    return config


def setup_optimized_training():
    """
    Complete setup for optimized training on RTX 5060

    Returns training-ready model and optimizer
    """
    print("\n" + "="*60)
    print("Setting Up Optimized Training")
    print("="*60)

    # 1. Configure GPU
    GPUOptimizer.configure_gpu()

    # 2. Create model
    print("\nCreating model...")
    from model import create_whisper_model
    model = create_whisper_model("base")

    # 3. Get training config
    config = create_optimized_training_config(gpu_memory_gb=8)

    # 4. Create optimizer with learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=config['initial_learning_rate'],
        decay_steps=config['total_steps'],
        warmup_target=config['initial_learning_rate'],
        warmup_steps=config['warmup_steps']
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # 5. Compile model
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )

    print("\n✓ Training setup complete!")
    print(f"✓ Batch size: {config['batch_size']}")
    print(f"✓ Effective batch size: {config['effective_batch_size']} (with accumulation)")
    print(f"✓ Mixed precision: Enabled")
    print(f"✓ XLA: Enabled")

    return model, optimizer, config


if __name__ == "__main__":
    # Test GPU configuration
    GPUOptimizer.configure_gpu()

    # Estimate batch size
    optimizer = GPUOptimizer()
    batch_info = optimizer.estimate_batch_size(model_params_M=46.0, gpu_memory_GB=8)

    print("\n" + "="*60)
    print("Ready to train on RTX 5060 8GB!")
    print("="*60)
    print("\nUse this configuration in your training script:")
    print(f"  batch_size={batch_info['recommended']}")
    print(f"  gradient_accumulation_steps=4")
    print(f"  mixed_precision=True")
