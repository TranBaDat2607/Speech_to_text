"""
TensorFlow implementation of Sinusoidal Positional Encoding
Matching OpenAI Whisper sinusoids function exactly
"""

import tensorflow as tf
import numpy as np


def sinusoidal_positional_encoding(length: int, channels: int, max_timescale: float = 10000.0) -> tf.Tensor:
    """
    Generate sinusoidal positional encodings matching OpenAI Whisper exactly
    
    This function replicates the exact logic from OpenAI Whisper's sinusoids function:
    ```python
    def sinusoids(length, channels, max_timescale=10000):
        assert channels % 2 == 0
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    ```
    
    Args:
        length: Sequence length (number of time steps)
        channels: Model dimension (must be even)
        max_timescale: Maximum timescale for encoding (default: 10000.0)
        
    Returns:
        tf.Tensor: Positional encodings of shape [length, channels]
    """
    assert channels % 2 == 0, f"channels must be even, got {channels}"

    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)

    # torch.arange(channels // 2) -> tf.range(channels // 2)
    # torch.exp(-log_timescale_increment * x) -> tf.exp(-log_timescale_increment * x)
    inv_timescales = tf.exp(-log_timescale_increment * tf.cast(tf.range(channels // 2), tf.float32))

    # torch.arange(length) -> tf.range(length)
    positions = tf.cast(tf.range(length), tf.float32)

    # torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    # becomes: positions[:, tf.newaxis] * inv_timescales[tf.newaxis, :]
    scaled_time = positions[:, tf.newaxis] * inv_timescales[tf.newaxis, :]

    # torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    # becomes: tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    pos_encoding = tf.concat([
        tf.sin(scaled_time),
        tf.cos(scaled_time)
    ], axis=1)
    
    return pos_encoding


def create_whisper_positional_encoding(n_audio_ctx: int, n_audio_state: int) -> tf.Tensor:
    """
    Create positional encoding specifically for Whisper AudioEncoder
    
    Args:
        n_audio_ctx: Audio context length (typically 1500 after conv downsampling)
        n_audio_state: Audio encoder embedding dimension (e.g., 384, 512, 768, 1024, 1280)
        
    Returns:
        tf.Tensor: Positional encodings of shape [n_audio_ctx, n_audio_state]
    """
    return sinusoidal_positional_encoding(
        length=n_audio_ctx,
        channels=n_audio_state,
        max_timescale=10000.0
    )
