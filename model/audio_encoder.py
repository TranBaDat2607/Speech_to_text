"""
TensorFlow implementation of Whisper Audio Preprocessing Layers
Contains only the 2 × Conv1D + GELU layers from AudioEncoder
"""

import tensorflow as tf
from typing import Optional
from model_dimensions import ModelDimensions


class AudioConvLayers(tf.keras.Model):
    """
    Whisper Audio Convolutional Layers - 2 × Conv1D + GELU
    
    Converts mel spectrograms through two convolutional layers matching OpenAI Whisper:
    1. Conv1D: n_mels -> n_audio_state, kernel=3, stride=1, padding=same + GELU
    2. Conv1D: n_audio_state -> n_audio_state, kernel=3, stride=2, padding=same + GELU
    """
    
    def __init__(self, dims: ModelDimensions, name: str = "audio_conv_layers"):
        super().__init__(name=name)
        
        self.dims = dims
        
        # First 1D convolutional layer matching OpenAI Whisper
        # After transpose: (batch, n_frames, n_mels) -> (batch, n_frames, n_audio_state)
        self.conv1 = tf.keras.layers.Conv1D(
            filters=dims.n_audio_state,
            kernel_size=3,
            strides=1,
            padding="same",  # equivalent to padding=1 in PyTorch
            activation="gelu",
            name="conv1"
        )
        
        # Second 1D convolutional layer with stride=2 for downsampling
        # (batch, n_frames, n_audio_state) -> (batch, n_frames//2, n_audio_state)
        self.conv2 = tf.keras.layers.Conv1D(
            filters=dims.n_audio_state,
            kernel_size=3,
            strides=2, 
            padding="same", 
            activation="gelu", 
            name="conv2"
        )
    
    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass of the convolutional layers
        
        Args:
            x: Mel spectrogram tensor of shape [batch_size, n_mels, n_frames]
               where n_mels=80 and n_frames=3000 (30 seconds at 16kHz)
            training: Whether in training mode
            
        Returns:
            tf.Tensor: Convolved features of shape [batch_size, n_audio_state, n_frames//2]
                      where n_frames//2=1500 (downsampled by conv2 stride=2)
        """
        # Input shape: [batch_size, n_mels, n_frames] = [batch, 80, 3000]
        
        # TensorFlow Conv1D expects (batch, time_steps, features)
        # Transpose from (batch, n_mels, n_frames) to (batch, n_frames, n_mels)
        x = tf.transpose(x, [0, 2, 1])  # [batch, 3000, 80]
        
        # First convolution + GELU: [batch, 3000, 80] -> [batch, 3000, n_audio_state]
        x = self.conv1(x, training=training)
        
        # Second convolution + GELU with stride=2: [batch, 3000, n_audio_state] -> [batch, 1500, n_audio_state]  
        x = self.conv2(x, training=training)
        
        # Transpose back to match Whisper format: [batch, 1500, n_audio_state] -> [batch, n_audio_state, 1500]
        x = tf.transpose(x, [0, 2, 1])  # [batch, n_audio_state, 1500]
        
        return x
    
    def get_config(self):
        return {
            "dims": {
                "n_mels": self.dims.n_mels,
                "n_audio_state": self.dims.n_audio_state,
            }
        }


def create_audio_conv_layers(model_name: str = "base") -> AudioConvLayers:
    """
    Factory function to create AudioConvLayers with predefined configurations
    
    Args:
        model_name: Model size ("tiny", "base", "small", "medium", "large")
        
    Returns:
        AudioConvLayers: Initialized audio convolutional layers
    """
    from .model_dimensions import get_whisper_dimensions
    
    dims = get_whisper_dimensions(model_name)
    return AudioConvLayers(dims)
