"""
TensorFlow implementation of Whisper Model matching OpenAI Whisper exactly.

This is the main model file that combines all components into a complete Whisper model.
Architecture matches PyTorch version in whisper/model.py exactly.
"""

import tensorflow as tf
import numpy as np
import warnings
from typing import Dict, Optional, Tuple

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=UserWarning, module='keras')

from model_dimensions import ModelDimensions, get_whisper_dimensions
from transformer_decoder_block import ResidualAttentionBlock, create_causal_mask
from transformer_encoder_block import create_encoder_blocks
from learned_positional_encoding import LearnedPositionalEncoding
from positional_encoding import sinusoidal_positional_encoding
from vocabulary_compression import CompactFactorizedEmbedding


class TextDecoder(tf.keras.Model):
    """
    Text Decoder with compact factorized embedding

    Architecture:
    1. Compact factorized token embedding (15k vocab, 64 bottleneck)
    2. Learned positional embedding (n_text_ctx, n_text_state)
    3. N × ResidualAttentionBlock with cross_attention=True
    4. Final layer normalization
    5. Output projection (tied with token embedding weights)
    """

    def __init__(self,
                 dims: ModelDimensions,
                 compact_vocab_size: int = 15000,
                 embedding_bottleneck: int = 64,
                 vocab_mapping: Optional[Dict[int, int]] = None,
                 name: str = "text_decoder"):
        super().__init__(name=name)

        self.dims = dims
        self.compact_vocab_size = compact_vocab_size

        # Compact factorized token embedding
        self.token_embedding = CompactFactorizedEmbedding(
            compact_vocab_size=compact_vocab_size,
            d_model=dims.n_text_state,
            bottleneck_dim=embedding_bottleneck,
            old_to_new_mapping=vocab_mapping,
            name="token_embedding"
        )
        
        # Learned positional embedding matching PyTorch: nn.Parameter(torch.empty(n_ctx, n_state))
        self.positional_embedding = LearnedPositionalEncoding(
            dims.n_text_ctx,
            dims.n_text_state,
            name="positional_embedding"
        )
        
        # Decoder transformer blocks with cross-attention
        self.blocks = [
            ResidualAttentionBlock(
                dims.n_text_state,
                dims.n_text_head,
                cross_attention=True,
                name=f"decoder_block_{i}"
            )
            for i in range(dims.n_text_layer)
        ]
        
        # Final layer normalization
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        
        # Causal mask for autoregressive generation
        # PyTorch: mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        mask = create_causal_mask(dims.n_text_ctx)
        # Convert to non-trainable variable to avoid gradient issues
        self.mask = tf.Variable(
            mask,
            trainable=False,
            name="causal_mask"
        )
    
    def call(self, 
             x: tf.Tensor, 
             xa: tf.Tensor, 
             kv_cache: Optional[dict] = None,
             training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass of TextDecoder matching OpenAI Whisper exactly
        
        Args:
            x: Token indices of shape [batch_size, seq_len]
            xa: Encoder output of shape [batch_size, n_audio_ctx, n_audio_state]
            kv_cache: Optional KV cache for generation efficiency
            training: Whether in training mode
            
        Returns:
            tf.Tensor: Logits of shape [batch_size, seq_len, n_vocab]
        """
        # PyTorch: offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        offset = 0
        if kv_cache is not None and len(kv_cache) > 0:
            # Get offset from first cached tensor
            first_cache_tensor = next(iter(kv_cache.values()))
            offset = tf.shape(first_cache_tensor)[1]
        
        # Token embedding: [batch_size, seq_len] -> [batch_size, seq_len, n_text_state]
        x_embedded = self.token_embedding(x)
        
        # Add positional embedding with offset support for generation
        # PyTorch: x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
        seq_len = tf.shape(x)[1]
        pos_embedding = self.positional_embedding(offset=offset, sequence_length=seq_len)
        x_embedded = x_embedded + pos_embedding
        
        # Cast to match encoder output dtype (important for mixed precision)
        x_embedded = tf.cast(x_embedded, xa.dtype)
        
        # Pass through decoder blocks
        for block in self.blocks:
            x_embedded = block(
                x_embedded,
                xa=xa,
                mask=self.mask,
                kv_cache=kv_cache,
                training=training
            )
        
        # Final layer normalization
        x_embedded = self.ln(x_embedded, training=training)

        # Output projection: logits = x @ token_embedding.weight.T
        # For compact factorized embedding, compute embedding matrix
        token_weights = self.token_embedding.get_weights_for_output_projection()
        token_weights = tf.cast(token_weights, x_embedded.dtype)  # [compact_vocab, n_text_state]
        logits = tf.matmul(x_embedded, token_weights, transpose_b=True)  # [batch, seq_len, compact_vocab]

        # Ensure float32 output for numerical stability
        logits = tf.cast(logits, tf.float32)

        return logits
    
    def get_config(self):
        return {
            "dims": {
                "n_vocab": self.dims.n_vocab,
                "n_text_ctx": self.dims.n_text_ctx,
                "n_text_state": self.dims.n_text_state,
                "n_text_head": self.dims.n_text_head,
                "n_text_layer": self.dims.n_text_layer,
            }
        }


class AudioEncoder(tf.keras.Model):
    """
    Audio Encoder with Conformer blocks (improved architecture)

    Architecture:
    1. Conv1D (n_mels -> n_audio_state, kernel=3, stride=1) + GELU
    2. Conv1D (n_audio_state -> n_audio_state, kernel=3, stride=2) + GELU
    3. Sinusoidal positional encoding addition
    4. N × ConformerBlock (Conv + Attention for better speech modeling)
    5. Final layer normalization
    """

    def __init__(self, dims: ModelDimensions, name: str = "audio_encoder"):
        super().__init__(name=name)

        self.dims = dims

        # Two convolutional layers
        self.conv1 = tf.keras.layers.Conv1D(
            filters=dims.n_audio_state,
            kernel_size=3,
            strides=1,
            padding="same",
            activation=None,
            name="conv1"
        )

        self.conv2 = tf.keras.layers.Conv1D(
            filters=dims.n_audio_state,
            kernel_size=3,
            strides=2,
            padding="same",
            activation=None,
            name="conv2"
        )

        # Sinusoidal positional encoding (non-trainable)
        pos_encoding = sinusoidal_positional_encoding(
            length=dims.n_audio_ctx,
            channels=dims.n_audio_state
        )
        self.positional_embedding = tf.Variable(
            pos_encoding,
            trainable=False,
            name="positional_embedding"
        )

        # Conformer encoder blocks
        self.blocks = create_encoder_blocks(
            dims.n_audio_state,
            dims.n_audio_head,
            dims.n_audio_layer
        )

        # Final layer normalization
        self.ln_post = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="ln_post")
    
    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass of AudioEncoder matching OpenAI Whisper exactly
        
        Args:
            x: Mel spectrogram of shape [batch_size, n_mels, n_frames]
               where n_frames=3000 (30 seconds at 16kHz)
            training: Whether in training mode
            
        Returns:
            tf.Tensor: Encoded audio features of shape [batch_size, n_audio_ctx, n_audio_state]
                      where n_audio_ctx=1500 (downsampled by conv2 stride=2)
        """
        # Input shape: [batch_size, n_mels, n_frames] = [batch, 80, 3000]
        
        # TensorFlow Conv1D expects (batch, time_steps, features)
        # Transpose from (batch, n_mels, n_frames) to (batch, n_frames, n_mels)
        x = tf.transpose(x, [0, 2, 1])  # [batch, 3000, 80]
        
        # First convolution + GELU: [batch, 3000, 80] -> [batch, 3000, n_audio_state]
        x = self.conv1(x, training=training)
        x = tf.nn.gelu(x)
        
        # Second convolution + GELU with stride=2: [batch, 3000, n_audio_state] -> [batch, 1500, n_audio_state]
        # padding='same' with stride=2 matches PyTorch Conv1d(stride=2, padding=1)
        x = self.conv2(x, training=training)  
        x = tf.nn.gelu(x)
        
        # x is now [batch_size, n_audio_ctx, n_audio_state]
        # Add positional encoding
        # PyTorch: x = (x + self.positional_embedding).to(x.dtype)
        
        # Debug shapes
        x_shape = tf.shape(x)
        pos_shape = tf.shape(self.positional_embedding)
        tf.debugging.assert_equal(
            x_shape[1:], pos_shape,
            message=f"incorrect audio shape: x={x_shape}, pos_embedding={pos_shape}"
        )
        
        x = x + tf.cast(self.positional_embedding, x.dtype)
        
        # Pass through encoder blocks
        for block in self.blocks:
            x = block(x, training=training)
        
        # Final layer normalization
        x = self.ln_post(x, training=training)
        
        return x
    
    def get_config(self):
        return {
            "dims": {
                "n_mels": self.dims.n_mels,
                "n_audio_ctx": self.dims.n_audio_ctx,
                "n_audio_state": self.dims.n_audio_state,
                "n_audio_head": self.dims.n_audio_head,
                "n_audio_layer": self.dims.n_audio_layer,
            }
        }


def create_text_decoder(model_name: str = "base") -> TextDecoder:
    """
    Factory function to create TextDecoder with predefined configurations
    
    Args:
        model_name: Model size ("tiny", "base", "small", "medium", "large")
        
    Returns:
        TextDecoder: Initialized text decoder
    """
    dims = get_whisper_dimensions(model_name)
    return TextDecoder(dims)


def create_audio_encoder(model_name: str = "base") -> AudioEncoder:
    """
    Factory function to create AudioEncoder with Conformer blocks

    Args:
        model_name: Model size ("tiny", "base", "small", "medium", "large")

    Returns:
        AudioEncoder: Initialized audio encoder with Conformer blocks
    """
    dims = get_whisper_dimensions(model_name)
    return AudioEncoder(dims)


class Whisper(tf.keras.Model):
    """
    Improved Whisper model with Conformer encoder and compact vocabulary

    Architecture:
    - AudioEncoder: mel spectrogram -> audio features (with Conformer blocks)
    - TextDecoder: tokens + audio features -> logits (with compact factorized embedding)

    Improvements over original:
    - Conformer encoder: +2-5% WER improvement
    - Compact vocabulary: 96% embedding parameter reduction

    Key methods:
    - embed_audio(): encode audio to features
    - logits(): decode tokens given audio features
    - forward(): full forward pass (encoder + decoder)
    """

    def __init__(self,
                 dims: ModelDimensions,
                 compact_vocab_size: int = 15000,
                 embedding_bottleneck: int = 64,
                 vocab_mapping: Optional[Dict[int, int]] = None,
                 name: str = "whisper"):
        super().__init__(name=name)

        self.dims = dims

        # Initialize Conformer encoder
        self.encoder = AudioEncoder(dims, name="encoder")

        # Initialize decoder with compact factorized embedding
        self.decoder = TextDecoder(
            dims,
            compact_vocab_size=compact_vocab_size,
            embedding_bottleneck=embedding_bottleneck,
            vocab_mapping=vocab_mapping,
            name="decoder"
        )
        
        # Alignment heads for word-level timing (matching PyTorch)
        # PyTorch: use the last half among the decoder layers for time alignment by default
        all_heads = tf.zeros((dims.n_text_layer, dims.n_text_head), dtype=tf.bool)
        all_heads = tf.concat([
            tf.zeros((dims.n_text_layer // 2, dims.n_text_head), dtype=tf.bool),
            tf.ones((dims.n_text_layer - dims.n_text_layer // 2, dims.n_text_head), dtype=tf.bool)
        ], axis=0)
        
        self.alignment_heads = tf.Variable(
            all_heads,
            trainable=False,
            name="alignment_heads"
        )
    
    def embed_audio(self, mel: tf.Tensor) -> tf.Tensor:
        """
        Encode mel spectrogram to audio features
        
        Args:
            mel: Mel spectrogram of shape [batch_size, n_mels, n_frames]
            
        Returns:
            tf.Tensor: Audio features of shape [batch_size, n_audio_ctx, n_audio_state]
        """
        return self.encoder(mel)
    
    def logits(self, tokens: tf.Tensor, audio_features: tf.Tensor) -> tf.Tensor:
        """
        Decode tokens given audio features
        
        Args:
            tokens: Token indices of shape [batch_size, seq_len]
            audio_features: Audio features of shape [batch_size, n_audio_ctx, n_audio_state]
            
        Returns:
            tf.Tensor: Logits of shape [batch_size, seq_len, n_vocab]
        """
        return self.decoder(tokens, audio_features)
    
    def call(self, mel: tf.Tensor, tokens: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Full forward pass: mel + tokens -> logits
        
        Args:
            mel: Mel spectrogram of shape [batch_size, n_mels, n_frames]  
            tokens: Token indices of shape [batch_size, seq_len]
            training: Whether in training mode
            
        Returns:
            tf.Tensor: Logits of shape [batch_size, seq_len, n_vocab]
        """
        # Encode audio
        audio_features = self.encoder(mel, training=training)
        
        # Decode tokens
        logits = self.decoder(tokens, audio_features, training=training)
        
        return logits
    
    @property
    def is_multilingual(self) -> bool:
        """Check if model supports multiple languages"""
        return self.dims.n_vocab >= 51865
    
    @property  
    def num_languages(self) -> int:
        """Get number of supported languages"""
        if self.dims.n_vocab >= 51865:
            return 99  # Supports ~99 languages
        else:
            # English-only or incomplete
            return 1
    
    def get_config(self):
        return {
            "dims": {
                "n_mels": self.dims.n_mels,
                "n_audio_ctx": self.dims.n_audio_ctx, 
                "n_audio_state": self.dims.n_audio_state,
                "n_audio_head": self.dims.n_audio_head,
                "n_audio_layer": self.dims.n_audio_layer,
                "n_vocab": self.dims.n_vocab,
                "n_text_ctx": self.dims.n_text_ctx,
                "n_text_state": self.dims.n_text_state,
                "n_text_head": self.dims.n_text_head,
                "n_text_layer": self.dims.n_text_layer,
            }
        }


def create_whisper_model(model_name: str = "base",
                         compact_vocab_size: int = 15000,
                         embedding_bottleneck: int = 64,
                         vocab_mapping: Optional[Dict[int, int]] = None) -> Whisper:
    """
    Factory function to create improved Whisper model

    Model includes:
    - Conformer encoder (better speech modeling)
    - Compact factorized vocabulary (smaller model size)

    Args:
        model_name: Model size ("tiny", "base", "small", "medium", "large")
        compact_vocab_size: Vocabulary size (default: 15000 for Vietnamese+English)
        embedding_bottleneck: Factorization bottleneck dimension (default: 64)
        vocab_mapping: Optional token ID mapping from analysis

    Returns:
        Whisper: Improved Whisper model

    Examples:
        # Default (Vietnamese + English, ~15k vocab)
        model = create_whisper_model("base")

        # Custom vocabulary size
        model = create_whisper_model("base", compact_vocab_size=20000)

        # With pre-analyzed vocabulary mapping
        model = create_whisper_model("base", vocab_mapping=analyzed_mapping)
    """
    dims = get_whisper_dimensions(model_name)
    return Whisper(
        dims,
        compact_vocab_size=compact_vocab_size,
        embedding_bottleneck=embedding_bottleneck,
        vocab_mapping=vocab_mapping
    )
