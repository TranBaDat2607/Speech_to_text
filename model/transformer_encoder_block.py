"""
TensorFlow implementation of Whisper Transformer Encoder Blocks
Contains MLP and ResidualAttentionBlock for encoder
"""

import tensorflow as tf
from typing import Optional

# Import MultiHeadAttention from decoder module (unified implementation)
from transformer_decoder_block import MultiHeadAttention


class MLP(tf.keras.layers.Layer):
    """
    Feed-forward network (MLP) matching OpenAI Whisper exactly
    
    Architecture: Linear -> GELU -> Linear
    Hidden dimension: n_state * 4 (expansion factor of 4)
    """
    
    def __init__(self, n_state: int, name: str = "mlp"):
        super().__init__(name=name)
        
        self.n_state = n_state
        self.n_hidden = n_state * 4  # Expansion factor of 4, matching OpenAI
        
        # Two linear layers with GELU activation
        self.dense1 = tf.keras.layers.Dense(
            self.n_hidden, 
            activation="gelu",  # GELU, not ReLU
            name="dense1"
        )
        self.dense2 = tf.keras.layers.Dense(
            self.n_state,
            name="dense2"
        )
    
    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass of MLP
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, n_state]
            training: Whether in training mode
            
        Returns:
            tf.Tensor: Output of shape [batch_size, seq_len, n_state]
        """
        # Linear -> GELU -> Linear
        x = self.dense1(x)  # [batch, seq_len, n_state] -> [batch, seq_len, n_state*4]
        x = self.dense2(x)  # [batch, seq_len, n_state*4] -> [batch, seq_len, n_state]
        return x


class ResidualAttentionBlock(tf.keras.layers.Layer):
    """
    Complete transformer encoder block matching OpenAI Whisper exactly
    
    Architecture (Pre-norm):
    1. x = x + self_attention(layer_norm(x))
    2. x = x + mlp(layer_norm(x))
    
    Key features:
    - Pre-norm architecture (LayerNorm before operations)
    - Residual connections around both attention and MLP
    - Self-attention only (no cross-attention for AudioEncoder)
    - GELU activation in MLP
    """
    
    def __init__(self, n_state: int, n_head: int, name: str = "residual_attention_block"):
        super().__init__(name=name)
        
        self.n_state = n_state
        self.n_head = n_head
        
        # Self-attention components
        self.attn = MultiHeadAttention(n_state, n_head, name="self_attention")
        self.attn_ln = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="attn_layer_norm")
        
        # MLP components  
        self.mlp = MLP(n_state, name="mlp")
        self.mlp_ln = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="mlp_layer_norm")
        
        # Note: AudioEncoder blocks don't have cross-attention
        # Cross-attention is only used in TextDecoder blocks
    
    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass of residual attention block

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_state]
            training: Whether in training mode

        Returns:
            tf.Tensor: Output of shape [batch_size, seq_len, n_state]
        """
        # Pre-norm self-attention with residual connection
        # x = x + self_attention(layer_norm(x))
        attn_input = self.attn_ln(x, training=training)
        # MultiHeadAttention returns (output, attention_weights), we only need output
        attn_output, _ = self.attn(attn_input, xa=None, mask=None, kv_cache=None, training=training)
        x = x + attn_output

        # Pre-norm MLP with residual connection
        # x = x + mlp(layer_norm(x))
        mlp_input = self.mlp_ln(x, training=training)
        mlp_output = self.mlp(mlp_input, training=training)
        x = x + mlp_output

        return x


def create_encoder_blocks(n_state: int, n_head: int, n_layer: int) -> list:
    """
    Create encoder blocks using ResidualAttentionBlock (matching OpenAI Whisper)

    Args:
        n_state: Model dimension (embedding size)
        n_head: Number of attention heads
        n_layer: Number of transformer layers

    Returns:
        list: List of ResidualAttentionBlock layers
    """
    return [
        ResidualAttentionBlock(n_state, n_head, name=f"encoder_block_{i}")
        for i in range(n_layer)
    ]
