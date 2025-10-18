"""
TensorFlow implementation of Whisper Transformer Encoder Blocks
Contains MultiHeadAttention, MLP, and ResidualAttentionBlock matching OpenAI Whisper exactly
"""

import tensorflow as tf
from typing import Optional


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention matching OpenAI Whisper implementation exactly
    
    Key differences from standard attention:
    - Key projection has no bias (bias=False)
    - Scale factor is (head_dim ** -0.25) instead of -0.5
    - Pre-norm architecture (LayerNorm applied before attention)
    """
    
    def __init__(self, n_state: int, n_head: int, name: str = "multi_head_attention"):
        super().__init__(name=name)
        assert n_state % n_head == 0, f"n_state ({n_state}) must be divisible by n_head ({n_head})"
        
        self.n_state = n_state
        self.n_head = n_head
        self.head_dim = n_state // n_head
        self.scale = self.head_dim ** -0.25  # OpenAI uses -0.25, not -0.5
        
        # Query, Key, Value projections matching OpenAI exactly
        self.query = tf.keras.layers.Dense(n_state, use_bias=True, name="query")
        self.key = tf.keras.layers.Dense(n_state, use_bias=False, name="key")  # No bias!
        self.value = tf.keras.layers.Dense(n_state, use_bias=True, name="value")
        self.out = tf.keras.layers.Dense(n_state, use_bias=True, name="out")
        
    def call(self, x: tf.Tensor, mask: Optional[tf.Tensor] = None, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass of multi-head attention
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, n_state]
            mask: Optional attention mask
            training: Whether in training mode
            
        Returns:
            tf.Tensor: Output of shape [batch_size, seq_len, n_state]
        """
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        
        # Project to Q, K, V
        q = self.query(x)  # [batch, seq_len, n_state]
        k = self.key(x)    # [batch, seq_len, n_state] 
        v = self.value(x)  # [batch, seq_len, n_state]
        
        # Reshape for multi-head: [batch, seq_len, n_head, head_dim]
        q = tf.reshape(q, [batch_size, seq_len, self.n_head, self.head_dim])
        k = tf.reshape(k, [batch_size, seq_len, self.n_head, self.head_dim])
        v = tf.reshape(v, [batch_size, seq_len, self.n_head, self.head_dim])
        
        # Transpose to [batch, n_head, seq_len, head_dim] for attention computation
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])
        
        # Scaled dot-product attention matching OpenAI exactly
        # OpenAI: (q * scale) @ (k * scale).T = (q @ k.T) * scale^2
        attention_scores = tf.matmul(q * self.scale, k * self.scale, transpose_b=True)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores += mask
            
        # Softmax over last dimension
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        # Apply attention to values
        attention_output = tf.matmul(attention_weights, v)
        
        # Transpose back: [batch, n_head, seq_len, head_dim] -> [batch, seq_len, n_head, head_dim]
        attention_output = tf.transpose(attention_output, [0, 2, 1, 3])
        
        # Reshape to [batch, seq_len, n_state]
        attention_output = tf.reshape(attention_output, [batch_size, seq_len, self.n_state])
        
        # Final output projection
        return self.out(attention_output)


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
        attn_output = self.attn(attn_input, training=training)
        x = x + attn_output
        
        # Pre-norm MLP with residual connection
        # x = x + mlp(layer_norm(x))
        mlp_input = self.mlp_ln(x, training=training)
        mlp_output = self.mlp(mlp_input, training=training)
        x = x + mlp_output
        
        return x


def create_encoder_blocks(n_state: int, n_head: int, n_layer: int) -> list:
    """
    Create a list of ResidualAttentionBlock layers for AudioEncoder
    
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
