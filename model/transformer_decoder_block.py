"""
TensorFlow implementation of Whisper Transformer Decoder Blocks
Contains CrossAttention, DecoderBlocks with causal masking matching OpenAI Whisper exactly
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Tuple


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention matching OpenAI Whisper implementation exactly
    
    Key features:
    - Self-attention: Q=K=V=x (decoder input)
    - Cross-attention: Q=x (decoder), K=V=xa (encoder output)
    - Key projection has no bias (bias=False)
    - Scale factor is (n_state // n_head) ** -0.25
    - KV cache support for efficient generation
    - Separated qkv_attention method like PyTorch
    """
    
    def __init__(self, n_state: int, n_head: int, name: str = "multi_head_attention"):
        super().__init__(name=name)
        assert n_state % n_head == 0, f"n_state ({n_state}) must be divisible by n_head ({n_head})"
        
        # Matching PyTorch exactly - only store n_head, not n_state
        self.n_head = n_head
        
        # Query, Key, Value projections matching OpenAI exactly
        self.query = tf.keras.layers.Dense(n_state, use_bias=True, name="query")
        self.key = tf.keras.layers.Dense(n_state, use_bias=False, name="key")  # No bias!
        self.value = tf.keras.layers.Dense(n_state, use_bias=True, name="value")
        self.out = tf.keras.layers.Dense(n_state, use_bias=True, name="out")
        
    def call(self, 
             x: tf.Tensor, 
             xa: Optional[tf.Tensor] = None, 
             mask: Optional[tf.Tensor] = None, 
             kv_cache: Optional[dict] = None,
             training: Optional[bool] = None) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        """
        Forward pass matching OpenAI Whisper MultiHeadAttention exactly
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, n_state]
            xa: Encoder output tensor [batch_size, enc_seq_len, n_state] (for cross-attention)
            mask: Optional attention mask
            kv_cache: Optional KV cache for generation efficiency
            training: Whether in training mode
            
        Returns:
            Tuple[tf.Tensor, Optional[tf.Tensor]]: (output, attention_weights)
        """
        # Store original dtype to preserve it
        original_dtype = x.dtype
        
        # Always compute Query from input x
        q = self.query(x)
        
        # Cast query to original dtype to handle Dense layer promotion
        q = tf.cast(q, original_dtype)
        
        # PyTorch: if kv_cache is None or xa is None or self.key not in kv_cache:
        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            kv_input = x if xa is None else xa
            k = self.key(kv_input)
            v = self.value(kv_input)
            
            # Cast K,V to original dtype to handle Dense layer promotion  
            k = tf.cast(k, original_dtype)
            v = tf.cast(v, original_dtype)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]
        
        # Call qkv_attention method like PyTorch: wv, qk = self.qkv_attention(q, k, v, mask)
        wv, qk = self.qkv_attention(q, k, v, mask)
        
        # Final output projection: return self.out(wv), qk
        output = self.out(wv)

        # Ensure output matches original input dtype (critical for mixed precision)
        output = tf.cast(output, original_dtype)

        return output, qk
    
    def qkv_attention(self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, mask: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        """
        QKV attention computation matching OpenAI Whisper exactly
        
        Args:
            q: Query tensor
            k: Key tensor  
            v: Value tensor
            mask: Optional attention mask
            
        Returns:
            Tuple[tf.Tensor, Optional[tf.Tensor]]: (attention_output, attention_weights)
        """
        # PyTorch: n_batch, n_ctx, n_state = q.shape
        n_batch = tf.shape(q)[0]
        n_ctx = tf.shape(q)[1] 
        n_state = tf.shape(q)[2]
        
        # PyTorch: scale = (n_state // self.n_head) ** -0.25
        # Convert n_state to float for power calculation
        n_state_float = tf.cast(n_state, tf.float32)
        head_dim_float = n_state_float / tf.cast(self.n_head, tf.float32)
        scale = tf.cast(head_dim_float ** -0.25, q.dtype)
        
        # PyTorch: q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        head_dim = n_state // self.n_head
        q = tf.reshape(q, [n_batch, n_ctx, self.n_head, head_dim])
        k = tf.reshape(k, [tf.shape(k)[0], tf.shape(k)[1], self.n_head, head_dim])
        v = tf.reshape(v, [tf.shape(v)[0], tf.shape(v)[1], self.n_head, head_dim])
        
        # permute(0, 2, 1, 3) = transpose([0, 2, 1, 3])
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])
        
        # PyTorch: qk = (q * scale) @ (k * scale).transpose(-1, -2)
        qk = tf.matmul(q * scale, k * scale, transpose_b=True)
        
        # PyTorch: if mask is not None: qk = qk + mask[:n_ctx, :n_ctx]
        if mask is not None:
            kv_seq_len = tf.shape(k)[2]  # k shape: [batch, n_head, kv_seq_len, head_dim]
            cropped_mask = mask[:n_ctx, :kv_seq_len]
            qk = qk + cropped_mask
            
        # PyTorch: qk = qk.float(); w = F.softmax(qk, dim=-1).to(q.dtype)
        qk_float = tf.cast(qk, tf.float32)
        w = tf.nn.softmax(qk_float, axis=-1)
        w = tf.cast(w, q.dtype)
        
        # PyTorch: out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        out = tf.matmul(w, v)  # [batch, n_head, seq_len, head_dim]
        out = tf.transpose(out, [0, 2, 1, 3])  # [batch, seq_len, n_head, head_dim]
        out = tf.reshape(out, [n_batch, n_ctx, n_state])  # flatten(start_dim=2)
        
        # Ensure output matches input dtype (important for float16)
        out = tf.cast(out, q.dtype)
        
        # PyTorch: qk = qk.detach()
        qk_detached = tf.stop_gradient(qk)
        
        return out, qk_detached


def create_causal_mask(n_ctx: int) -> tf.Tensor:
    """
    Create causal attention mask matching OpenAI Whisper exactly
    
    PyTorch equivalent:
    ```python
    mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
    ```
    
    Args:
        n_ctx: Maximum context length
        
    Returns:
        tf.Tensor: Causal mask of shape [n_ctx, n_ctx]
    """
    # Create upper triangular matrix with 1s above diagonal
    mask = tf.linalg.band_part(tf.ones((n_ctx, n_ctx)), 0, -1)  # Upper triangular
    mask = mask - tf.linalg.band_part(mask, 0, 0)  # Remove diagonal, keep upper tri
    
    # Replace 1s with -inf to mask future tokens
    mask = tf.where(mask == 1, -np.inf, 0.0)
    
    return tf.cast(mask, tf.float32)


class DecoderMLP(tf.keras.layers.Layer):
    """
    Feed-forward network (MLP) for decoder blocks - identical to encoder MLP
    
    Architecture: Linear -> GELU -> Linear
    Hidden dimension: n_state * 4 (expansion factor of 4)
    """
    
    def __init__(self, n_state: int, name: str = "decoder_mlp"):
        super().__init__(name=name)
        
        self.n_state = n_state
        self.n_hidden = n_state * 4 

        self.dense1 = tf.keras.layers.Dense(
            self.n_hidden, 
            activation=None,  # No activation - will apply GELU separately
            name="dense1"
        )
        self.dense2 = tf.keras.layers.Dense(
            self.n_state,
            activation=None,
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
        # Linear -> GELU -> Linear matching PyTorch exactly
        x = self.dense1(x)  # [batch, seq_len, n_state] -> [batch, seq_len, n_state*4]
        x = tf.nn.gelu(x)   # Apply GELU separately like PyTorch nn.GELU()
        x = self.dense2(x)  # [batch, seq_len, n_state*4] -> [batch, seq_len, n_state]
        return x


class ResidualAttentionBlock(tf.keras.layers.Layer):
    """
    Complete transformer block matching OpenAI Whisper exactly
    
    This is the SAME block used for both AudioEncoder and TextDecoder
    Controlled by cross_attention parameter:
    - cross_attention=False: AudioEncoder blocks (self-attention + MLP only)
    - cross_attention=True: TextDecoder blocks (self-attention + cross-attention + MLP)
    
    Architecture (Pre-norm):
    1. x = x + self_attention(layer_norm(x), mask=mask)
    2. x = x + cross_attention(layer_norm(x), xa=encoder_output) [if cross_attention=True]
    3. x = x + mlp(layer_norm(x))
    """
    
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False, name: str = "residual_attention_block"):
        super().__init__(name=name)
        
        self.n_state = n_state
        self.n_head = n_head
        self.cross_attention = cross_attention
        
        # Self-attention components (always present)
        self.attn = MultiHeadAttention(n_state, n_head, name="self_attention")
        self.attn_ln = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="attn_layer_norm")
        
        # Cross-attention components (only if cross_attention=True)
        # Matching PyTorch: self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn = MultiHeadAttention(n_state, n_head, name="cross_attention") if cross_attention else None
        self.cross_attn_ln = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="cross_attn_layer_norm") if cross_attention else None
        
        # MLP components  
        self.mlp = DecoderMLP(n_state, name="mlp")
        self.mlp_ln = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="mlp_layer_norm")
    
    def call(self, 
             x: tf.Tensor, 
             xa: Optional[tf.Tensor] = None, 
             mask: Optional[tf.Tensor] = None, 
             kv_cache: Optional[dict] = None,
             training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass matching OpenAI Whisper ResidualAttentionBlock exactly
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, n_state]
            xa: Encoder output tensor [batch_size, enc_seq_len, n_state] (None for AudioEncoder)
            mask: Attention mask (causal mask for TextDecoder, None for AudioEncoder)
            kv_cache: Optional KV cache for generation efficiency
            training: Whether in training mode
            
        Returns:
            tf.Tensor: Output of shape [batch_size, seq_len, n_state]
        """
        # 1. Self-attention with residual connection
        attn_input = self.attn_ln(x, training=training)
        attn_output, _ = self.attn(attn_input, xa=None, mask=mask, kv_cache=kv_cache, training=training)
        x = x + attn_output
        
        # 2. Cross-attention (only if cross_attention=True)
        if self.cross_attn is not None:
            cross_attn_input = self.cross_attn_ln(x, training=training)
            cross_attn_output, _ = self.cross_attn(cross_attn_input, xa=xa, kv_cache=kv_cache, training=training)
            x = x + cross_attn_output
        
        # 3. MLP with residual connection
        mlp_input = self.mlp_ln(x, training=training)
        mlp_output = self.mlp(mlp_input, training=training)
        x = x + mlp_output
        
        return x


def create_decoder_blocks(n_state: int, n_head: int, n_layer: int) -> list:
    """
    Create a list of ResidualAttentionBlock layers for TextDecoder
    
    Args:
        n_state: Model dimension (embedding size)
        n_head: Number of attention heads
        n_layer: Number of transformer layers
        
    Returns:
        list: List of ResidualAttentionBlock layers with cross_attention=True
    """
    return [
        ResidualAttentionBlock(n_state, n_head, cross_attention=True, name=f"decoder_block_{i}")
        for i in range(n_layer)
    ]


def create_encoder_blocks(n_state: int, n_head: int, n_layer: int) -> list:
    """
    Create a list of ResidualAttentionBlock layers for AudioEncoder
    
    Args:
        n_state: Model dimension (embedding size)
        n_head: Number of attention heads
        n_layer: Number of transformer layers
        
    Returns:
        list: List of ResidualAttentionBlock layers with cross_attention=False
    """
    return [
        ResidualAttentionBlock(n_state, n_head, cross_attention=False, name=f"encoder_block_{i}")
        for i in range(n_layer)
    ]


class CausalMask:
    """
    Utility class for managing causal masks
    """
    
    def __init__(self, n_ctx: int):
        self.n_ctx = n_ctx
        self._mask = None
    
    @property 
    def mask(self) -> tf.Tensor:
        if self._mask is None:
            self._mask = create_causal_mask(self.n_ctx)
        return self._mask
    
    def get_mask(self, seq_len: int) -> tf.Tensor:
        """Get causal mask cropped to sequence length"""
        return self.mask[:seq_len, :seq_len]
