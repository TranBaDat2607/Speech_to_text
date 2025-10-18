"""
TensorFlow implementation of Learned Positional Encoding
For Whisper TextDecoder - trainable positional embeddings
"""

import tensorflow as tf


class LearnedPositionalEncoding(tf.keras.layers.Layer):
    """
    Learned positional encoding for TextDecoder
    Equivalent to nn.Parameter(torch.empty(n_ctx, n_state)) in PyTorch
    
    This matches the TextDecoder implementation in OpenAI Whisper:
    ```python
    self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))
    
    # Usage in forward:
    x = (
        self.token_embedding(x)
        + self.positional_embedding[offset : offset + x.shape[-1]]
    )
    ```
    """
    
    def __init__(self, n_ctx: int, n_state: int, **kwargs):
        """
        Initialize learned positional encoding
        
        Args:
            n_ctx: Maximum context length (e.g., 448 for text)
            n_state: Model dimension (e.g., 384, 512, 768, 1024, 1280)
        """
        super().__init__(**kwargs)
        self.n_ctx = n_ctx
        self.n_state = n_state
        
    def build(self, input_shape):
        # Create learnable positional embeddings
        # Matching PyTorch nn.Parameter(torch.empty(n_ctx, n_state))
        self.positional_embedding = self.add_weight(
            name='positional_embedding',
            shape=(self.n_ctx, self.n_state),
            initializer='random_normal',
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, offset: int = 0, sequence_length: int = None):
        """
        Get positional embeddings for given offset and sequence length
        
        Args:
            offset: Starting position (for kv_cache support)
            sequence_length: Length of sequence to get embeddings for
            
        Returns:
            tf.Tensor: Positional embeddings of shape [sequence_length, n_state]
        """
        if sequence_length is None:
            return self.positional_embedding
        
        end_pos = offset + sequence_length
        return self.positional_embedding[offset:end_pos]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'n_ctx': self.n_ctx,
            'n_state': self.n_state,
        })
        return config