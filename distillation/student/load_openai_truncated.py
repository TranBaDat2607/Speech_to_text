"""
Load OpenAI pretrained weights with vocab truncation for PhoWhisper compatibility
Truncates decoder embedding from 51865 → 50364 tokens
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Optional

# Import OpenAI weight loader
from .load_openai_pretrained import download_openai_weights


def load_openai_with_truncated_vocab(
    model_name: str,
    tf_model,
    target_vocab_size: int = 50364,
    cache_dir: str = "./pretrained_weights"
):
    """
    Load OpenAI pretrained weights with vocab truncation
    
    Strategy:
    1. Load full OpenAI weights (vocab=51865)
    2. Truncate decoder.token_embedding: 51865 → 50364
    3. Keep encoder weights unchanged (no vocab dependency)
    4. Keep decoder architecture weights (attention, MLP, etc.)
    
    Args:
        model_name: Model size ("tiny", "base", "small", etc.)
        tf_model: TensorFlow model to load weights into
        target_vocab_size: Target vocab size (50364 for PhoWhisper)
        cache_dir: Cache directory for downloaded weights
    
    Returns:
        tf_model: Model with loaded & truncated weights
    """
    
    print(f"\nLoading OpenAI '{model_name}' pretrained (truncating vocab: 51865 -> {target_vocab_size})")
    checkpoint_file, pytorch_state_dict = download_openai_weights(model_name)
    
    weight_count = 0
    
    # Encoder conv layers
    if 'encoder.conv1.weight' in pytorch_state_dict:
        conv1_weight = pytorch_state_dict['encoder.conv1.weight'].numpy()
        conv1_bias = pytorch_state_dict['encoder.conv1.bias'].numpy()
        tf_model.encoder.conv1.set_weights([
            conv1_weight.transpose(2, 1, 0),
            conv1_bias
        ])
        weight_count += 1
    
    if 'encoder.conv2.weight' in pytorch_state_dict:
        conv2_weight = pytorch_state_dict['encoder.conv2.weight'].numpy()
        conv2_bias = pytorch_state_dict['encoder.conv2.bias'].numpy()
        tf_model.encoder.conv2.set_weights([
            conv2_weight.transpose(2, 1, 0),
            conv2_bias
        ])
        weight_count += 1
    
    # Encoder positional embedding
    if 'encoder.positional_embedding' in pytorch_state_dict:
        # Encoder positional_embedding is tf.Variable directly
        tf_model.encoder.positional_embedding.assign(
            pytorch_state_dict['encoder.positional_embedding'].numpy()
        )
        weight_count += 1
    
    # Encoder blocks
    for i in range(len(tf_model.encoder.blocks)):
        prefix = f'encoder.blocks.{i}'
        tf_block = tf_model.encoder.blocks[i]
        
        # Attention
        tf_block.attn.query.set_weights([
            pytorch_state_dict[f'{prefix}.attn.query.weight'].numpy().T,
            pytorch_state_dict[f'{prefix}.attn.query.bias'].numpy()
        ])
        tf_block.attn.key.set_weights([
            pytorch_state_dict[f'{prefix}.attn.key.weight'].numpy().T
        ])
        tf_block.attn.value.set_weights([
            pytorch_state_dict[f'{prefix}.attn.value.weight'].numpy().T,
            pytorch_state_dict[f'{prefix}.attn.value.bias'].numpy()
        ])
        tf_block.attn.out.set_weights([
            pytorch_state_dict[f'{prefix}.attn.out.weight'].numpy().T,
            pytorch_state_dict[f'{prefix}.attn.out.bias'].numpy()
        ])
        
        # MLP
        tf_block.mlp.dense1.set_weights([
            pytorch_state_dict[f'{prefix}.mlp.0.weight'].numpy().T,
            pytorch_state_dict[f'{prefix}.mlp.0.bias'].numpy()
        ])
        tf_block.mlp.dense2.set_weights([
            pytorch_state_dict[f'{prefix}.mlp.2.weight'].numpy().T,
            pytorch_state_dict[f'{prefix}.mlp.2.bias'].numpy()
        ])
        
        # Layer norms
        tf_block.attn_ln.set_weights([
            pytorch_state_dict[f'{prefix}.attn_ln.weight'].numpy(),
            pytorch_state_dict[f'{prefix}.attn_ln.bias'].numpy()
        ])
        tf_block.mlp_ln.set_weights([
            pytorch_state_dict[f'{prefix}.mlp_ln.weight'].numpy(),
            pytorch_state_dict[f'{prefix}.mlp_ln.bias'].numpy()
        ])
        
        weight_count += 4
    
    # Encoder final layer norm
    tf_model.encoder.ln_post.set_weights([
        pytorch_state_dict['encoder.ln_post.weight'].numpy(),
        pytorch_state_dict['encoder.ln_post.bias'].numpy()
    ])
    weight_count += 1
    
    # TRUNCATE token embedding: 51865 -> target_vocab_size
    if 'decoder.token_embedding.weight' in pytorch_state_dict:
        full_embedding = pytorch_state_dict['decoder.token_embedding.weight'].numpy()
        truncated_embedding = full_embedding[:target_vocab_size, :]
        tf_model.decoder.token_embedding.set_weights([truncated_embedding])
        weight_count += 1
    
    # Decoder positional embedding (unchanged)
    if 'decoder.positional_embedding' in pytorch_state_dict:
        # Decoder positional_embedding is LearnedPositionalEncoding layer with .positional_embedding attribute
        tf_model.decoder.positional_embedding.positional_embedding.assign(
            pytorch_state_dict['decoder.positional_embedding'].numpy()
        )
        weight_count += 1
    
    # Decoder blocks
    for i in range(len(tf_model.decoder.blocks)):
        prefix = f'decoder.blocks.{i}'
        tf_block = tf_model.decoder.blocks[i]
        
        # Self-attention
        tf_block.attn.query.set_weights([
            pytorch_state_dict[f'{prefix}.attn.query.weight'].numpy().T,
            pytorch_state_dict[f'{prefix}.attn.query.bias'].numpy()
        ])
        tf_block.attn.key.set_weights([
            pytorch_state_dict[f'{prefix}.attn.key.weight'].numpy().T
        ])
        tf_block.attn.value.set_weights([
            pytorch_state_dict[f'{prefix}.attn.value.weight'].numpy().T,
            pytorch_state_dict[f'{prefix}.attn.value.bias'].numpy()
        ])
        tf_block.attn.out.set_weights([
            pytorch_state_dict[f'{prefix}.attn.out.weight'].numpy().T,
            pytorch_state_dict[f'{prefix}.attn.out.bias'].numpy()
        ])
        
        # Cross-attention
        tf_block.cross_attn.query.set_weights([
            pytorch_state_dict[f'{prefix}.cross_attn.query.weight'].numpy().T,
            pytorch_state_dict[f'{prefix}.cross_attn.query.bias'].numpy()
        ])
        tf_block.cross_attn.key.set_weights([
            pytorch_state_dict[f'{prefix}.cross_attn.key.weight'].numpy().T
        ])
        tf_block.cross_attn.value.set_weights([
            pytorch_state_dict[f'{prefix}.cross_attn.value.weight'].numpy().T,
            pytorch_state_dict[f'{prefix}.cross_attn.value.bias'].numpy()
        ])
        tf_block.cross_attn.out.set_weights([
            pytorch_state_dict[f'{prefix}.cross_attn.out.weight'].numpy().T,
            pytorch_state_dict[f'{prefix}.cross_attn.out.bias'].numpy()
        ])
        
        # MLP
        tf_block.mlp.dense1.set_weights([
            pytorch_state_dict[f'{prefix}.mlp.0.weight'].numpy().T,
            pytorch_state_dict[f'{prefix}.mlp.0.bias'].numpy()
        ])
        tf_block.mlp.dense2.set_weights([
            pytorch_state_dict[f'{prefix}.mlp.2.weight'].numpy().T,
            pytorch_state_dict[f'{prefix}.mlp.2.bias'].numpy()
        ])
        
        # Layer norms
        tf_block.attn_ln.set_weights([
            pytorch_state_dict[f'{prefix}.attn_ln.weight'].numpy(),
            pytorch_state_dict[f'{prefix}.attn_ln.bias'].numpy()
        ])
        tf_block.mlp_ln.set_weights([
            pytorch_state_dict[f'{prefix}.mlp_ln.weight'].numpy(),
            pytorch_state_dict[f'{prefix}.mlp_ln.bias'].numpy()
        ])
        tf_block.cross_attn_ln.set_weights([
            pytorch_state_dict[f'{prefix}.cross_attn_ln.weight'].numpy(),
            pytorch_state_dict[f'{prefix}.cross_attn_ln.bias'].numpy()
        ])
        
        weight_count += 6
    
    # Decoder final layer norm
    tf_model.decoder.ln.set_weights([
        pytorch_state_dict['decoder.ln.weight'].numpy(),
        pytorch_state_dict['decoder.ln.bias'].numpy()
    ])
    weight_count += 1
    
    print(f"Loaded {weight_count} weight groups. Encoder + decoder pretrained, vocab truncated to {target_vocab_size}.")
    
    return tf_model
