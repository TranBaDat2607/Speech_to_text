"""
Load OpenAI Whisper pretrained weights and convert to TensorFlow
Similar to model/convert_pytorch_to_tensorflow.py but integrated for distillation
"""

import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

try:
    import whisper
    import torch
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

def download_openai_weights(model_name: str = "base"):
    """
    Download OpenAI Whisper pretrained weights using official library
    
    Args:
        model_name: Model size (tiny, base, small, medium, large)
        
    Returns:
        Path to downloaded .pt file
    """
    # Use whisper's download mechanism (silent)
    model = whisper.load_model(model_name, device="cpu", download_root=None)
    
    # Get checkpoint path from whisper's cache
    checkpoint_path = whisper._MODELS[model_name]
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
    checkpoint_file = os.path.join(cache_dir, os.path.basename(checkpoint_path))
    
    if os.path.exists(checkpoint_file):
        return checkpoint_file, model.state_dict()
    else:
        raise FileNotFoundError(f"Could not find checkpoint at {checkpoint_file}")


def convert_pytorch_to_tensorflow_weights(pytorch_state_dict, tf_model, model_name: str = "base"):
    """
    Convert PyTorch Whisper weights to TensorFlow model
    
    Args:
        pytorch_state_dict: PyTorch state_dict from whisper.load_model()
        tf_model: TensorFlow Whisper model
        model_name: Model size for logging
        
    Returns:
        TensorFlow model with loaded weights
    """
    print(f"\n{'='*60}")
    print(f"Converting PyTorch → TensorFlow weights...")
    print(f"{'='*60}")
    
    weight_count = 0
    
    # Encoder conv layers
    if 'encoder.conv1.weight' in pytorch_state_dict:
        pt_weight = pytorch_state_dict['encoder.conv1.weight'].numpy()
        tf_weight = np.transpose(pt_weight, (2, 1, 0))  # PyTorch [out, in, k] -> TF [k, in, out]
        tf_model.encoder.conv1.set_weights([
            tf_weight, 
            pytorch_state_dict['encoder.conv1.bias'].numpy()
        ])
        print(f"[CONV] encoder.conv1: {pt_weight.shape} -> {tf_weight.shape}")
        weight_count += 1
    
    if 'encoder.conv2.weight' in pytorch_state_dict:
        pt_weight = pytorch_state_dict['encoder.conv2.weight'].numpy()
        tf_weight = np.transpose(pt_weight, (2, 1, 0))
        tf_model.encoder.conv2.set_weights([
            tf_weight,
            pytorch_state_dict['encoder.conv2.bias'].numpy()
        ])
        print(f"[CONV] encoder.conv2: {pt_weight.shape} -> {tf_weight.shape}")
        weight_count += 1
    
    # Positional embeddings
    if 'encoder.positional_embedding' in pytorch_state_dict:
        tf_model.encoder.positional_embedding.assign(
            pytorch_state_dict['encoder.positional_embedding'].numpy()
        )
        print(f"[LOAD] encoder.positional_embedding")
        weight_count += 1
    
    if 'decoder.token_embedding.weight' in pytorch_state_dict:
        tf_model.decoder.token_embedding.set_weights([
            pytorch_state_dict['decoder.token_embedding.weight'].numpy()
        ])
        print(f"[LOAD] decoder.token_embedding")
        weight_count += 1
    
    if 'decoder.positional_embedding' in pytorch_state_dict:
        tf_model.decoder.positional_embedding.positional_embedding.assign(
            pytorch_state_dict['decoder.positional_embedding'].numpy()
        )
        print(f"[LOAD] decoder.positional_embedding")
        weight_count += 1
    
    # Encoder blocks
    print(f"\nConverting encoder blocks...")
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
        
        weight_count += 4  # attn, mlp, 2x ln
    
    print(f"[DONE] Converted {len(tf_model.encoder.blocks)} encoder blocks")
    
    # Encoder final layer norm
    tf_model.encoder.ln_post.set_weights([
        pytorch_state_dict['encoder.ln_post.weight'].numpy(),
        pytorch_state_dict['encoder.ln_post.bias'].numpy()
    ])
    weight_count += 1
    
    # Decoder blocks
    print(f"\nConverting decoder blocks...")
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
        tf_block.cross_attn_ln.set_weights([
            pytorch_state_dict[f'{prefix}.cross_attn_ln.weight'].numpy(),
            pytorch_state_dict[f'{prefix}.cross_attn_ln.bias'].numpy()
        ])
        tf_block.mlp_ln.set_weights([
            pytorch_state_dict[f'{prefix}.mlp_ln.weight'].numpy(),
            pytorch_state_dict[f'{prefix}.mlp_ln.bias'].numpy()
        ])
        
        weight_count += 6  # self-attn, cross-attn, mlp, 3x ln
    
    print(f"[DONE] Converted {len(tf_model.decoder.blocks)} decoder blocks")
    
    # Decoder final layer norm
    tf_model.decoder.ln.set_weights([
        pytorch_state_dict['decoder.ln.weight'].numpy(),
        pytorch_state_dict['decoder.ln.bias'].numpy()
    ])
    weight_count += 1
    
    print(f"\n{'='*60}")
    print(f"[DONE] Conversion complete!")
    print(f"       Total weight groups converted: {weight_count}")
    print(f"{'='*60}")
    
    return tf_model


def load_and_convert_openai_weights(model_name: str, tf_model, cache_dir: str = "./pretrained_weights"):
    """
    Main function to download OpenAI weights and convert to TensorFlow
    
    Args:
        model_name: Model size (tiny, base, small, medium, large)
        tf_model: TensorFlow Whisper model to load weights into
        cache_dir: Directory to cache converted TensorFlow weights
        
    Returns:
        TensorFlow model with loaded OpenAI pretrained weights
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    
    # Check if already converted
    tf_weights_path = cache_dir / f"openai_whisper_{model_name}_tf.weights.h5"
    
    if tf_weights_path.exists():
        print(f"\n[CACHE] Found cached TensorFlow weights: {tf_weights_path}")
        print(f"        Loading directly...")
        tf_model.load_weights(str(tf_weights_path))
        print(f"[DONE] Loaded cached weights successfully!")
        return tf_model
    
    # Download and convert
    print(f"\n[WARN] No cached weights found. Will download and convert...")
    
    if not WHISPER_AVAILABLE:
        raise ImportError(
            "whisper library required for first-time download.\n"
            "Install with: pip install openai-whisper torch"
        )
    
    # Download OpenAI weights
    checkpoint_file, pytorch_state_dict = download_openai_weights(model_name)
    
    # Convert to TensorFlow
    tf_model = convert_pytorch_to_tensorflow_weights(pytorch_state_dict, tf_model, model_name)
    
    # Save converted weights for future use
    print(f"\nSaving converted weights to cache...")
    tf_model.save_weights(str(tf_weights_path))
    size_mb = tf_weights_path.stat().st_size / (1024**2)
    print(f"[SAVE] Saved to: {tf_weights_path} ({size_mb:.2f} MB)")
    print(f"       Next time will load directly from cache!")
    
    return tf_model


if __name__ == "__main__":
    # Test conversion
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from student.load_student_tensorflow import WhisperStudentTensorFlow
    
    print("\nTesting OpenAI pretrained weights loading...")
    
    model_name = "base"
    student = WhisperStudentTensorFlow(model_name=model_name, freeze_encoder=False)
    
    # Load OpenAI weights
    student.model = load_and_convert_openai_weights(
        model_name=model_name,
        tf_model=student.model,
        cache_dir="./pretrained_weights"
    )
    
    print("\n[DONE] Test complete! OpenAI weights loaded successfully.")
