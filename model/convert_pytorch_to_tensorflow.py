"""
Convert PyTorch Whisper weights to TensorFlow format
"""

import os
import numpy as np
import torch
import tensorflow as tf
from model import create_whisper_model


def load_pytorch_weights(pt_file_path):
    """Load PyTorch weights from .pt file"""
    print(f"Loading PyTorch weights from {pt_file_path}...")
    state_dict = torch.load(pt_file_path, map_location='cpu')
    print(f"Loaded {len(state_dict)} weight tensors")
    return state_dict


def convert_pytorch_to_tensorflow(pytorch_weights, tf_model):
    """
    Convert PyTorch Whisper weights to TensorFlow model
    
    Args:
        pytorch_weights: PyTorch state_dict
        tf_model: TensorFlow Whisper model
    """
    print("\nConverting weights from PyTorch to TensorFlow...")
    
    weight_mapping = []
    
    for pt_name, pt_tensor in pytorch_weights.items():
        pt_weight = pt_tensor.numpy()
        
        if 'encoder.conv1.weight' in pt_name:
            tf_weight = np.transpose(pt_weight, (2, 1, 0))
            tf_model.encoder.conv1.set_weights([tf_weight, pytorch_weights['encoder.conv1.bias'].numpy()])
            weight_mapping.append(f"[OK] encoder.conv1: {pt_weight.shape} -> {tf_weight.shape}")
            
        elif 'encoder.conv2.weight' in pt_name:
            tf_weight = np.transpose(pt_weight, (2, 1, 0))
            tf_model.encoder.conv2.set_weights([tf_weight, pytorch_weights['encoder.conv2.bias'].numpy()])
            weight_mapping.append(f"[OK] encoder.conv2: {pt_weight.shape} -> {tf_weight.shape}")
            
        elif 'encoder.positional_embedding' in pt_name:
            tf_model.encoder.positional_embedding.assign(pt_weight)
            weight_mapping.append(f"[OK] encoder.positional_embedding: {pt_weight.shape}")
            
        elif 'decoder.token_embedding.weight' in pt_name:
            tf_model.decoder.token_embedding.set_weights([pt_weight])
            weight_mapping.append(f"[OK] decoder.token_embedding: {pt_weight.shape}")
            
        elif 'decoder.positional_embedding' in pt_name:
            tf_model.decoder.positional_embedding.positional_embedding.assign(pt_weight)
            weight_mapping.append(f"[OK] decoder.positional_embedding: {pt_weight.shape}")
    
    for i in range(len(tf_model.encoder.blocks)):
        block_prefix = f'encoder.blocks.{i}'
        tf_block = tf_model.encoder.blocks[i]
        
        attn_query_w = pytorch_weights[f'{block_prefix}.attn.query.weight'].numpy()
        attn_query_b = pytorch_weights[f'{block_prefix}.attn.query.bias'].numpy()
        attn_key_w = pytorch_weights[f'{block_prefix}.attn.key.weight'].numpy()
        attn_value_w = pytorch_weights[f'{block_prefix}.attn.value.weight'].numpy()
        attn_value_b = pytorch_weights[f'{block_prefix}.attn.value.bias'].numpy()
        attn_out_w = pytorch_weights[f'{block_prefix}.attn.out.weight'].numpy()
        attn_out_b = pytorch_weights[f'{block_prefix}.attn.out.bias'].numpy()
        
        tf_block.attn.query.set_weights([attn_query_w.T, attn_query_b])
        tf_block.attn.key.set_weights([attn_key_w.T])
        tf_block.attn.value.set_weights([attn_value_w.T, attn_value_b])
        tf_block.attn.out.set_weights([attn_out_w.T, attn_out_b])
        weight_mapping.append(f"[OK] encoder.block_{i}.attn")
        
        mlp_0_weight = pytorch_weights[f'{block_prefix}.mlp.0.weight'].numpy()
        mlp_0_bias = pytorch_weights[f'{block_prefix}.mlp.0.bias'].numpy()
        mlp_2_weight = pytorch_weights[f'{block_prefix}.mlp.2.weight'].numpy()
        mlp_2_bias = pytorch_weights[f'{block_prefix}.mlp.2.bias'].numpy()
        
        tf_block.mlp.dense1.set_weights([mlp_0_weight.T, mlp_0_bias])
        tf_block.mlp.dense2.set_weights([mlp_2_weight.T, mlp_2_bias])
        weight_mapping.append(f"[OK] encoder.block_{i}.mlp")
        
        ln1_weight = pytorch_weights[f'{block_prefix}.attn_ln.weight'].numpy()
        ln1_bias = pytorch_weights[f'{block_prefix}.attn_ln.bias'].numpy()
        ln2_weight = pytorch_weights[f'{block_prefix}.mlp_ln.weight'].numpy()
        ln2_bias = pytorch_weights[f'{block_prefix}.mlp_ln.bias'].numpy()
        
        tf_block.attn_ln.set_weights([ln1_weight, ln1_bias])
        tf_block.mlp_ln.set_weights([ln2_weight, ln2_bias])
        weight_mapping.append(f"[OK] encoder.block_{i}.layer_norms")
    
    encoder_ln_weight = pytorch_weights['encoder.ln_post.weight'].numpy()
    encoder_ln_bias = pytorch_weights['encoder.ln_post.bias'].numpy()
    tf_model.encoder.ln_post.set_weights([encoder_ln_weight, encoder_ln_bias])
    weight_mapping.append(f"[OK] encoder.ln_post")
    
    for i in range(len(tf_model.decoder.blocks)):
        block_prefix = f'decoder.blocks.{i}'
        tf_block = tf_model.decoder.blocks[i]
        
        attn_query_w = pytorch_weights[f'{block_prefix}.attn.query.weight'].numpy()
        attn_query_b = pytorch_weights[f'{block_prefix}.attn.query.bias'].numpy()
        attn_key_w = pytorch_weights[f'{block_prefix}.attn.key.weight'].numpy()
        attn_value_w = pytorch_weights[f'{block_prefix}.attn.value.weight'].numpy()
        attn_value_b = pytorch_weights[f'{block_prefix}.attn.value.bias'].numpy()
        attn_out_w = pytorch_weights[f'{block_prefix}.attn.out.weight'].numpy()
        attn_out_b = pytorch_weights[f'{block_prefix}.attn.out.bias'].numpy()
        
        tf_block.attn.query.set_weights([attn_query_w.T, attn_query_b])
        tf_block.attn.key.set_weights([attn_key_w.T])
        tf_block.attn.value.set_weights([attn_value_w.T, attn_value_b])
        tf_block.attn.out.set_weights([attn_out_w.T, attn_out_b])
        weight_mapping.append(f"[OK] decoder.block_{i}.self_attn")
        
        cross_attn_query_w = pytorch_weights[f'{block_prefix}.cross_attn.query.weight'].numpy()
        cross_attn_query_b = pytorch_weights[f'{block_prefix}.cross_attn.query.bias'].numpy()
        cross_attn_key_w = pytorch_weights[f'{block_prefix}.cross_attn.key.weight'].numpy()
        cross_attn_value_w = pytorch_weights[f'{block_prefix}.cross_attn.value.weight'].numpy()
        cross_attn_value_b = pytorch_weights[f'{block_prefix}.cross_attn.value.bias'].numpy()
        cross_attn_out_w = pytorch_weights[f'{block_prefix}.cross_attn.out.weight'].numpy()
        cross_attn_out_b = pytorch_weights[f'{block_prefix}.cross_attn.out.bias'].numpy()
        
        tf_block.cross_attn.query.set_weights([cross_attn_query_w.T, cross_attn_query_b])
        tf_block.cross_attn.key.set_weights([cross_attn_key_w.T])
        tf_block.cross_attn.value.set_weights([cross_attn_value_w.T, cross_attn_value_b])
        tf_block.cross_attn.out.set_weights([cross_attn_out_w.T, cross_attn_out_b])
        weight_mapping.append(f"[OK] decoder.block_{i}.cross_attn")
        
        mlp_0_weight = pytorch_weights[f'{block_prefix}.mlp.0.weight'].numpy()
        mlp_0_bias = pytorch_weights[f'{block_prefix}.mlp.0.bias'].numpy()
        mlp_2_weight = pytorch_weights[f'{block_prefix}.mlp.2.weight'].numpy()
        mlp_2_bias = pytorch_weights[f'{block_prefix}.mlp.2.bias'].numpy()
        
        tf_block.mlp.dense1.set_weights([mlp_0_weight.T, mlp_0_bias])
        tf_block.mlp.dense2.set_weights([mlp_2_weight.T, mlp_2_bias])
        weight_mapping.append(f"[OK] decoder.block_{i}.mlp")
        
        ln1_weight = pytorch_weights[f'{block_prefix}.attn_ln.weight'].numpy()
        ln1_bias = pytorch_weights[f'{block_prefix}.attn_ln.bias'].numpy()
        ln2_weight = pytorch_weights[f'{block_prefix}.cross_attn_ln.weight'].numpy()
        ln2_bias = pytorch_weights[f'{block_prefix}.cross_attn_ln.bias'].numpy()
        ln3_weight = pytorch_weights[f'{block_prefix}.mlp_ln.weight'].numpy()
        ln3_bias = pytorch_weights[f'{block_prefix}.mlp_ln.bias'].numpy()
        
        tf_block.attn_ln.set_weights([ln1_weight, ln1_bias])
        tf_block.cross_attn_ln.set_weights([ln2_weight, ln2_bias])
        tf_block.mlp_ln.set_weights([ln3_weight, ln3_bias])
        weight_mapping.append(f"[OK] decoder.block_{i}.layer_norms")
    
    decoder_ln_weight = pytorch_weights['decoder.ln.weight'].numpy()
    decoder_ln_bias = pytorch_weights['decoder.ln.bias'].numpy()
    tf_model.decoder.ln.set_weights([decoder_ln_weight, decoder_ln_bias])
    weight_mapping.append(f"[OK] decoder.ln")
    
    print("\nWeight conversion summary:")
    for mapping in weight_mapping[:10]:
        print(f"  {mapping}")
    print(f"  ... and {len(weight_mapping) - 10} more layers")
    
    return tf_model


def save_tensorflow_weights(tf_model, save_dir, model_name="tiny"):
    """Save TensorFlow model weights"""
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"whisper_{model_name}_tf.weights.h5")
    print(f"\nSaving TensorFlow weights to {save_path}...")
    tf_model.save_weights(save_path)
    
    print(f"Successfully saved TensorFlow weights!")
    print(f"File size: {os.path.getsize(save_path) / (1024*1024):.2f} MB")
    
    return save_path


def main():
    model_name = "tiny"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pt_file = os.path.join(script_dir, "pretrained_weights", f"whisper_{model_name}.pt")
    tf_save_dir = os.path.join(script_dir, "pretrained_weights")
    
    print(f"Converting Whisper {model_name} from PyTorch to TensorFlow")
    print(f"PyTorch file: {pt_file}")
    print(f"TensorFlow save dir: {tf_save_dir}")
    
    pytorch_weights = load_pytorch_weights(pt_file)
    
    print(f"\nCreating TensorFlow model...")
    tf_model = create_whisper_model(model_name)
    
    dummy_mel = tf.random.normal([1, 80, 3000])
    dummy_tokens = tf.constant([[50258, 50259, 50359]], dtype=tf.int32)
    _ = tf_model(dummy_mel, dummy_tokens, training=False)
    print(f"TensorFlow model initialized")
    
    tf_model = convert_pytorch_to_tensorflow(pytorch_weights, tf_model)
    
    save_path = save_tensorflow_weights(tf_model, tf_save_dir, model_name)
    
    print(f"\n[SUCCESS] Conversion complete!")
    print(f"You can now load the weights in your training script:")
    print(f"  model.load_weights('{save_path}')")


if __name__ == "__main__":
    main()
