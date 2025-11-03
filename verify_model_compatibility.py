"""
Verify TensorFlow student and PyTorch teacher model compatibility
Check vocab sizes and output shapes match for distillation
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Add model directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'distillation', 'teacher'))

from model import create_whisper_model
from load_teacher_pytorch import WhisperTeacherPyTorch
import torch

print("="*70)
print("VERIFYING MODEL COMPATIBILITY FOR KNOWLEDGE DISTILLATION")
print("="*70)

# 1. Check TensorFlow student model
print("\n[1] TensorFlow Student Model (base)")
tf_model = create_whisper_model("base")
print(f"    Vocab size: {tf_model.dims.n_vocab}")
print(f"    Encoder output: [batch, {tf_model.dims.n_audio_ctx}, {tf_model.dims.n_audio_state}]")
print(f"    Decoder output: [batch, seq_len, {tf_model.dims.n_vocab}]")

# 2. Check PyTorch teacher model
print("\n[2] PyTorch Teacher Model (PhoWhisper-large)")
teacher = WhisperTeacherPyTorch(model_name="vinai/PhoWhisper-large")
print(f"    Vocab size: {teacher.config.vocab_size}")
print(f"    Encoder output: [batch, 1500, {teacher.config.d_model}]")
print(f"    Decoder output: [batch, seq_len, {teacher.config.vocab_size}]")

# 3. Test forward pass with matching shapes
print("\n[3] Testing Forward Pass Compatibility")
batch_size = 2
seq_len = 10

# Create dummy inputs
mel_tf = tf.random.normal([batch_size, 80, 3000])
tokens_tf = tf.random.uniform([batch_size, seq_len], maxval=tf_model.dims.n_vocab, dtype=tf.int32)

# TensorFlow forward
print(f"    Input mel shape: {mel_tf.shape}")
print(f"    Input tokens shape: {tokens_tf.shape}")

logits_tf = tf_model(mel_tf, tokens_tf, training=False)
print(f"    TF Student logits: {logits_tf.shape}")

# PyTorch forward
audio_features_pt = torch.randn(batch_size, 1500, teacher.config.d_model).to(teacher.device)
tokens_pt = torch.randint(0, teacher.config.vocab_size, (batch_size, seq_len)).to(teacher.device)

logits_pt = teacher.generate_logits(audio_features_pt, tokens_pt, temperature=1.0)
print(f"    PT Teacher logits: {logits_pt.shape}")

# 4. Verify compatibility
print("\n[4] Compatibility Check")
vocab_match = tf_model.dims.n_vocab == teacher.config.vocab_size
shape_match = logits_tf.shape[-1] == logits_pt.shape[-1]

print(f"    ✓ Vocab size match: {vocab_match} (TF={tf_model.dims.n_vocab}, PT={teacher.config.vocab_size})")
print(f"    ✓ Output shape match: {shape_match} (TF={logits_tf.shape}, PT={tuple(logits_pt.shape)})")

if vocab_match and shape_match:
    print("\n" + "="*70)
    print("✓ SUCCESS: Models are compatible for knowledge distillation!")
    print("="*70)
else:
    print("\n" + "="*70)
    print("✗ ERROR: Models have incompatible shapes!")
    print("="*70)
    sys.exit(1)

print("\n[5] Vocab Size Clarification")
print("    - tokenizer.vocab_size = 50258 (base vocabulary for encoding/decoding text)")
print("    - model.config.vocab_size = 51865 (output layer size)")
print("    - Difference: 51865 - 50258 = 1607 tokens (1501 timestamps + specials)")
print("    - Timestamps: <|0.00|> to <|30.00|> (1501 tokens for 0.02s intervals)")

print("\n[6] Next Steps")
print("    1. Load OpenAI weights into TensorFlow model to verify architecture")
print("    2. Test distillation loss calculation")
print("    3. Run distillation training")
