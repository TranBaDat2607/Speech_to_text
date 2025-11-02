"""
Knowledge Distillation Loss Functions
Combines soft targets (teacher) and hard targets (ground truth)
"""

import tensorflow as tf
from typing import Optional, Tuple


class DistillationLoss:
    """
    Knowledge Distillation Loss for Whisper training
    
    Loss = alpha * KL_Divergence(student, teacher) + (1-alpha) * CrossEntropy(student, labels)
    
    Where:
    - KL_Divergence: Soft target loss (learns from teacher's probability distribution)
    - CrossEntropy: Hard target loss (learns from ground truth labels)
    - alpha: Weight balancing soft vs hard targets (typically 0.5-0.9)
    - Temperature: Softens probability distributions (typically 2-5)
    """
    
    def __init__(
        self,
        alpha: float = 0.7,
        temperature: float = 3.0,
        ignore_index: int = -100
    ):
        """
        Initialize distillation loss
        
        Args:
            alpha: Weight for soft target loss (0-1)
                  alpha=1.0: only learn from teacher
                  alpha=0.0: only learn from labels
                  alpha=0.7: balanced (recommended)
            temperature: Temperature for softening distributions
                        Higher = softer, more information transfer
            ignore_index: Label index to ignore (padding tokens)
        """
        self.alpha = alpha
        self.temperature = temperature
        self.ignore_index = ignore_index
        
        # Loss functions
        self.kl_loss = tf.keras.losses.KLDivergence(reduction='none')
        self.ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction='none'
        )
    
    def __call__(
        self,
        student_logits: tf.Tensor,
        teacher_logits: tf.Tensor,
        labels: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None
    ) -> Tuple[tf.Tensor, dict]:
        """
        Compute distillation loss
        
        Args:
            student_logits: [batch_size, seq_len, vocab_size] - Student model output
            teacher_logits: [batch_size, seq_len, vocab_size] - Teacher soft labels
            labels: [batch_size, seq_len] - Ground truth token IDs
            attention_mask: [batch_size, seq_len] - Mask for valid tokens (optional)
            
        Returns:
            loss: Scalar tensor - Combined loss
            loss_dict: Dictionary with loss components for logging
        """
        batch_size = tf.shape(student_logits)[0]
        seq_len = tf.shape(student_logits)[1]
        student_vocab_size = tf.shape(student_logits)[2]
        teacher_vocab_size = tf.shape(teacher_logits)[2]
        
        # Handle vocab size mismatch (PhoWhisper=51864 vs OpenAI=51865)
        if teacher_vocab_size != student_vocab_size:
            # Pad teacher logits to match student vocab size
            pad_size = student_vocab_size - teacher_vocab_size
            padding = tf.zeros([batch_size, seq_len, pad_size], dtype=teacher_logits.dtype)
            teacher_logits = tf.concat([teacher_logits, padding], axis=-1)
        
        vocab_size = student_vocab_size
        
        # 1. Soft target loss (KL Divergence)
        # Apply temperature scaling
        student_soft = tf.nn.log_softmax(student_logits / self.temperature, axis=-1)
        teacher_soft = tf.nn.softmax(teacher_logits / self.temperature, axis=-1)
        
        # Reshape for KL divergence
        student_soft_flat = tf.reshape(student_soft, [-1, vocab_size])
        teacher_soft_flat = tf.reshape(teacher_soft, [-1, vocab_size])
        
        # Compute KL divergence per token
        kl_div_per_token = self.kl_loss(teacher_soft_flat, student_soft_flat)
        kl_div_per_token = tf.reshape(kl_div_per_token, [batch_size, seq_len])
        
        # Scale by temperature^2 (standard in knowledge distillation)
        kl_div_per_token = kl_div_per_token * (self.temperature ** 2)
        
        # 2. Hard target loss (Cross Entropy)
        # Reshape for cross entropy
        student_logits_flat = tf.reshape(student_logits, [-1, vocab_size])
        labels_flat = tf.reshape(labels, [-1])
        
        # Compute cross entropy per token
        ce_per_token = self.ce_loss(labels_flat, student_logits_flat)
        ce_per_token = tf.reshape(ce_per_token, [batch_size, seq_len])
        
        # 3. Apply mask (ignore padding and special tokens)
        if attention_mask is None:
            # Create mask from labels (ignore_index = padding)
            mask = tf.cast(tf.not_equal(labels, self.ignore_index), tf.float32)
        else:
            mask = tf.cast(attention_mask, tf.float32)
        
        # Apply mask to both losses
        kl_div_per_token = kl_div_per_token * mask
        ce_per_token = ce_per_token * mask
        
        # 4. Compute average losses
        num_valid_tokens = tf.reduce_sum(mask) + 1e-8  # Avoid division by zero
        
        kl_loss = tf.reduce_sum(kl_div_per_token) / num_valid_tokens
        ce_loss = tf.reduce_sum(ce_per_token) / num_valid_tokens
        
        # 5. Combine losses
        total_loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss
        
        # 6. Prepare loss dict for logging
        loss_dict = {
            'loss': total_loss,
            'kl_loss': kl_loss,
            'ce_loss': ce_loss,
            'alpha': self.alpha,
            'temperature': self.temperature
        }
        
        return total_loss, loss_dict
    
    def get_config(self) -> dict:
        """Get loss configuration"""
        return {
            'alpha': self.alpha,
            'temperature': self.temperature,
            'ignore_index': self.ignore_index
        }


def compute_distillation_loss(
    student_logits: tf.Tensor,
    teacher_logits: tf.Tensor,
    labels: tf.Tensor,
    alpha: float = 0.7,
    temperature: float = 3.0,
    ignore_index: int = -100
) -> Tuple[tf.Tensor, dict]:
    """
    Convenience function to compute distillation loss
    
    Args:
        student_logits: [batch_size, seq_len, vocab_size]
        teacher_logits: [batch_size, seq_len, vocab_size]
        labels: [batch_size, seq_len]
        alpha: Weight for soft target loss
        temperature: Temperature scaling
        ignore_index: Label to ignore
        
    Returns:
        loss: Scalar tensor
        loss_dict: Dictionary with loss components
    """
    loss_fn = DistillationLoss(alpha, temperature, ignore_index)
    return loss_fn(student_logits, teacher_logits, labels)
