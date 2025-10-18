"""
Text processing utilities for Whisper training
"""

import tensorflow as tf
import numpy as np
from typing import List, Tuple, Optional
from tokenizer import get_tokenizer


class WhisperTextProcessor:
    """
    Text processor for Whisper training
    
    Handles tokenization and preparing text sequences for autoregressive training
    """
    
    def __init__(self, language: str = "vi", task: str = "transcribe", max_length: int = 448):
        """
        Initialize text processor
        
        Args:
            language: Language code (vi for Vietnamese)
            task: Task type ("transcribe" or "translate")
            max_length: Maximum sequence length (Whisper uses 448 for training)
        """
        self.language = language
        self.task = task
        self.max_length = max_length
        
        # Initialize tokenizer
        # For Vietnamese (multilingual model), use multilingual=True
        self.tokenizer = get_tokenizer(
            multilingual=True,
            language=language,
            task=task
        )
        
        print(f"Initialized tokenizer for language: {language}, task: {task}")
        print(f"SOT sequence: {self.tokenizer.sot_sequence}")
        print(f"EOT token: {self.tokenizer.eot}")
        print(f"Max sequence length: {max_length}")
    
    def tokenize_text(self, text: str) -> List[int]:
        """
        Tokenize text to token IDs
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token IDs
        """
        # Encode text using the tokenizer
        token_ids = self.tokenizer.encode(text)
        return token_ids
    
    def prepare_decoder_input_target(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare decoder input and target sequences for autoregressive training
        
        In autoregressive training:
        - decoder_input: [SOT, lang, task, token1, token2, ..., tokenN]
        - decoder_target: [lang, task, token1, token2, ..., tokenN, EOT]
        
        Args:
            text: Input transcript text
            
        Returns:
            Tuple of (decoder_input, decoder_target) as numpy arrays
        """
        # Tokenize the text
        text_tokens = self.tokenize_text(text)
        
        # Create complete sequence: SOT + text + EOT
        sot_sequence = list(self.tokenizer.sot_sequence)  # [SOT, lang, task]
        complete_sequence = sot_sequence + text_tokens + [self.tokenizer.eot]
        
        # Decoder input: everything except the last token
        decoder_input = complete_sequence[:-1]
        
        # Decoder target: everything except the first token (SOT)
        decoder_target = complete_sequence[1:]
        
        # Pad or truncate to max_length
        decoder_input = self._pad_or_truncate(decoder_input, self.max_length)
        decoder_target = self._pad_or_truncate(decoder_target, self.max_length)
        
        return np.array(decoder_input, dtype=np.int32), np.array(decoder_target, dtype=np.int32)
    
    def _pad_or_truncate(self, sequence: List[int], max_length: int) -> List[int]:
        """
        Pad sequence with EOT tokens or truncate to max_length
        
        Args:
            sequence: Input sequence
            max_length: Target length
            
        Returns:
            Padded or truncated sequence
        """
        if len(sequence) > max_length:
            # Truncate, but ensure EOT at the end
            sequence = sequence[:max_length-1] + [self.tokenizer.eot]
        elif len(sequence) < max_length:
            # Pad with EOT tokens
            sequence = sequence + [self.tokenizer.eot] * (max_length - len(sequence))
        
        return sequence
    
    def process_batch_texts(self, texts: List[str]) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Process a batch of texts for training
        
        Args:
            texts: List of transcript texts
            
        Returns:
            Tuple of (decoder_inputs, decoder_targets) as TensorFlow tensors
        """
        batch_decoder_inputs = []
        batch_decoder_targets = []
        
        for text in texts:
            decoder_input, decoder_target = self.prepare_decoder_input_target(text)
            batch_decoder_inputs.append(decoder_input)
            batch_decoder_targets.append(decoder_target)
        
        # Convert to TensorFlow tensors
        batch_decoder_inputs = tf.constant(batch_decoder_inputs, dtype=tf.int32)
        batch_decoder_targets = tf.constant(batch_decoder_targets, dtype=tf.int32)
        
        return batch_decoder_inputs, batch_decoder_targets
    
    def decode_tokens(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids)
    
    def test_tokenization(self, sample_texts: List[str]):
        """Test tokenization functionality"""
        print("\n=== Testing Text Processing ===")
        
        for i, text in enumerate(sample_texts):
            print(f"\nSample {i+1}:")
            print(f"Original text: {text[:100]}...")
            
            # Test tokenization
            tokens = self.tokenize_text(text)
            print(f"Tokenized ({len(tokens)} tokens): {tokens[:10]}...")
            
            # Test decoder input/target preparation
            decoder_input, decoder_target = self.prepare_decoder_input_target(text)
            print(f"Decoder input shape: {decoder_input.shape}")
            print(f"Decoder target shape: {decoder_target.shape}")
            print(f"Decoder input (first 10): {decoder_input[:10]}")
            print(f"Decoder target (first 10): {decoder_target[:10]}")
            
            # Test decoding
            decoded_text = self.decode_tokens(tokens)
            print(f"Decoded text: {decoded_text[:100]}...")
        
        # Test batch processing
        print(f"\n=== Testing Batch Processing ===")
        batch_inputs, batch_targets = self.process_batch_texts(sample_texts)
        print(f"Batch inputs shape: {batch_inputs.shape}")
        print(f"Batch targets shape: {batch_targets.shape}")
        
        print("=== Text Processing Test Complete ===\n")


if __name__ == "__main__":
    # Test the text processor
    print("Testing WhisperTextProcessor...")
    
    # Initialize for Vietnamese
    text_processor = WhisperTextProcessor(
        language="vi",  # Vietnamese
        task="transcribe",
        max_length=448
    )
    
    # Test with some sample Vietnamese texts
    sample_texts = [
        "Xin chào các bạn, hôm nay chúng ta sẽ học về trí tuệ nhân tạo.",
        "Đây là một ví dụ về xử lý ngôn ngữ tự nhiên trong tiếng Việt.",
        "Machine learning là một lĩnh vực rất thú vị và có nhiều ứng dụng."
    ]
    
    text_processor.test_tokenization(sample_texts)
