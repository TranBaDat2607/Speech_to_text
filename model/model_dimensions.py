"""
Model dimensions configuration for Whisper TensorFlow implementation.
Based on OpenAI Whisper model architecture.

IMPORTANT - Vocab Size Clarification:
- tokenizer.vocab_size = 50258 (base vocabulary for text encoding/decoding)
- model.config.vocab_size = 51865 (output projection layer size)
- Difference: 1607 additional tokens (1501 timestamp tokens + special tokens)
- Timestamp tokens: <|0.00|> to <|30.00|> (0.02s intervals)
- This n_vocab (51865) must match teacher model output shape for distillation
"""
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ModelDimensions:
    """Model dimensions configuration matching OpenAI Whisper specification"""
    n_mels: int                # Number of mel frequency bins (usually 80)
    n_audio_ctx: int          # Audio context length (usually 1500 after conv downsampling)
    n_audio_state: int        # Audio encoder embedding dimension
    n_audio_head: int         # Number of attention heads in audio encoder
    n_audio_layer: int        # Number of transformer layers in audio encoder
    n_vocab: int              # Vocabulary size for text decoder
    n_text_ctx: int           # Text context length (usually 448)
    n_text_state: int         # Text decoder embedding dimension
    n_text_head: int          # Number of attention heads in text decoder
    n_text_layer: int         # Number of transformer layers in text decoder


def get_whisper_dimensions(model_name: str) -> ModelDimensions:
    """
    Get model dimensions for different Whisper model sizes.
    Based on OpenAI Whisper official configurations.
    """
    dimensions = {
        "tiny": ModelDimensions(
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=384,
            n_audio_head=6,
            n_audio_layer=4,
            n_vocab=51865,  # PhoWhisper model vocab (includes timestamp tokens)
            n_text_ctx=448,
            n_text_state=384,
            n_text_head=6,
            n_text_layer=4,
        ),
        "base": ModelDimensions(
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=512,
            n_audio_head=8,
            n_audio_layer=6,
            n_vocab=51865, 
            n_text_ctx=448,
            n_text_state=512,
            n_text_head=8,
            n_text_layer=6,
        ),
        "small": ModelDimensions(
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=768,
            n_audio_head=12,
            n_audio_layer=12,
            n_vocab=51865, 
            n_text_ctx=448,
            n_text_state=768,
            n_text_head=12,
            n_text_layer=12,
        ),
        "medium": ModelDimensions(
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=1024,
            n_audio_head=16,
            n_audio_layer=24,
            n_vocab=51865,  # PhoWhisper model vocab (includes timestamp tokens)
            n_text_ctx=448,
            n_text_state=1024,
            n_text_head=16,
            n_text_layer=24,
        ),
        "large": ModelDimensions(
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=1280,
            n_audio_head=20,
            n_audio_layer=32,
            n_vocab=51865,  # PhoWhisper model vocab (includes timestamp tokens)
            n_text_ctx=448,
            n_text_state=1280,
            n_text_head=20,
            n_text_layer=32,
        ),
    }
    
    if model_name not in dimensions:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(dimensions.keys())}")
    
    return dimensions[model_name]


def get_custom_dimensions(**kwargs) -> ModelDimensions:
    """
    Create custom model dimensions.
    Useful for experimentation or custom model sizes.
    """
    defaults = {
        "n_mels": 80,
        "n_audio_ctx": 1500,
        "n_audio_state": 512,
        "n_audio_head": 8,
        "n_audio_layer": 6,
        "n_vocab": 51865,  # PhoWhisper model vocab (includes timestamp tokens)
        "n_text_ctx": 448,
        "n_text_state": 512,
        "n_text_head": 8,
        "n_text_layer": 6,
    }
    
    # Update defaults with provided kwargs
    defaults.update(kwargs)
    
    return ModelDimensions(**defaults)
