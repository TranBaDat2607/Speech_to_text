"""
Load Whisper Small student model using TensorFlow
Uses custom TensorFlow implementation from model/
"""

import os
import sys
from pathlib import Path
import tensorflow as tf
from typing import Optional, Dict

# Add model directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "model"))

from model_dimensions import get_whisper_dimensions
from model import create_whisper_model, Whisper


class WhisperStudentTensorFlow:
    """
    TensorFlow-based student model for distillation training
    Uses custom TensorFlow implementation
    """
    
    def __init__(
        self,
        model_name: str = "small",
        freeze_encoder: bool = False,
        weights_path: Optional[str] = None,
        load_openai_weights: bool = True
    ):
        """
        Initialize Whisper Small student model
        
        Args:
            model_name: Model size ("tiny", "base", "small", "medium", "large")
            freeze_encoder: If True, freeze encoder weights (train decoder only)
            weights_path: Optional path to load pretrained weights (.weights.h5)
            load_openai_weights: If True, load OpenAI pretrained weights
        """
        self.model_name = model_name
        self.freeze_encoder = freeze_encoder
        self.weights_path = weights_path
        self.load_openai_weights = load_openai_weights
        
        print(f"\nStudent: {model_name}")
        
        self._setup_gpu()
        self._load_model()
        self._print_model_info()
        
        if freeze_encoder:
            self._freeze_encoder()
        
        # Load weights priority:
        # 1. If weights_path provided and exists -> load from checkpoint
        # 2. Else if load_openai_weights=True -> load OpenAI pretrained
        # 3. Else -> use random initialization
        if weights_path and os.path.exists(weights_path):
            self._load_weights(weights_path)
        elif load_openai_weights:
            self._load_openai_pretrained()
    
    def _setup_gpu(self):
        """Setup GPU memory growth"""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU setup error: {e}")
    
    def _load_model(self):
        """Load TensorFlow Whisper model"""
        try:
            # Create model using factory function
            self.model = create_whisper_model(self.model_name)
            
            # Get model dimensions
            self.dims = get_whisper_dimensions(self.model_name)
            
            # Build model with dummy inputs to initialize weights
            dummy_mel = tf.random.normal([1, self.dims.n_mels, 3000])
            dummy_tokens = tf.random.uniform([1, 10], 0, self.dims.n_vocab, dtype=tf.int32)
            _ = self.model(dummy_mel, dummy_tokens, training=False)
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _freeze_encoder(self):
        """Freeze encoder weights (train decoder only)"""
        self.model.encoder.trainable = False
        trainable_params = sum([tf.size(var).numpy() for var in self.model.trainable_variables])
        print(f"Encoder frozen. Trainable: {trainable_params / 1e6:.1f}M")
    
    def _load_weights(self, weights_path: str):
        """Load pretrained weights from .weights.h5 file"""
        try:
            self.model.load_weights(weights_path)
            print(f"Loaded weights: {weights_path}")
        except Exception as e:
            print(f"Warning: Could not load weights: {e}")
    
    def _load_openai_pretrained(self):
        """
        Load OpenAI Whisper pretrained weights
        Downloads from OpenAI if needed, converts PyTorch -> TensorFlow
        
        Uses caching to avoid re-downloading
        Only downloads/converts once, then loads from cache
        """
        
        try:
            # Set cache directory
            cache_dir = Path(__file__).parent / "pretrained_weights"
            
            # Load full OpenAI weights (vocab_size=51865, matches PhoWhisper)
            try:
                from .load_openai_pretrained import load_and_convert_openai_weights
            except ImportError:
                # Fallback for when script is run directly
                import sys
                sys.path.insert(0, str(Path(__file__).parent))
                from load_openai_pretrained import load_and_convert_openai_weights
            
            self.model = load_and_convert_openai_weights(
                model_name=self.model_name,
                tf_model=self.model,
                cache_dir=str(cache_dir)
            )
            
        except ImportError as e:
            print(f"\n[WARN] Warning: Could not load OpenAI weights: {e}")
            print(f"       Install dependencies: pip install openai-whisper torch")
            print(f"       Continuing with random initialization...")
        except Exception as e:
            print(f"\n[WARN] Warning: Error loading OpenAI weights: {e}")
            print(f"       Continuing with random initialization...")
    
    def _print_model_info(self):
        """Print model information"""
        total_params = sum([tf.size(var).numpy() for var in self.model.variables])
        trainable_params = sum([tf.size(var).numpy() for var in self.model.trainable_variables])
        
        print(f"{total_params / 1e6:.1f}M params, vocab {self.dims.n_vocab}, {self.dims.n_audio_layer}E/{self.dims.n_text_layer}D layers")
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        total_params = sum([tf.size(var).numpy() for var in self.model.variables])
        trainable_params = sum([tf.size(var).numpy() for var in self.model.trainable_variables])
        
        return {
            "model_name": self.model_name,
            "framework": "TensorFlow",
            "vocab_size": self.dims.n_vocab,
            "encoder_layers": self.dims.n_audio_layer,
            "decoder_layers": self.dims.n_text_layer,
            "hidden_size": self.dims.n_audio_state,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "freeze_encoder": self.freeze_encoder
        }
    
    def save_checkpoint(self, checkpoint_path: str, optimizer_state: Optional[Dict] = None,
                       epoch: int = 0, step: int = 0, loss: float = 0.0):
        """
        Save model checkpoint
        
        Args:
            checkpoint_path: Path to save checkpoint
            optimizer_state: Not used for TensorFlow (handled by tf.train.Checkpoint)
            epoch: Current epoch
            step: Current step
            loss: Current loss
        """
        # Save weights
        if checkpoint_path.endswith('.weights.h5'):
            weights_path = checkpoint_path
            metadata_path = checkpoint_path.replace('.weights.h5', '_metadata.json')
        else:
            weights_path = checkpoint_path.replace('.ckpt', '_weights.h5')
            metadata_path = checkpoint_path.replace('.ckpt', '_metadata.json')
        
        self.model.save_weights(weights_path)
        
        # Save metadata
        import json
        with open(metadata_path, 'w') as f:
            json.dump({
                'epoch': epoch,
                'step': step,
                'loss': loss,
                'model_name': self.model_name
            }, f)
        
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary with checkpoint info (epoch, step, loss)
        """
        # Load weights
        if checkpoint_path.endswith('.weights.h5'):
            weights_path = checkpoint_path
            metadata_path = checkpoint_path.replace('.weights.h5', '_metadata.json')
        else:
            weights_path = checkpoint_path.replace('.ckpt', '_weights.h5')
            metadata_path = checkpoint_path.replace('.ckpt', '_metadata.json')
        
        self.model.load_weights(weights_path)
        
        # Load metadata
        import json
        with open(metadata_path, 'r') as f:
            info = json.load(f)
        
        print(f"Loaded: {weights_path} (epoch {info['epoch']}, step {info['step']}, loss {info['loss']:.4f})")
        
        return info