"""
Load Whisper Large teacher model using PyTorch
Works with RTX 5060 in NVIDIA Docker!
"""

import os
import sys
import torch
import numpy as np
from typing import Optional, Dict
import warnings

try:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
except ImportError as e:
    print(f"Error: {e}")
    print("Install with: pip install transformers")
    sys.exit(1)


class WhisperTeacherPyTorch:
    """
    PyTorch-based teacher model for distillation
    
    Uses Hugging Face Transformers WhisperModel (PyTorch)
    Compatible with RTX 5060 via NVIDIA Docker
    """
    
    def __init__(
        self,
        model_name: str = "openai/whisper-large-v2",
        device: str = "auto",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize Whisper Large teacher model (PyTorch)
        
        Args:
            model_name: Model ID from Hugging Face
            device: "auto", "cuda", or "cpu"
            cache_dir: Directory to cache model weights
        """
        
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        print(f"\n{'='*60}")
        print(f"Loading Whisper Teacher Model (PyTorch)")
        print(f"{'='*60}")
        print(f"  Model: {model_name}")
        
        self._setup_device(device)
        self._load_model()
        self._print_model_info()
        
        print(f"{'='*60}")
        print("Teacher model loaded successfully!")
        print(f"{'='*60}\n")
    
    def _setup_device(self, device: str):
        """Setup compute device"""
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"  Device: {self.device}")
        
        if self.device == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  PyTorch Version: {torch.__version__}")
    
    def _load_model(self):
        """Load PyTorch Whisper model and processor"""
        try:
            print(f"\n  Loading processor...")
            self.processor = WhisperProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            print(f"  Loading model (this may take a few minutes)...")
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            # Get config
            self.config = self.model.config
            
            print(f"  OK: Model loaded on {self.device}")
            
        except Exception as e:
            print(f"  Error loading model: {e}")
            raise
    
    def _print_model_info(self):
        """Print model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"\n  Model Configuration:")
        print(f"    Total Parameters: {total_params / 1e6:.1f}M")
        print(f"    Vocabulary Size: {self.config.vocab_size}")
        print(f"    Max Length: {self.config.max_length}")
        print(f"    Encoder Layers: {self.config.encoder_layers}")
        print(f"    Decoder Layers: {self.config.decoder_layers}")
        print(f"    Hidden Size: {self.config.d_model}")
    
    def generate_logits(
        self,
        audio_features: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate logits from teacher model with temperature scaling
        
        Args:
            audio_features: Audio features from encoder [batch, seq_len, hidden]
            decoder_input_ids: Decoder input IDs [batch, seq_len]
            temperature: Temperature for soft labels
            
        Returns:
            Soft logits [batch, seq_len, vocab_size]
        """
        
        with torch.no_grad():
            # Use model's forward pass to get logits
            outputs = self.model(
                encoder_outputs=(audio_features,),
                decoder_input_ids=decoder_input_ids,
                return_dict=True
            )
            
            # Get logits and apply temperature scaling
            logits = outputs.logits / temperature
        
        return logits
    
    def encode_audio(
        self,
        audio_array: np.ndarray,
        sampling_rate: int = 16000
    ) -> torch.Tensor:
        """
        Encode audio to features using Whisper encoder
        
        Args:
            audio_array: Audio array [samples]
            sampling_rate: Sampling rate
            
        Returns:
            Audio features [1, seq_len, hidden]
        """
        
        # Process audio with processor
        inputs = self.processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )
        
        # Move to device
        input_features = inputs.input_features.to(self.device)
        
        # Encode
        with torch.no_grad():
            encoder_outputs = self.model.model.encoder(input_features)
        
        return encoder_outputs.last_hidden_state
    
    def generate_logits_from_audio(
        self,
        audio_array: np.ndarray,
        decoder_input_ids: torch.Tensor,
        temperature: float = 3.0,
        sampling_rate: int = 16000
    ) -> torch.Tensor:
        """
        Complete pipeline: audio -> logits
        
        Args:
            audio_array: Raw audio [samples]
            decoder_input_ids: Decoder tokens [batch, seq_len]
            temperature: Temperature scaling
            sampling_rate: Audio sampling rate
            
        Returns:
            Teacher logits [batch, seq_len, vocab_size]
        """
        
        # Encode audio
        audio_features = self.encode_audio(audio_array, sampling_rate)
        
        # Move decoder_input_ids to device
        decoder_input_ids = decoder_input_ids.to(self.device)
        
        # Generate logits
        logits = self.generate_logits(
            audio_features,
            decoder_input_ids,
            temperature
        )
        
        return logits
    
    def transcribe_audio(
        self,
        audio_array: np.ndarray,
        sampling_rate: int = 16000,
        language: str = "vietnamese"
    ) -> str:
        """
        Transcribe audio using generate method
        
        Args:
            audio_array: Audio array
            sampling_rate: Sampling rate
            language: Language code
            
        Returns:
            Transcribed text
        """
        
        # Process audio
        inputs = self.processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )
        
        input_features = inputs.input_features.to(self.device)
        
        # Generate transcription
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_features,
                language=language,
                task="transcribe"
            )
        
        # Decode
        transcription = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        return transcription
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "framework": "PyTorch",
            "device": self.device,
            "vocab_size": self.config.vocab_size,
            "encoder_layers": self.config.encoder_layers,
            "decoder_layers": self.config.decoder_layers,
            "hidden_size": self.config.d_model,
            "total_params": sum(p.numel() for p in self.model.parameters()),
        }


def test_teacher_model_pytorch():
    """Test PyTorch teacher model"""
    
    print("\n" + "="*60)
    print("TESTING PYTORCH WHISPER TEACHER MODEL")
    print("="*60 + "\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        teacher = WhisperTeacherPyTorch(
            model_name="openai/whisper-large-v2",
            device=device
        )
        
        info = teacher.get_model_info()
        print("\nModel Information:")
        print(f"  Total Parameters: {info['total_params'] / 1e6:.1f}M")
        print(f"  Vocabulary Size: {info['vocab_size']}")
        print(f"  Encoder Layers: {info['encoder_layers']}")
        print(f"  Decoder Layers: {info['decoder_layers']}")
        
        print("\nTesting forward pass...")
        
        # Dummy audio features (simulating encoder output)
        batch_size = 2
        seq_len = 1500
        hidden_size = teacher.config.d_model
        
        audio_features = torch.randn(batch_size, seq_len, hidden_size).to(device)
        decoder_ids = torch.randint(
            0,
            teacher.config.vocab_size,
            (batch_size, 10)
        ).to(device)
        
        logits = teacher.generate_logits(
            audio_features,
            decoder_ids,
            temperature=3.0
        )
        
        print(f"  Audio features shape: {audio_features.shape}")
        print(f"  Decoder IDs shape: {decoder_ids.shape}")
        print(f"  Output logits shape: {logits.shape}")
        print(f"  OK: Forward pass successful!")
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60 + "\n")
        
        return teacher
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_teacher_model_pytorch()
