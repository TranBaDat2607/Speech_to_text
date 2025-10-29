#!/usr/bin/env python3
"""
Quick test script để verify setup
Chạy trong Docker container
"""

import sys
import warnings

def test_imports():
    """Test import các thư viện cần thiết"""
    print("\n" + "="*60)
    print("TEST 1: Import Libraries")
    print("="*60)
    
    try:
        import torch
        print(f"OK: PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"ERROR: PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"OK: Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"ERROR: Transformers import failed: {e}")
        return False
    
    try:
        import librosa
        print(f"OK: librosa")
    except ImportError as e:
        print(f"ERROR: librosa import failed: {e}")
        return False
    
    try:
        import soundfile
        print(f"OK: soundfile")
    except ImportError as e:
        print(f"ERROR: soundfile import failed: {e}")
        return False
    
    return True


def test_gpu():
    """Test GPU availability"""
    print("\n" + "="*60)
    print("TEST 2: GPU Detection")
    print("="*60)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available")
            print("  GPU training will not work!")
            return False
        
        print(f"OK: CUDA available")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        print(f"  GPU name: {torch.cuda.get_device_name(0)}")
        
        # Check compute capability
        capability = torch.cuda.get_device_capability(0)
        print(f"  Compute capability: sm_{capability[0]}{capability[1]}")
        
        # Warning for RTX 5060
        if capability[0] == 12 and capability[1] == 0:
            print("\n  WARNING: RTX 5060 (sm_120) detected")
            print("  PyTorch may show warnings but will work")
            # Suppress warning
            warnings.filterwarnings('ignore', category=UserWarning, module='torch.cuda')
        
        return True
        
    except Exception as e:
        print(f"ERROR: GPU test failed: {e}")
        return False


def test_gpu_operation():
    """Test actual GPU computation"""
    print("\n" + "="*60)
    print("TEST 3: GPU Computation")
    print("="*60)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("SKIP: Skipped (no GPU)")
            return True
        
        # Create tensor on GPU
        x = torch.randn(1000, 1000, device='cuda')
        print(f"OK: Created tensor on GPU: {x.shape}")
        
        # Matrix multiplication
        y = torch.matmul(x, x)
        print(f"OK: Matrix multiplication: {y.shape}")
        
        # Check memory
        mem_allocated = torch.cuda.memory_allocated() / 1024**2
        mem_reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"OK: GPU memory allocated: {mem_allocated:.1f} MB")
        print(f"  GPU memory reserved: {mem_reserved:.1f} MB")
        
        # Cleanup
        del x, y
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"ERROR: GPU computation failed: {e}")
        return False


def test_whisper_model():
    """Test loading Whisper components"""
    print("\n" + "="*60)
    print("TEST 4: Whisper Model Components")
    print("="*60)
    
    try:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        print("OK: WhisperProcessor imported")
        print("OK: WhisperForConditionalGeneration imported")
        
        # Test processor (quick, no download)
        print("\nLoading processor (quick test)...")
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        print("OK: Processor loaded successfully")
        
        print("\nNOTE: Not loading full model (takes time)")
        print("  Run teacher/load_teacher_pytorch.py to test full model")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Whisper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset():
    """Test dataset availability"""
    print("\n" + "="*60)
    print("TEST 5: Dataset")
    print("="*60)
    
    import os
    import json
    
    dataset_path = "./preprocessing_data/phoaudiobook_100h"
    audio_path = os.path.join(dataset_path, "audio")
    metadata_path = os.path.join(dataset_path, "metadata.json")
    
    # Check audio folder
    if os.path.exists(audio_path):
        audio_files = [f for f in os.listdir(audio_path) if f.endswith('.wav')]
        print(f"OK: Audio folder exists: {len(audio_files)} files")
    else:
        print(f"ERROR: Audio folder not found: {audio_path}")
        return False
    
    # Check metadata
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        total_hours = metadata.get('total_duration_hours', 0)
        total_samples = metadata.get('total_samples', 0)
        print(f"OK: Metadata exists")
        print(f"  Total samples: {total_samples}")
        print(f"  Total duration: {total_hours:.2f} hours")
    else:
        print(f"WARNING: Metadata not found: {metadata_path}")
        print("  Dataset may not be fully processed")
        return False
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("QUICK SETUP VERIFICATION")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Import Libraries", test_imports()))
    results.append(("GPU Detection", test_gpu()))
    results.append(("GPU Computation", test_gpu_operation()))
    results.append(("Whisper Components", test_whisper_model()))
    results.append(("Dataset", test_dataset()))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{status:8} {name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED - SETUP IS READY!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Test full teacher model:")
        print("     python teacher/load_teacher_pytorch.py")
        print("\n  2. Update config:")
        print("     nano config/distillation_config.yaml")
        print("\n  3. Start training:")
        print("     python scripts/step1_generate_teacher_logits.py")
    else:
        print("WARNING: SOME TESTS FAILED - CHECK SETUP")
        print("="*60)
        print("\nFailed tests need to be fixed before training")
        sys.exit(1)
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
