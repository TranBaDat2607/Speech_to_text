"""
Verify Whisper Large teacher model on sample audio
"""

import os
import sys
import torch
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from teacher.load_teacher import WhisperTeacher


def verify_on_audio(
    audio_path: str,
    model_name: str = "large",
    language: str = "vi",
    device: str = None
):
    """
    Verify teacher model by transcribing an audio file
    
    Args:
        audio_path: Path to audio file
        model_name: Whisper model size
        language: Language code
        device: Device to use (auto-detect if None)
    """
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*60}")
    print(f"VERIFYING TEACHER MODEL ON AUDIO")
    print(f"{'='*60}")
    print(f"  Audio file: {audio_path}")
    print(f"  Model: {model_name}")
    print(f"  Language: {language}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")
    
    teacher = WhisperTeacher(
        model_name=model_name,
        device=device
    )
    
    print(f"\n🎙️  Transcribing audio...")
    result = teacher.transcribe_sample(
        audio_path=audio_path,
        language=language,
        task="transcribe",
        verbose=False
    )
    
    print(f"\n{'='*60}")
    print("TRANSCRIPTION RESULT")
    print(f"{'='*60}")
    print(f"\n{result['text']}\n")
    print(f"{'='*60}")
    print(f"  Language detected: {result.get('language', 'N/A')}")
    
    if 'segments' in result and len(result['segments']) > 0:
        print(f"  Number of segments: {len(result['segments'])}")
        print(f"\n  First segment:")
        seg = result['segments'][0]
        print(f"    Start: {seg['start']:.2f}s")
        print(f"    End: {seg['end']:.2f}s")
        print(f"    Text: {seg['text']}")
    
    print(f"{'='*60}\n")
    
    return result


def verify_on_dataset_sample(
    dataset_dir: str,
    num_samples: int = 3,
    model_name: str = "large",
    language: str = "vi"
):
    """
    Verify teacher on multiple samples from dataset
    
    Args:
        dataset_dir: Path to dataset directory
        num_samples: Number of samples to test
        model_name: Whisper model size
        language: Language code
    """
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return
    
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac']:
        audio_files.extend(Path(dataset_dir).rglob(ext))
    
    if len(audio_files) == 0:
        print(f"Error: No audio files found in {dataset_dir}")
        return
    
    audio_files = audio_files[:num_samples]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*60}")
    print(f"VERIFYING ON {len(audio_files)} DATASET SAMPLES")
    print(f"{'='*60}\n")
    
    teacher = WhisperTeacher(
        model_name=model_name,
        device=device
    )
    
    results = []
    
    for idx, audio_path in enumerate(audio_files, 1):
        print(f"\n--- Sample {idx}/{len(audio_files)} ---")
        print(f"File: {audio_path.name}")
        
        result = teacher.transcribe_sample(
            audio_path=str(audio_path),
            language=language,
            task="transcribe",
            verbose=False
        )
        
        print(f"Transcription: {result['text'][:100]}...")
        results.append(result)
    
    print(f"\n{'='*60}")
    print(f"✓ Verified {len(results)} samples successfully!")
    print(f"{'='*60}\n")
    
    return results


def main():
    """Main verification script"""
    
    parser = argparse.ArgumentParser(
        description="Verify Whisper Large teacher model"
    )
    parser.add_argument(
        "--audio",
        type=str,
        help="Path to audio file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to dataset directory (will test on samples)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples to test from dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="large",
        choices=["large", "large-v2", "large-v3"],
        help="Whisper model size"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="vi",
        help="Language code"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Device to use (auto-detect if not specified)"
    )
    
    args = parser.parse_args()
    
    if args.audio:
        verify_on_audio(
            audio_path=args.audio,
            model_name=args.model,
            language=args.language,
            device=args.device
        )
    
    elif args.dataset:
        verify_on_dataset_sample(
            dataset_dir=args.dataset,
            num_samples=args.num_samples,
            model_name=args.model,
            language=args.language
        )
    
    else:
        print("Error: Please specify --audio or --dataset")
        parser.print_help()


if __name__ == "__main__":
    main()
