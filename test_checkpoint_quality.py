"""
Test Student Model Quality After Mini-Batch Training
Compares checkpoint performance vs OpenAI baseline
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "model"))
sys.path.insert(0, str(Path(__file__).parent / "distillation" / "student"))

from load_student_tensorflow import WhisperStudentTensorFlow


print("="*70)
print("TESTING CHECKPOINT QUALITY vs BASELINE")
print("="*70)


def load_audio(audio_path: str, sr: int = 16000):
    """Load audio and convert to mel spectrogram"""
    try:
        import librosa
    except ImportError:
        print("Error: librosa not installed")
        print("Install with: pip install librosa")
        sys.exit(1)
    
    # Load audio
    audio, _ = librosa.load(audio_path, sr=sr)
    
    # Pad or trim to 30 seconds
    target_length = sr * 30
    if len(audio) > target_length:
        audio = audio[:target_length]
    else:
        audio = np.pad(audio, (0, target_length - len(audio)))
    
    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=400,
        hop_length=160,
        n_mels=80,
        fmin=0,
        fmax=8000
    )
    
    # Trim to exactly 3000 frames (OpenAI Whisper spec)
    if mel.shape[1] > 3000:
        mel = mel[:, :3000]
    elif mel.shape[1] < 3000:
        mel = np.pad(mel, ((0, 0), (0, 3000 - mel.shape[1])), mode='constant')
    
    # Convert to log scale and normalize
    mel = np.log10(np.maximum(mel, 1e-10))
    mel = (mel + 4.0) / 4.0
    
    # Add batch dimension
    mel = mel[np.newaxis, ...]
    
    return tf.convert_to_tensor(mel, dtype=tf.float32)


def greedy_decode(model, audio_features, tokenizer, max_length: int = 224, language: str = "vi"):
    """Simple greedy decoding"""
    # Create initial tokens
    sot = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    lang = tokenizer.convert_tokens_to_ids(f"<|{language}|>")
    task = tokenizer.convert_tokens_to_ids("<|transcribe|>")
    no_timestamps = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    eot = tokenizer.eos_token_id
    
    tokens = [sot, lang, task, no_timestamps]
    tokens_tensor = tf.constant([tokens], dtype=tf.int32)
    
    # Autoregressive generation
    for i in range(max_length):
        logits = model.decoder(tokens_tensor, audio_features, training=False)
        next_token_logits = logits[0, -1, :]
        next_token = tf.argmax(next_token_logits, axis=-1, output_type=tf.int32)
        next_token_value = int(next_token.numpy())
        
        if next_token_value == eot:
            break
        
        tokens.append(next_token_value)
        tokens_tensor = tf.constant([tokens], dtype=tf.int32)
    
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return text, len(tokens) - 4  # Exclude initial tokens


def find_latest_checkpoint(checkpoint_dir: str = "./distillation/checkpoints"):
    """Find the latest checkpoint file"""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        return None
    
    # Find all .weights.h5 files
    checkpoints = list(checkpoint_path.glob("**/*.weights.h5"))
    
    if not checkpoints:
        return None
    
    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return str(checkpoints[0])


def get_checkpoint_info(checkpoint_path: str):
    """Get checkpoint metadata if available"""
    metadata_path = checkpoint_path.replace('.weights.h5', '_metadata.json')
    
    info = {
        'path': checkpoint_path,
        'filename': Path(checkpoint_path).name,
        'size_mb': Path(checkpoint_path).stat().st_size / (1024*1024),
        'modified': datetime.fromtimestamp(Path(checkpoint_path).stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            info.update(metadata)
    
    return info


def calculate_wer(reference: str, hypothesis: str):
    """Calculate Word Error Rate"""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    # Simple Levenshtein distance
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    
    if len(ref_words) == 0:
        return 0.0
    
    wer = d[len(ref_words)][len(hyp_words)] / len(ref_words)
    return wer * 100


def test_checkpoint_quality(
    checkpoint_path: str,
    audio_path: str,
    ground_truth: str = None,
    model_name: str = "base",
    language: str = "vi"
):
    """
    Test checkpoint quality vs baseline
    
    Args:
        checkpoint_path: Path to checkpoint .weights.h5 file
        audio_path: Path to test audio file
        ground_truth: Optional ground truth text for WER calculation
        model_name: Model size
        language: Language code
    """
    
    # Check files
    if not os.path.exists(checkpoint_path):
        print(f"\nError: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    if not os.path.exists(audio_path):
        print(f"\nError: Audio file not found: {audio_path}")
        sys.exit(1)
    
    # Load tokenizer
    print(f"\n[1] Loading Tokenizer")
    print("-" * 70)
    try:
        from transformers import WhisperProcessor
        processor = WhisperProcessor.from_pretrained("vinai/PhoWhisper-large")
        tokenizer = processor.tokenizer
        print(f"    Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)
    
    # Load audio
    print(f"\n[2] Loading Audio")
    print("-" * 70)
    print(f"    File: {audio_path}")
    mel = load_audio(audio_path)
    print(f"    Mel shape: {mel.shape}")
    
    # Test 1: Baseline (OpenAI pretrained)
    print(f"\n[3] Testing BASELINE (OpenAI Pretrained)")
    print("-" * 70)
    baseline = WhisperStudentTensorFlow(
        model_name=model_name,
        load_openai_weights=True
    )
    baseline_model = baseline.model
    
    audio_features = baseline_model.encoder(mel, training=False)
    baseline_text, baseline_tokens = greedy_decode(
        baseline_model, audio_features, tokenizer, language=language
    )
    
    print(f"    Tokens: {baseline_tokens}")
    print(f"    Text: {baseline_text}")
    
    # Test 2: Checkpoint (After training)
    print(f"\n[4] Testing CHECKPOINT (After Training)")
    print("-" * 70)
    
    # Get checkpoint info
    ckpt_info = get_checkpoint_info(checkpoint_path)
    print(f"    Checkpoint: {ckpt_info['filename']}")
    print(f"    Size: {ckpt_info['size_mb']:.1f} MB")
    print(f"    Modified: {ckpt_info['modified']}")
    if 'epoch' in ckpt_info:
        print(f"    Epoch: {ckpt_info['epoch']}, Step: {ckpt_info['step']}, Loss: {ckpt_info.get('loss', 'N/A')}")
    
    trained = WhisperStudentTensorFlow(
        model_name=model_name,
        weights_path=checkpoint_path,
        load_openai_weights=False
    )
    trained_model = trained.model
    
    audio_features = trained_model.encoder(mel, training=False)
    trained_text, trained_tokens = greedy_decode(
        trained_model, audio_features, tokenizer, language=language
    )
    
    print(f"    Tokens: {trained_tokens}")
    print(f"    Text: {trained_text}")
    
    # Comparison
    print(f"\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    print(f"\n{'Metric':<25} {'Baseline':<25} {'Checkpoint':<25}")
    print("-" * 70)
    print(f"{'Tokens Generated':<25} {baseline_tokens:<25} {trained_tokens:<25}")
    print(f"{'Text Length':<25} {len(baseline_text):<25} {len(trained_text):<25}")
    
    # Calculate similarity
    same_text = baseline_text == trained_text
    print(f"\n{'Same Output?':<25} {same_text}")
    
    if not same_text:
        # Show differences
        print(f"\n{'='*70}")
        print("TEXT COMPARISON")
        print("="*70)
        print(f"\nBaseline:   {baseline_text}")
        print(f"Checkpoint: {trained_text}")
    
    # WER if ground truth provided
    if ground_truth:
        print(f"\n{'='*70}")
        print("WORD ERROR RATE (WER)")
        print("="*70)
        
        baseline_wer = calculate_wer(ground_truth, baseline_text)
        trained_wer = calculate_wer(ground_truth, trained_text)
        
        print(f"\nGround Truth: {ground_truth}")
        print(f"\n{'Model':<25} {'WER (%)':<15} {'Status'}")
        print("-" * 70)
        print(f"{'Baseline':<25} {baseline_wer:>7.2f}%       {'-'}")
        print(f"{'Checkpoint':<25} {trained_wer:>7.2f}%       ", end="")
        
        if trained_wer < baseline_wer:
            print("✓ IMPROVED")
        elif trained_wer == baseline_wer:
            print("= SAME")
        else:
            print("✗ DEGRADED")
        
        wer_delta = trained_wer - baseline_wer
        print(f"\nDelta: {wer_delta:+.2f}%")
    
    # Verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print("="*70)
    
    if same_text:
        print("✓ Model produces SAME output as baseline")
        print("  → Training did not degrade quality")
        print("  → Safe to continue training with --resume")
    else:
        if ground_truth:
            if trained_wer <= baseline_wer * 1.1:  # Within 10% of baseline
                print("✓ Model output DIFFERENT but quality acceptable")
                print(f"  → WER within acceptable range ({trained_wer:.1f}% vs {baseline_wer:.1f}%)")
                print("  → Safe to continue training")
            else:
                print("⚠ Model output DIFFERENT and quality degraded")
                print(f"  → WER significantly worse ({trained_wer:.1f}% vs {baseline_wer:.1f}%)")
                print("  → Review training config before continuing")
        else:
            print("⚠ Model output DIFFERENT from baseline")
            print("  → Manual review recommended")
            print("  → Provide ground_truth for WER calculation")
    
    print("\n" + "="*70)
    
    return {
        'baseline_text': baseline_text,
        'trained_text': trained_text,
        'same_output': same_text,
        'baseline_wer': baseline_wer if ground_truth else None,
        'trained_wer': trained_wer if ground_truth else None
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test checkpoint quality vs baseline")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint .weights.h5 file (default: auto-detect latest)"
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to test audio file"
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Ground truth transcription for WER calculation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Model size (default: base)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="vi",
        help="Language code (default: vi)"
    )
    
    args = parser.parse_args()
    
    # Auto-detect checkpoint if not specified
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        print("\nAuto-detecting latest checkpoint...")
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            print("Error: No checkpoints found in ./distillation/checkpoints/")
            print("Please specify --checkpoint path manually")
            sys.exit(1)
        print(f"Found: {checkpoint_path}")
    
    # Run test
    test_checkpoint_quality(
        checkpoint_path=checkpoint_path,
        audio_path=args.audio,
        ground_truth=args.ground_truth,
        model_name=args.model,
        language=args.language
    )
