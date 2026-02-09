"""Pre-compute mel spectrograms for training (eliminates 3-5x CPU bottleneck)"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import librosa


def compute_mel_spectrogram(audio_path: str, sample_rate: int = 16000, n_mels: int = 80) -> np.ndarray:
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

    max_samples = sample_rate * 30
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    else:
        audio = np.pad(audio, (0, max_samples - len(audio)), mode='constant')

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=400,
        hop_length=160,
        n_mels=n_mels,
        power=2.0
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    return mel.astype(np.float32)


def precompute_mels(audio_dir: str, output_dir: str, sample_rate: int = 16000, n_mels: int = 80):
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(list(audio_dir.glob(f"**/*{ext}")))

    if len(audio_files) == 0:
        print(f"No audio files found in {audio_dir}")
        return

    for audio_path in tqdm(audio_files, desc="Computing mels"):
        try:
            mel = compute_mel_spectrogram(str(audio_path), sample_rate=sample_rate, n_mels=n_mels)
            output_path = output_dir / f"{audio_path.stem}_mel.npy"
            np.save(output_path, mel)
        except Exception as e:
            print(f"\nError: {audio_path.name}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", default="./distillation/preprocessing_data/phoaudiobook_100h/audio")
    parser.add_argument("--output_dir", default="./distillation/preprocessing_data/phoaudiobook_100h/mels")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=80)

    args = parser.parse_args()
    precompute_mels(args.audio_dir, args.output_dir, args.sample_rate, args.n_mels)


if __name__ == "__main__":
    main()
