"""Distillation dataset loader for offline knowledge distillation"""

import numpy as np
from pathlib import Path
import json


class DistillationDataset:
    def __init__(
        self,
        audio_dir: str,
        logits_dir: str,
        sample_rate: int = 16000,
        max_audio_length: float = 30.0,
        use_precomputed_mels: bool = True,
        mel_dir: str = None
    ):
        self.audio_dir = Path(audio_dir)
        self.logits_dir = Path(logits_dir)
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        self.max_samples = int(max_audio_length * sample_rate)
        self.use_precomputed_mels = use_precomputed_mels

        if use_precomputed_mels:
            self.mel_dir = Path(mel_dir) if mel_dir else self.audio_dir.parent / "mels"
        else:
            self.mel_dir = None

        metadata_path = self.logits_dir / "logits_metadata.json"
        self.metadata = json.load(open(metadata_path)) if metadata_path.exists() else {}

        self.logits_files = sorted(list(self.logits_dir.glob("*.npy")))

        self.audio_files = []
        self.mel_files = []
        for logits_file in self.logits_files:
            audio_name = logits_file.stem.replace("_logits", "")

            if use_precomputed_mels:
                mel_path = self.mel_dir / f"{audio_name}_mel.npy"
                if mel_path.exists():
                    self.mel_files.append(mel_path)
                    self.audio_files.append(None)
            else:
                audio_path = self.audio_dir / f"{audio_name}.npy"
                if not audio_path.exists():
                    audio_path = self.audio_dir / f"{audio_name}.wav"
                if audio_path.exists():
                    self.audio_files.append(audio_path)
                    self.mel_files.append(None)

        dataset_size = len(self.mel_files) if use_precomputed_mels else len(self.audio_files)
        mode = "pre-computed mels" if use_precomputed_mels else "raw audio"
        print(f"Found {dataset_size} {mode}/logits pairs")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx: int):
        if self.use_precomputed_mels:
            mel = np.load(self.mel_files[idx])
            sample_name = self.mel_files[idx].stem.replace("_mel", "")
            teacher_logits = np.load(self.logits_files[idx])

            text = self.metadata.get(sample_name, {}).get('text', '')
            tokens = self.metadata.get(sample_name, {}).get('tokens', None)
            if tokens is not None:
                tokens = np.array(tokens)

            return {
                'mel': mel,
                'teacher_logits': teacher_logits,
                'text': text,
                'tokens': tokens
            }
        else:
            audio_path = self.audio_files[idx]
            if audio_path.suffix == '.npy':
                audio = np.load(audio_path)
            else:
                import librosa
                audio, _ = librosa.load(audio_path, sr=self.sample_rate)

            if len(audio) > self.max_samples:
                audio = audio[:self.max_samples]
            elif len(audio) < self.max_samples:
                audio = np.pad(audio, (0, self.max_samples - len(audio)))

            teacher_logits = np.load(self.logits_files[idx])

            sample_name = audio_path.stem
            text = self.metadata.get(sample_name, {}).get('text', '')
            tokens = self.metadata.get(sample_name, {}).get('tokens', None)
            if tokens is not None:
                tokens = np.array(tokens)

            return {
                'audio': audio,
                'teacher_logits': teacher_logits,
                'text': text,
                'tokens': tokens
            }


def collate_fn_distillation(batch):
    audios = np.stack([sample['audio'] for sample in batch])

    max_logits_len = max(sample['teacher_logits'].shape[0] for sample in batch)
    vocab_size = batch[0]['teacher_logits'].shape[-1]

    padded_logits = []
    for sample in batch:
        logits = sample['teacher_logits']
        if logits.shape[0] < max_logits_len:
            pad_len = max_logits_len - logits.shape[0]
            logits = np.pad(logits, ((0, pad_len), (0, 0)), constant_values=-100)
        padded_logits.append(logits)

    teacher_logits = np.stack(padded_logits)
    texts = [sample['text'] for sample in batch]

    tokens = [sample.get('tokens') for sample in batch]
    if all(t is not None for t in tokens):
        max_token_len = max(len(t) for t in tokens)
        padded_tokens = []
        for t in tokens:
            if len(t) < max_token_len:
                t = np.pad(t, (0, max_token_len - len(t)), constant_values=0)
            padded_tokens.append(t)
        tokens = np.stack(padded_tokens)
    else:
        tokens = None

    return {
        'audio': audios,
        'teacher_logits': teacher_logits,
        'text': texts,
        'tokens': tokens
    }
