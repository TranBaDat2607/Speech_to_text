"""Convert mels + logits to TFRecord format (2-3x faster training)"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import tensorflow as tf


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(mel, teacher_logits, tokens, text):
    mel_flat = mel.flatten().tobytes()
    logits_flat = teacher_logits.flatten().tobytes()
    tokens_flat = tokens.tobytes()

    feature = {
        'mel': _bytes_feature(mel_flat),
        'mel_height': _int64_feature(mel.shape[0]),
        'mel_width': _int64_feature(mel.shape[1]),
        'teacher_logits': _bytes_feature(logits_flat),
        'logits_seq_len': _int64_feature(teacher_logits.shape[0]),
        'logits_vocab_size': _int64_feature(teacher_logits.shape[1]),
        'tokens': _bytes_feature(tokens_flat),
        'tokens_len': _int64_feature(len(tokens)),
        'text': _bytes_feature(text.encode('utf-8')),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def convert_to_tfrecord(mel_dir: str, logits_dir: str, output_dir: str, samples_per_shard: int = 1000):
    mel_dir = Path(mel_dir)
    logits_dir = Path(logits_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not mel_dir.exists():
        print(f"ERROR: Mel directory not found: {mel_dir}")
        return

    metadata_path = logits_dir / "logits_metadata.json"
    metadata = json.load(open(metadata_path)) if metadata_path.exists() else {}

    logits_files = sorted(list(logits_dir.glob("*.npy")))
    if len(logits_files) == 0:
        print(f"ERROR: No logits files in {logits_dir}")
        return

    shard_idx = 0
    sample_count = 0
    writer = None
    skipped = 0

    for logits_file in tqdm(logits_files, desc="Converting"):
        try:
            audio_name = logits_file.stem.replace("_logits", "")
            mel_path = mel_dir / f"{audio_name}_mel.npy"

            if not mel_path.exists():
                skipped += 1
                continue

            mel = np.load(mel_path).astype(np.float32)
            teacher_logits = np.load(logits_file).astype(np.float32)

            if len(teacher_logits.shape) == 3 and teacher_logits.shape[0] == 1:
                teacher_logits = teacher_logits[0]

            if teacher_logits.shape[-1] > 50364:
                teacher_logits = teacher_logits[..., :50364]

            sample_info = metadata.get(audio_name, {})
            tokens = sample_info.get('tokens', [])

            if not tokens:
                skipped += 1
                continue

            tokens = np.array(tokens, dtype=np.int32)
            text = sample_info.get('text', '')

            if sample_count % samples_per_shard == 0:
                if writer:
                    writer.close()
                writer = tf.io.TFRecordWriter(str(output_dir / f"train-{shard_idx:05d}.tfrecord"))
                shard_idx += 1

            writer.write(serialize_example(mel, teacher_logits, tokens, text))
            sample_count += 1

        except Exception as e:
            print(f"\nError: {logits_file.name}: {e}")
            skipped += 1

    if writer:
        writer.close()

    print(f"\nConverted: {sample_count} samples ({shard_idx} shards), Skipped: {skipped}")

    info = {
        'num_samples': sample_count,
        'num_shards': shard_idx,
        'samples_per_shard': samples_per_shard,
        'mel_height': 80,
        'vocab_size': 50364,
    }
    json.dump(info, open(output_dir / "dataset_info.json", 'w'), indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mel_dir", default="./distillation/preprocessing_data/phoaudiobook_100h/mels")
    parser.add_argument("--logits_dir", default="./distillation/preprocessing_data/phoaudiobook_100h/teacher_logits")
    parser.add_argument("--output_dir", default="./distillation/preprocessing_data/phoaudiobook_100h/tfrecords")
    parser.add_argument("--samples_per_shard", type=int, default=1000)

    args = parser.parse_args()
    convert_to_tfrecord(args.mel_dir, args.logits_dir, args.output_dir, args.samples_per_shard)


if __name__ == "__main__":
    main()
