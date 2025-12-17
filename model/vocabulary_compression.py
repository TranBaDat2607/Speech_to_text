"""
Vocabulary Compression for Whisper

Implements hybrid approach:
1. Prune vocabulary to Vietnamese + English only (51,865 → ~15,000 tokens)
2. Factorize embedding matrix (15,000 × 512 → 15,000 × 64 + 64 × 512)

Total savings: 26.5M → 1.0M parameters (96% reduction!)
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Set, Optional
from pathlib import Path
import json


class VocabularyPruner:
    """
    Analyzes and prunes Whisper vocabulary to keep only necessary tokens

    For Vietnamese + English, we need:
    - Special tokens (~100 tokens)
    - Vietnamese language token
    - English language token
    - Timestamp tokens (~1,500 tokens)
    - BPE tokens for Vietnamese text (~5,000-8,000 tokens)
    - BPE tokens for English text (~5,000-8,000 tokens)
    - Common punctuation and symbols (~200 tokens)

    Total: ~15,000 tokens instead of 51,865 (71% reduction)
    """

    def __init__(self, keep_languages: List[str] = ['vi', 'en']):
        self.keep_languages = keep_languages
        self.original_vocab_size = 51865
        self.token_usage_stats = {}

    def analyze_token_usage(self, dataset_texts: List[str], tokenizer) -> Dict[int, int]:
        """
        Analyze which tokens are actually used in the dataset

        Args:
            dataset_texts: List of transcript texts from your Vietnamese dataset
            tokenizer: Whisper tokenizer

        Returns:
            Dict mapping token_id -> usage_count
        """
        print("Analyzing token usage in dataset...")
        token_counts = {}

        for text in dataset_texts:
            tokens = tokenizer.encode(text)
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1

        print(f"Found {len(token_counts)} unique tokens used in dataset")
        self.token_usage_stats = token_counts
        return token_counts

    def create_compact_vocabulary(self,
                                   tokenizer,
                                   token_usage: Optional[Dict[int, int]] = None) -> Dict[int, int]:
        """
        Create compact vocabulary mapping

        Args:
            tokenizer: Whisper tokenizer
            token_usage: Optional token usage statistics

        Returns:
            Dict mapping original_token_id -> compact_token_id
        """
        keep_tokens = set()

        # 1. Keep all special tokens
        for token_name, token_id in tokenizer.special_tokens.items():
            keep_tokens.add(token_id)
        print(f"Special tokens: {len(keep_tokens)}")

        # 2. Keep language tokens for specified languages
        for lang in self.keep_languages:
            lang_token = f"<|{lang}|>"
            if lang_token in tokenizer.special_tokens:
                keep_tokens.add(tokenizer.special_tokens[lang_token])
        print(f"After language tokens: {len(keep_tokens)}")

        # 3. Keep timestamp tokens (all of them for now)
        timestamp_start = tokenizer.special_tokens.get("<|0.00|>", 50364)
        for i in range(1501):  # Whisper has 1501 timestamp tokens
            keep_tokens.add(timestamp_start + i)
        print(f"After timestamp tokens: {len(keep_tokens)}")

        # 4. Keep actually used BPE tokens (if usage stats provided)
        if token_usage is not None:
            # Keep tokens used at least once
            for token_id in token_usage.keys():
                if token_id < timestamp_start:  # BPE tokens
                    keep_tokens.add(token_id)
            print(f"After used BPE tokens: {len(keep_tokens)}")
        else:
            # Keep all BPE tokens for now (conservative)
            for token_id in range(50257):  # GPT-2 vocabulary size
                keep_tokens.add(token_id)
            print(f"After all BPE tokens (conservative): {len(keep_tokens)}")

        # Create mapping: old_id -> new_id
        keep_tokens = sorted(list(keep_tokens))
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(keep_tokens)}

        print(f"\nVocabulary compression:")
        print(f"  Original: {self.original_vocab_size} tokens")
        print(f"  Compact: {len(keep_tokens)} tokens")
        print(f"  Reduction: {(1 - len(keep_tokens)/self.original_vocab_size)*100:.1f}%")

        return old_to_new, keep_tokens

    def save_vocabulary_mapping(self,
                                 old_to_new: Dict[int, int],
                                 keep_tokens: List[int],
                                 output_path: str = "compact_vocab_mapping.json"):
        """Save vocabulary mapping for later use"""
        mapping_data = {
            'old_to_new': old_to_new,
            'keep_tokens': keep_tokens,
            'original_vocab_size': self.original_vocab_size,
            'compact_vocab_size': len(keep_tokens),
            'languages': self.keep_languages
        }

        with open(output_path, 'w') as f:
            json.dump(mapping_data, f, indent=2)

        print(f"Vocabulary mapping saved to: {output_path}")


class FactorizedEmbedding(tf.keras.layers.Layer):
    """
    Factorized Embedding Layer

    Instead of: vocab_size × d_model (e.g., 15,000 × 512 = 7.7M params)
    Use: (vocab_size × bottleneck) + (bottleneck × d_model)
         (15,000 × 64) + (64 × 512) = 0.96M + 0.03M = 1.0M params

    Reduction: 7.7M → 1.0M = 87% savings!
    Accuracy loss: Typically <0.5% WER
    """

    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 bottleneck_dim: int = 64,
                 name: str = "factorized_embedding"):
        super().__init__(name=name)

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim

        # Low-rank factorization: V × D = (V × B) × (B × D)
        self.embed_low = tf.keras.layers.Embedding(
            vocab_size,
            bottleneck_dim,
            name="embed_low"
        )

        self.project_high = tf.keras.layers.Dense(
            d_model,
            use_bias=False,  # No bias to match standard embedding
            name="project_high"
        )

        # Calculate parameter savings
        original_params = vocab_size * d_model
        factorized_params = vocab_size * bottleneck_dim + bottleneck_dim * d_model
        savings = original_params - factorized_params
        savings_pct = (savings / original_params) * 100

        print(f"Factorized Embedding initialized:")
        print(f"  Original params: {original_params:,} ({original_params/1e6:.1f}M)")
        print(f"  Factorized params: {factorized_params:,} ({factorized_params/1e6:.2f}M)")
        print(f"  Savings: {savings:,} ({savings_pct:.1f}%)")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Forward pass

        Args:
            x: Token IDs [batch, seq_len]

        Returns:
            Embeddings [batch, seq_len, d_model]
        """
        # First embedding: token_ids → bottleneck
        x = self.embed_low(x)  # [batch, seq_len, bottleneck_dim]

        # Project to full dimension
        x = self.project_high(x)  # [batch, seq_len, d_model]

        return x

    def get_config(self):
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'bottleneck_dim': self.bottleneck_dim
        }


class CompactFactorizedEmbedding(tf.keras.layers.Layer):
    """
    Hybrid approach: Compact vocabulary + Factorized embedding

    Step 1: Prune vocab (51,865 → 15,000)
    Step 2: Factorize (15,000 × 64 + 64 × 512)

    Total: 26.5M → 1.0M params (96% reduction!)
    """

    def __init__(self,
                 compact_vocab_size: int,
                 d_model: int,
                 bottleneck_dim: int = 64,
                 old_to_new_mapping: Optional[Dict[int, int]] = None,
                 name: str = "compact_factorized_embedding"):
        super().__init__(name=name)

        self.compact_vocab_size = compact_vocab_size
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim

        # Factorized embedding on compact vocabulary
        self.factorized_emb = FactorizedEmbedding(
            compact_vocab_size,
            d_model,
            bottleneck_dim
        )

        # Mapping from original token IDs to compact token IDs
        if old_to_new_mapping is not None:
            # Create lookup table
            max_old_id = max(old_to_new_mapping.keys())
            mapping_array = np.full(max_old_id + 1, -1, dtype=np.int32)
            for old_id, new_id in old_to_new_mapping.items():
                mapping_array[old_id] = new_id

            self.id_mapping = tf.constant(mapping_array, dtype=tf.int32)
            self.has_mapping = True
        else:
            self.has_mapping = False

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Forward pass with automatic token ID remapping

        Args:
            x: Original token IDs [batch, seq_len]

        Returns:
            Embeddings [batch, seq_len, d_model]
        """
        if self.has_mapping:
            # Remap token IDs from original to compact vocabulary
            x = tf.gather(self.id_mapping, x)

        # Get embeddings
        return self.factorized_emb(x)

    def get_weights_for_output_projection(self):
        """
        Get weights for tied output projection (decoder head)

        Returns embedding matrix of shape [compact_vocab_size, d_model]
        """
        # Compute full embedding matrix by passing all token IDs
        all_token_ids = tf.range(self.compact_vocab_size, dtype=tf.int32)
        all_token_ids = tf.expand_dims(all_token_ids, 0)  # [1, vocab_size]

        # Get embeddings
        embeddings = self.factorized_emb(all_token_ids)  # [1, vocab_size, d_model]
        embeddings = tf.squeeze(embeddings, axis=0)  # [vocab_size, d_model]

        return embeddings


def analyze_dataset_vocabulary(dataset_dir: str, num_samples: int = 1000):
    """
    Analyze which tokens are actually used in your Vietnamese dataset

    Args:
        dataset_dir: Path to your dataset
        num_samples: Number of samples to analyze

    Returns:
        Dict of token usage statistics
    """
    from tokenizer import get_tokenizer
    import json

    print("Analyzing vocabulary usage in dataset...")
    print(f"Dataset: {dataset_dir}")
    print(f"Samples: {num_samples}")

    tokenizer = get_tokenizer(multilingual=True, language='vi', task='transcribe')

    # Load transcripts
    dataset_path = Path(dataset_dir)
    json_files = list(dataset_path.glob("*.json"))[:num_samples]

    print(f"Found {len(json_files)} JSON files")

    texts = []
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'text' in data:
                texts.append(data['text'])

    print(f"Loaded {len(texts)} transcripts")

    # Analyze token usage
    pruner = VocabularyPruner(keep_languages=['vi', 'en'])
    token_usage = pruner.analyze_token_usage(texts, tokenizer)

    # Create compact vocabulary
    old_to_new, keep_tokens = pruner.create_compact_vocabulary(tokenizer, token_usage)

    # Save mapping
    pruner.save_vocabulary_mapping(old_to_new, keep_tokens)

    return {
        'original_vocab_size': 51865,
        'compact_vocab_size': len(keep_tokens),
        'reduction_pct': (1 - len(keep_tokens)/51865) * 100,
        'token_usage': token_usage,
        'old_to_new_mapping': old_to_new,
        'keep_tokens': keep_tokens
    }


if __name__ == "__main__":
    print("="*60)
    print("Vocabulary Compression Analysis")
    print("="*60)

    # Example: Analyze your dataset
    # results = analyze_dataset_vocabulary("../datasets/sachnoivietnam15/dataset")

    print("\nTo use this:")
    print("1. Run analysis on your Vietnamese dataset")
    print("2. Create compact vocabulary mapping")
    print("3. Use CompactFactorizedEmbedding in model")
    print("4. Retrain with 96% fewer embedding parameters!")
