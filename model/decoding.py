"""
TensorFlow implementation of Whisper Decoding with Beam Search
Converted from OpenAI Whisper PyTorch implementation
"""

import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

@dataclass(frozen=True)
class DecodingOptions:
    """
    Controls decoding behavior (beam search, greedy, temperature, etc.)
    """
    task: str = "transcribe"
    language: Optional[str] = None
    
    temperature: float = 0.0
    sample_len: Optional[int] = None
    best_of: Optional[int] = None
    beam_size: Optional[int] = None
    patience: Optional[float] = None
    
    length_penalty: Optional[float] = None
    
    prompt: Optional[Union[str, List[int]]] = None
    prefix: Optional[Union[str, List[int]]] = None
    
    suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"
    suppress_blank: bool = True
    
    without_timestamps: bool = False
    max_initial_timestamp: Optional[float] = 1.0
    
    fp16: bool = True

@dataclass(frozen=True)
class DecodingResult:
    """
    Result from decoding operation matching OpenAI Whisper
    """
    audio_features: tf.Tensor
    language: str
    language_probs: Optional[Dict[str, float]] = None
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    avg_logprob: float = np.nan
    no_speech_prob: float = np.nan
    temperature: float = np.nan
    compression_ratio: float = np.nan

class TokenDecoder:
    """
    Base class for token decoding strategies (Greedy, BeamSearch)
    """
    
    def reset(self):
        """Initialize stateful variables for decoding a new sequence"""
        pass
    
    def update(self, 
               tokens: tf.Tensor, 
               logits: tf.Tensor, 
               sum_logprobs: tf.Tensor) -> Tuple[tf.Tensor, bool]:
        """
        Select next token based on current trace and logits
        
        Args:
            tokens: [n_batch, current_sequence_length] - all tokens so far
            logits: [n_batch, vocab_size] - probability distribution
            sum_logprobs: [n_batch] - cumulative log probabilities
            
        Returns:
            tokens: [n_batch, current_sequence_length + 1] - with next token
            completed: bool - True if all sequences reached EOT
        """
        raise NotImplementedError
    
    def finalize(self, 
                 tokens: tf.Tensor, 
                 sum_logprobs: tf.Tensor) -> Tuple[Sequence[Sequence[tf.Tensor]], List[List[float]]]:
        """
        Finalize search and return final candidate sequences
        
        Args:
            tokens: [n_audio, n_group, current_sequence_length]
            sum_logprobs: [n_audio, n_group]
            
        Returns:
            tokens: Sequence of candidate token sequences for each audio
            sum_logprobs: Corresponding cumulative log probabilities
        """
        raise NotImplementedError


class GreedyDecoder(TokenDecoder):
    """
    Greedy decoding strategy 
    """
    
    def __init__(self, temperature: float, eot: int):
        """
        Args:
            temperature: Sampling temperature (0 = greedy, >0 = sampling)
            eot: End-of-text token ID
        """
        self.temperature = temperature
        self.eot = eot
    
    def update(self, 
               tokens: tf.Tensor, 
               logits: tf.Tensor, 
               sum_logprobs: tf.Tensor) -> Tuple[tf.Tensor, bool]:
        if self.temperature == 0:
            next_tokens = tf.argmax(logits, axis=-1, output_type=tf.int32)
        else:
            next_tokens = tf.random.categorical(
                logits / self.temperature, 
                num_samples=1,
                dtype=tf.int32
            )[:, 0]
        
        # Compute log probabilities
        logprobs = tf.nn.log_softmax(tf.cast(logits, tf.float32), axis=-1)
        
        # Get log probability of selected tokens
        batch_size = tf.shape(logprobs)[0]
        batch_indices = tf.range(batch_size, dtype=tf.int32)
        indices = tf.stack([batch_indices, next_tokens], axis=1)
        current_logprobs = tf.gather_nd(logprobs, indices)
        
        # Update cumulative log probabilities (only for non-EOT sequences)
        not_eot_mask = tf.cast(tokens[:, -1] != self.eot, tf.float32)
        sum_logprobs = sum_logprobs + current_logprobs * not_eot_mask
        
        # Keep EOT token for already completed sequences
        is_eot = tokens[:, -1] == self.eot
        next_tokens = tf.where(is_eot, self.eot, next_tokens)
        
        # Append next tokens to sequence
        next_tokens_expanded = tf.expand_dims(next_tokens, axis=1)  # [batch] -> [batch, 1]
        tokens = tf.concat([tokens, next_tokens_expanded], axis=-1)
        
        # Check if all sequences completed
        completed = tf.reduce_all(tokens[:, -1] == self.eot)
        
        return tokens, completed
    
    def finalize(self, 
                 tokens: tf.Tensor, 
                 sum_logprobs: tf.Tensor) -> Tuple[tf.Tensor, List[List[float]]]:
        # Ensure each sequence has at least one EOT token at the end
        # Handle both 2D [batch, seq_len] and 3D [n_audio, n_group, seq_len]
        if len(tokens.shape) == 2:
            # 2D: [batch, seq_len]
            tokens = tf.pad(
                tokens, 
                paddings=[[0, 0], [0, 1]], 
                constant_values=self.eot
            )
        elif len(tokens.shape) == 3:
            # 3D: [n_audio, n_group, seq_len]
            tokens = tf.pad(
                tokens,
                paddings=[[0, 0], [0, 0], [0, 1]],
                constant_values=self.eot
            )
        else:
            raise ValueError(f"Unexpected tokens shape: {tokens.shape}")
        
        # Convert sum_logprobs to list format
        sum_logprobs_list = sum_logprobs.numpy().tolist() if hasattr(sum_logprobs, 'numpy') else sum_logprobs
        
        return tokens, sum_logprobs_list


class Inference:
    """
    Base class for inference with model forward pass and KV cache management
    Matching OpenAI Whisper Inference interface
    """
    
    def logits(self, tokens: tf.Tensor, audio_features: tf.Tensor) -> tf.Tensor:
        """Perform forward pass on decoder and return per-token logits"""
        raise NotImplementedError
    
    def rearrange_kv_cache(self, source_indices: List[int]) -> None:
        """Update key-value cache according to updated beams"""
        raise NotImplementedError
    
    def cleanup_caching(self) -> None:
        """Clean up any resources or hooks after decoding is finished"""
        pass


class BeamSearchDecoder(TokenDecoder):
    """
    Beam Search decoding strategy
    Implements beam search with patience for efficient diverse candidate generation
    """
    
    def __init__(self,
                 beam_size: int,
                 eot: int,
                 inference: Inference,
                 patience: Optional[float] = None):
        """
        Args:
            beam_size: Number of beams to maintain during search
            eot: End-of-text token ID
            inference: Inference object for model forward passes
            patience: Patience factor (arxiv:2204.05424). Higher = more candidates
        """
        self.beam_size = beam_size
        self.eot = eot
        self.inference = inference
        self.patience = patience or 1.0
        self.max_candidates = round(beam_size * self.patience)
        self.finished_sequences = None
        
        assert self.max_candidates > 0, \
            f"Invalid beam size ({beam_size}) or patience ({patience})"
    
    def reset(self):
        """
        Initialize stateful variables for decoding a new sequence
        """
        self.finished_sequences = None
    
    def update(self,
               tokens: tf.Tensor,
               logits: tf.Tensor,
               sum_logprobs: tf.Tensor) -> Tuple[tf.Tensor, bool]:
        # Validate batch size
        batch_size = tf.shape(tokens)[0]
        if batch_size % self.beam_size != 0:
            raise ValueError(f"tokens.shape[0]={batch_size} % beam_size={self.beam_size} != 0")
        
        n_audio = batch_size // self.beam_size
        
        # Initialize finished_sequences on first update
        if self.finished_sequences is None:
            self.finished_sequences = [{} for _ in range(n_audio)]
        
        # Compute log probabilities
        logprobs = tf.nn.log_softmax(tf.cast(logits, tf.float32), axis=-1)
        
        next_tokens = []
        source_indices = []
        finished_sequences = []
        
        # Process each audio sequence independently
        for i in range(n_audio):
            scores = {}  # sequence -> cumulative log probability
            sources = {}  # sequence -> source beam index
            finished = {}  # finished sequences for this audio
            
            # STEP 1: Calculate cumulative log probabilities for all possible candidates
            for j in range(self.beam_size):
                idx = i * self.beam_size + j
                
                # Get current token sequence as list
                prefix = tokens[idx].numpy().tolist()
                
                # Get top-k tokens for this beam
                beam_logprobs = logprobs[idx]
                top_k_logprobs, top_k_indices = tf.nn.top_k(
                    beam_logprobs, 
                    k=min(self.beam_size + 1, tf.shape(beam_logprobs)[0])
                )
                
                for logprob, token in zip(top_k_logprobs.numpy(), top_k_indices.numpy()):
                    new_logprob = float(sum_logprobs[idx].numpy() + logprob)
                    sequence = tuple(prefix + [int(token)])
                    
                    scores[sequence] = new_logprob
                    sources[sequence] = idx
            
            # STEP 2: Rank candidates and keep top beam_size sequences
            saved = 0
            for sequence in sorted(scores, key=scores.get, reverse=True):
                if sequence[-1] == self.eot:
                    finished[sequence] = scores[sequence]
                else:
                    sum_logprobs = tf.tensor_scatter_nd_update(
                        sum_logprobs,
                        indices=[[len(next_tokens)]],
                        updates=[scores[sequence]]
                    )
                    next_tokens.append(sequence)
                    source_indices.append(sources[sequence])
                    
                    saved += 1
                    if saved == self.beam_size:
                        break
            
            finished_sequences.append(finished)
        
        # Convert next_tokens to tensor
        if len(next_tokens) > 0:
            tokens = tf.constant(next_tokens, dtype=tf.int32)
        else:
            tokens = tf.zeros((0, tf.shape(tokens)[1] + 1), dtype=tf.int32)
        
        # Rearrange KV cache according to beam selection
        self.inference.rearrange_kv_cache(source_indices)
        
        # Add newly finished sequences to accumulated finished sequences
        assert len(self.finished_sequences) == len(finished_sequences)
        for previously_finished, newly_finished in zip(self.finished_sequences, finished_sequences):
            # Add in order of score (highest first)
            for seq in sorted(newly_finished, key=newly_finished.get, reverse=True):
                if len(previously_finished) >= self.max_candidates:
                    break  # Candidate list is full
                previously_finished[seq] = newly_finished[seq]
        
        # Check if all audio sequences have enough candidates
        completed = all(
            len(sequences) >= self.max_candidates
            for sequences in self.finished_sequences
        )
        
        return tokens, completed
    
    def finalize(self,
                 preceding_tokens: tf.Tensor,
                 sum_logprobs: tf.Tensor) -> Tuple[List[List[tf.Tensor]], List[List[float]]]:
        """
        Finalize beam search and return final candidate sequences
        Args:
            preceding_tokens: [n_audio, beam_size, seq_len] - unfinished sequences
            sum_logprobs: [n_audio, beam_size] - cumulative log probs
            
        Returns:
            tokens: List of candidate sequences for each audio
            sum_logprobs: Corresponding log probabilities
        """
        # Convert to numpy for easier manipulation
        sum_logprobs_np = sum_logprobs.numpy()
        
        # Collect all finished sequences, add unfinished ones if needed
        for i, sequences in enumerate(self.finished_sequences):
            if len(sequences) < self.beam_size:
                # Not enough finished sequences - add best unfinished ones
                sorted_indices = np.argsort(sum_logprobs_np[i])[::-1]  # Descending order
                
                for j in sorted_indices:
                    sequence = tuple(preceding_tokens[i, j].numpy().tolist() + [self.eot])
                    sequences[sequence] = float(sum_logprobs_np[i][j])
                    
                    if len(sequences) >= self.beam_size:
                        break
        
        # Convert finished sequences to list of tensors
        tokens = [
            [tf.constant(list(seq), dtype=tf.int32) for seq in sequences.keys()]
            for sequences in self.finished_sequences
        ]
        
        # Extract log probabilities
        sum_logprobs_list = [
            list(sequences.values())
            for sequences in self.finished_sequences
        ]
        
        return tokens, sum_logprobs_list


class LogitFilter:
    """
    Base class for logit filtering operations
    Matching OpenAI Whisper LogitFilter interface
    """
    
    def apply(self, logits: tf.Tensor, tokens: tf.Tensor) -> None:
        """
        Apply filtering or masking to logits in-place
        Args:
            logits: [n_batch, vocab_size] - per-token logits
            tokens: [n_batch, current_sequence_length] - all tokens so far
        """
        raise NotImplementedError


class SuppressTokens(LogitFilter):
    """
    Suppress specific tokens by setting their logits to -inf
    Matching OpenAI Whisper SuppressTokens exactly
    """
    
    def __init__(self, suppress_tokens: Sequence[int]):
        """
        Args:
            suppress_tokens: List of token IDs to suppress
        """
        self.suppress_tokens = list(suppress_tokens)
    
    def apply(self, logits: tf.Tensor, tokens: tf.Tensor) -> tf.Tensor:
        if len(self.suppress_tokens) == 0:
            return logits
        
        batch_size = tf.shape(logits)[0]
        batch_indices = tf.repeat(
            tf.range(batch_size, dtype=tf.int32),
            repeats=len(self.suppress_tokens)
        )
        token_indices = tf.tile(
            tf.constant(self.suppress_tokens, dtype=tf.int32),
            multiples=[batch_size]
        )
        indices = tf.stack([batch_indices, token_indices], axis=1)
        
        updates = tf.fill([tf.shape(indices)[0]], -np.inf)
        logits = tf.tensor_scatter_nd_update(logits, indices, updates)
        
        return logits


class SuppressBlank(LogitFilter):
    """
    Suppress blank output at the beginning of sampling
    """
    
    def __init__(self, tokenizer, sample_begin: int):
        """
        Args:
            tokenizer: Tokenizer instance with encode() and eot attribute
            sample_begin: Index where sampling begins
        """
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
    
    def apply(self, logits: tf.Tensor, tokens: tf.Tensor) -> tf.Tensor:
        seq_len = tf.shape(tokens)[1]
        
        if seq_len == self.sample_begin:
            blank_tokens = self.tokenizer.encode(" ")
            suppress_tokens = blank_tokens + [self.tokenizer.eot]
            
            # Apply suppression
            batch_size = tf.shape(logits)[0]
            
            batch_indices = tf.repeat(
                tf.range(batch_size, dtype=tf.int32),
                repeats=len(suppress_tokens)
            )
            token_indices = tf.tile(
                tf.constant(suppress_tokens, dtype=tf.int32),
                multiples=[batch_size]
            )
            indices = tf.stack([batch_indices, token_indices], axis=1)
            
            updates = tf.fill([tf.shape(indices)[0]], -np.inf)
            logits = tf.tensor_scatter_nd_update(logits, indices, updates)
        
        return logits


class ApplyTimestampRules(LogitFilter):
    """
    Apply timestamp generation rules when without_timestamps=False
    Enforces timestamp pairing and suppresses <|notimestamps|>
    """
    
    def __init__(self, tokenizer, sample_begin: int, max_initial_timestamp_index: Optional[int]):
        """
        Args:
            tokenizer: Tokenizer instance
            sample_begin: Index where sampling begins
            max_initial_timestamp_index: Max timestamp index for initial timestamp
        """
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp_index = max_initial_timestamp_index
    
    def apply(self, logits: tf.Tensor, tokens: tf.Tensor) -> tf.Tensor:
        """
        Apply timestamp rules to logits
        """
        # Suppress <|notimestamps|> which is handled by without_timestamps
        if hasattr(self.tokenizer, 'no_timestamps') and self.tokenizer.no_timestamps is not None:
            no_timestamps_idx = self.tokenizer.no_timestamps
            batch_size = tf.shape(logits)[0]
            indices = tf.stack([
                tf.range(batch_size, dtype=tf.int32),
                tf.fill([batch_size], no_timestamps_idx)
            ], axis=1)
            updates = tf.fill([batch_size], -np.inf)
            logits = tf.tensor_scatter_nd_update(logits, indices, updates)
        
        # Get timestamp_begin token
        timestamp_begin = self.tokenizer.timestamp_begin if hasattr(self.tokenizer, 'timestamp_begin') else 50364
        eot = self.tokenizer.eot if hasattr(self.tokenizer, 'eot') else 50257
        
        # Timestamps have to appear in pairs, except directly before EOT
        # Convert to numpy for easier processing (this runs in eager mode during inference)
        batch_size = tf.shape(logits)[0]  # For TF operations
        batch_size_np = int(tokens.shape[0]) if tokens.shape[0] is not None else 1  # For Python loops
        
        for k in range(batch_size_np):
            sampled_tokens = tokens[k, self.sample_begin:]
            seq = sampled_tokens.numpy().tolist() if hasattr(sampled_tokens, 'numpy') else list(sampled_tokens)
            
            last_was_timestamp = len(seq) >= 1 and seq[-1] >= timestamp_begin
            penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= timestamp_begin
            
            if last_was_timestamp:
                if penultimate_was_timestamp:  # has to be non-timestamp
                    # Suppress all timestamp tokens
                    logits = tf.tensor_scatter_nd_update(
                        logits,
                        [[k, i] for i in range(timestamp_begin, tf.shape(logits)[1])],
                        tf.fill([tf.shape(logits)[1] - timestamp_begin], -np.inf)
                    )
                else:  # cannot be normal text tokens
                    # Suppress all non-timestamp tokens before EOT
                    logits = tf.tensor_scatter_nd_update(
                        logits,
                        [[k, i] for i in range(eot)],
                        tf.fill([eot], -np.inf)
                    )
            
            # Timestamps shouldn't decrease
            timestamps = tf.boolean_mask(sampled_tokens, sampled_tokens >= timestamp_begin)
            if tf.size(timestamps) > 0:
                if last_was_timestamp and not penultimate_was_timestamp:
                    timestamp_last = timestamps[-1]
                else:
                    timestamp_last = timestamps[-1] + 1
                
                # Suppress timestamps up to timestamp_last
                num_suppress = timestamp_last - timestamp_begin
                if num_suppress > 0:
                    logits = tf.tensor_scatter_nd_update(
                        logits,
                        [[k, i] for i in range(timestamp_begin, timestamp_last)],
                        tf.fill([num_suppress], -np.inf)
                    )
        
        # At the beginning, force timestamp token
        seq_len = tf.shape(tokens)[1]
        if seq_len == self.sample_begin:
            # Suppress generating non-timestamp tokens at the beginning
            batch_indices = tf.repeat(tf.range(batch_size, dtype=tf.int32), repeats=timestamp_begin)
            token_indices = tf.tile(tf.range(timestamp_begin, dtype=tf.int32), multiples=[batch_size])
            indices = tf.stack([batch_indices, token_indices], axis=1)
            updates = tf.fill([tf.shape(indices)[0]], -np.inf)
            logits = tf.tensor_scatter_nd_update(logits, indices, updates)
            
            # Apply max_initial_timestamp option
            if self.max_initial_timestamp_index is not None:
                last_allowed = timestamp_begin + self.max_initial_timestamp_index
                vocab_size = tf.shape(logits)[1]
                num_suppress = vocab_size - last_allowed - 1
                if num_suppress > 0:
                    batch_indices = tf.repeat(tf.range(batch_size, dtype=tf.int32), repeats=num_suppress)
                    token_indices = tf.tile(tf.range(last_allowed + 1, vocab_size, dtype=tf.int32), multiples=[batch_size])
                    indices = tf.stack([batch_indices, token_indices], axis=1)
                    updates = tf.fill([tf.shape(indices)[0]], -np.inf)
                    logits = tf.tensor_scatter_nd_update(logits, indices, updates)
        
        # If sum of probability over timestamps is above any other token, sample timestamp
        logprobs = tf.nn.log_softmax(tf.cast(logits, tf.float32), axis=-1)
        for k in range(batch_size_np):
            timestamp_logprob = tf.reduce_logsumexp(logprobs[k, timestamp_begin:])
            max_text_token_logprob = tf.reduce_max(logprobs[k, :timestamp_begin])
            if timestamp_logprob > max_text_token_logprob:
                # Suppress all non-timestamp tokens
                logits = tf.tensor_scatter_nd_update(
                    logits,
                    [[k, i] for i in range(timestamp_begin)],
                    tf.fill([timestamp_begin], -np.inf)
                )
        
        return logits


class TensorFlowInference(Inference):
    """
    Manages KV cache for efficient autoregressive generation
    """
    
    def __init__(self, model, initial_token_length: int):
        """
        Args:
            model: Whisper model instance
            initial_token_length: Length of initial prompt tokens
        """
        self.model = model
        self.initial_token_length = initial_token_length
        self.kv_cache = {}

        # KV cache is managed directly in the decoder forward pass
        # Store references to key/value layers for cache management
        self.kv_modules = []
        for block in self.model.decoder.blocks:
            self.kv_modules.append(block.cross_attn.key if hasattr(block, 'cross_attn') and block.cross_attn else None)
            self.kv_modules.append(block.cross_attn.value if hasattr(block, 'cross_attn') and block.cross_attn else None)
        # Filter out None values
        self.kv_modules = [m for m in self.kv_modules if m is not None]
    
    def logits(self, tokens: tf.Tensor, audio_features: tf.Tensor) -> tf.Tensor:
        """
        Perform forward pass through decoder with KV caching
        
        Args:
            tokens: [batch_size, seq_len] - token indices
            audio_features: [batch_size, n_audio_ctx, n_audio_state] - encoder output
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        # Initialize KV cache on first call
        if not self.kv_cache:
            self.kv_cache = {}
        
        # Debug: Check input tokens
        print(f"[DEBUG TFInference.logits START] input tokens.shape={tokens.shape}")
        
        # NOTE: KV cache disabled - decoder blocks need proper implementation
        # TODO: Implement KV cache hooks in decoder blocks like OpenAI Whisper
        # For now, process full sequence each time (slower but correct)
        
        # Forward pass through decoder WITHOUT KV cache
        logits_output = self.model.decoder(tokens, audio_features, kv_cache=None, training=False)
        
        print(f"[DEBUG TFInference.logits END] output logits.shape={logits_output.shape}")
        return logits_output
    
    def cleanup_caching(self) -> None:
        """
        Clean up KV cache after decoding
        """
        self.kv_cache = {}
    
    def rearrange_kv_cache(self, source_indices: List[int]) -> None:
        """
        Rearrange KV cache according to beam search selection
        
        Args:
            source_indices: Indices of beams to keep
        """
        # Only rearrange if order changed
        if source_indices != list(range(len(source_indices))):
            # Rearrange cached tensors according to source_indices
            for module in self.kv_modules:
                if module in self.kv_cache:
                    cached_tensor = self.kv_cache[module]
                    # Gather along batch dimension (axis 0)
                    self.kv_cache[module] = tf.gather(cached_tensor, source_indices, axis=0)

class SequenceRanker:
    """
    Base class for ranking candidate sequences
    """
    
    def rank(self, 
             tokens: List[List[tf.Tensor]], 
             sum_logprobs: List[List[float]]) -> List[int]:
        """
        Rank sequences and return indices of best candidates
        
        Args:
            tokens: List of groups of token sequences
            sum_logprobs: Cumulative log probabilities for each sequence
            
        Returns:
            Indices of best sequence in each group
        """
        raise NotImplementedError


class MaximumLikelihoodRanker(SequenceRanker):
    """
    Select sample with highest log probability, with length penalty
    """
    
    def __init__(self, length_penalty: Optional[float]):
        """
        Args:
            length_penalty: Alpha in Google NMT paper, or None for simple length norm
        """
        self.length_penalty = length_penalty
    
    def rank(self, 
             tokens: List[List[tf.Tensor]], 
             sum_logprobs: List[List[float]]) -> List[int]:
        """
        Returns:
            List of indices for best sequence in each group
        """
        def scores(logprobs: List[float], lengths: List[int]) -> List[float]:
            """Calculate length-normalized scores"""
            result = []
            for logprob, length in zip(logprobs, lengths):
                if self.length_penalty is None:
                    # Simple length normalization
                    penalty = length
                else:
                    # Google NMT length penalty: ((5 + length) / 6) ** alpha
                    penalty = ((5 + length) / 6) ** self.length_penalty
                result.append(logprob / penalty)
            return result
        
        # Calculate lengths for all sequences
        lengths = [[len(t) for t in s] for s in tokens]
        
        # Get sequence with highest score for each group
        return [np.argmax(scores(p, l)) for p, l in zip(sum_logprobs, lengths)]

class DecodingTask:
    """
    Main decoding task orchestrator
    """
    
    def __init__(self, model, options: DecodingOptions, tokenizer=None):
        """
        Args:
            model: Whisper model instance
            options: Decoding configuration options
            tokenizer: Optional tokenizer (if None, will create default)
        """
        self.model = model
        
        # Use provided tokenizer or create default
        if tokenizer is None:
            from tokenizer import get_tokenizer
            language = options.language or "en"
            tokenizer = get_tokenizer(
                model.is_multilingual,
                num_languages=model.num_languages,
                language=language,
                task=options.task,
            )
        self.tokenizer = tokenizer
        self.options = self._verify_options(options)

        self.n_group = options.beam_size or options.best_of or 1
        self.n_ctx = model.dims.n_text_ctx
        self.sample_len = options.sample_len or model.dims.n_text_ctx // 2

        self.sot_sequence = tuple(tokenizer.sot_sequence)
        if self.options.without_timestamps:
            self.sot_sequence = tuple(tokenizer.sot_sequence_including_notimestamps)

        self.initial_tokens = self._get_initial_tokens()
        self.sample_begin = len(self.initial_tokens)
        self.sot_index = self.initial_tokens.index(tokenizer.sot)
        
        print(f"[DEBUG] Initial tokens: {self.initial_tokens}")
        print(f"[DEBUG] Tokenizer sot_sequence: {tokenizer.sot_sequence}")

        self.inference = TensorFlowInference(model, len(self.initial_tokens))

        self.sequence_ranker = MaximumLikelihoodRanker(options.length_penalty)

        if options.beam_size is not None:
            self.decoder = BeamSearchDecoder(
                options.beam_size,
                tokenizer.eot,
                self.inference,
                options.patience
            )
        else:
            self.decoder = GreedyDecoder(options.temperature, tokenizer.eot)
        
        # Logit filters
        self.logit_filters = []
        if self.options.suppress_blank:
            self.logit_filters.append(SuppressBlank(self.tokenizer, self.sample_begin))
        self.logit_filters.append(SuppressTokens(self._get_suppress_tokens()))
        
        if not options.without_timestamps:
            vocab_size = model.dims.n_vocab
            
            if vocab_size > 50364:
                precision = 30.0 / model.dims.n_audio_ctx
                max_initial_timestamp_index = None
                if options.max_initial_timestamp:
                    max_initial_timestamp_index = round(options.max_initial_timestamp / precision)
                self.logit_filters.append(
                    ApplyTimestampRules(tokenizer, self.sample_begin, max_initial_timestamp_index)
                )
    
    def _verify_options(self, options: DecodingOptions) -> DecodingOptions:
        """
        Validate decoding options for consistency
        
        Args:
            options: DecodingOptions to validate
            
        Returns:
            Validated DecodingOptions
            
        Raises:
            ValueError: If options are inconsistent
        """
        if options.beam_size is not None and options.best_of is not None:
            raise ValueError("beam_size and best_of can't be given together")
        
        if options.temperature == 0:
            if options.best_of is not None:
                raise ValueError("best_of with greedy sampling (T=0) is not compatible")

        if options.patience is not None and options.beam_size is None:
            raise ValueError("patience requires beam_size to be given")

        if options.length_penalty is not None and not (0 <= options.length_penalty <= 1):
            raise ValueError("length_penalty (alpha) should be a value between 0 and 1")
        
        return options
    
    def _get_initial_tokens(self) -> Tuple[int]:
        """ 
        Get initial token sequence (SOT + prefix + prompt)
        
        Returns:
            Tuple of initial token IDs
        """
        # Start with SOT sequence
        tokens = list(self.sot_sequence)
        
        # Add prefix if provided
        if self.options.prefix:
            prefix = self.options.prefix
            # Encode prefix string or use directly if already tokens
            if isinstance(prefix, str):
                prefix_tokens = self.tokenizer.encode(" " + prefix.strip())
            else:
                prefix_tokens = prefix
            
            # Limit prefix length
            if self.sample_len is not None:
                max_prefix_len = self.n_ctx // 2 - self.sample_len
                prefix_tokens = prefix_tokens[-max_prefix_len:]
            
            tokens = tokens + list(prefix_tokens)
        
        # Add prompt if provided (for context conditioning)
        if self.options.prompt:
            prompt = self.options.prompt
            # Encode prompt string or use directly
            if isinstance(prompt, str):
                prompt_tokens = self.tokenizer.encode(" " + prompt.strip())
            else:
                prompt_tokens = prompt
            
            # Add sot_prev + prompt tokens before main tokens
            tokens = (
                [self.tokenizer.sot_prev]
                + prompt_tokens[-(self.n_ctx // 2 - 1):]
                + tokens
            )
        
        return tuple(tokens)
    
    def _get_suppress_tokens(self) -> Tuple[int]:
        """
        Get list of token IDs to suppress during decoding
        
        Returns:
            Tuple of token IDs to suppress
        """
        suppress_tokens = self.options.suppress_tokens
        
        # Parse suppress_tokens if string
        if isinstance(suppress_tokens, str):
            suppress_tokens = [int(t) for t in suppress_tokens.split(",")]
        
        # Handle special -1 value (means suppress all non-speech tokens)
        if suppress_tokens and -1 in suppress_tokens:
            # Remove -1 and add all non-speech tokens
            suppress_tokens = [t for t in suppress_tokens if t >= 0]
            # Check if tokenizer has non_speech_tokens property
            if hasattr(self.tokenizer, 'non_speech_tokens'):
                suppress_tokens.extend(self.tokenizer.non_speech_tokens)
        elif suppress_tokens is None or len(suppress_tokens) == 0:
            suppress_tokens = []
        else:
            assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"
        
        suppress_tokens.extend([
            50359, 50358, 50258, 50361, 50360  # transcribe, translate, sot, sot_prev, sot_lm
        ])
        
        if hasattr(self.tokenizer, 'no_speech') and self.tokenizer.no_speech is not None:
            suppress_tokens.append(self.tokenizer.no_speech)
        else:
            suppress_tokens.append(50362)
        vocab_size = self.model.dims.n_vocab
        
        if self.options.without_timestamps or vocab_size <= 50364:
            suppress_tokens.extend(range(50363, vocab_size))
        
        return tuple(sorted(set(suppress_tokens)))
    
    def _get_audio_features(self, mel: tf.Tensor) -> tf.Tensor:
        """
        Get audio features (encode if needed, or use pre-encoded)
        
        Args:
            mel: Mel spectrogram tensor or pre-encoded features
            
        Returns:
            Audio features tensor [batch, n_audio_ctx, n_audio_state]
        """
        # Convert to FP16 if requested
        if self.options.fp16:
            mel = tf.cast(mel, tf.float16)
        
        # Check if already encoded (skip encoder if features match expected shape)
        if mel.shape[-2:] == (self.model.dims.n_audio_ctx, self.model.dims.n_audio_state):
            audio_features = mel
        else:
            # Need to encode
            audio_features = self.model.encoder(mel, training=False)
        
        # Verify dtype
        expected_dtype = tf.float16 if self.options.fp16 else tf.float32
        if audio_features.dtype != expected_dtype:
            raise TypeError(f"audio_features has incorrect dtype: {audio_features.dtype}")
        
        return audio_features
    
    def _main_loop(self, audio_features: tf.Tensor, tokens: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, List[float]]:
        """
        Core autoregressive generation loop
        
        Args:
            audio_features: [n_batch, n_audio_ctx, n_audio_state] - encoder output
            tokens: [n_batch, initial_seq_len] - initial tokens
            
        Returns:
            tokens: [n_batch, final_seq_len] - generated tokens
            sum_logprobs: [n_batch] - cumulative log probabilities
            no_speech_probs: List of no-speech probabilities for each batch
        """
        # Initialize
        n_batch = tf.shape(tokens)[0].numpy()
        sum_logprobs = tf.zeros((n_batch,), dtype=tf.float32)
        no_speech_probs = [np.nan] * n_batch
        
        try:
            # Autoregressive generation loop
            for i in range(self.sample_len):
                # Get logits from decoder
                logits = self.inference.logits(tokens, audio_features)
                
                # On first iteration, save no_speech probability
                if i == 0:
                    # Get no_speech token ID (50362)
                    no_speech_token = getattr(self.tokenizer, 'no_speech', 50362)
                    if no_speech_token is not None:
                        # Get probabilities at SOT position
                        probs_at_sot = tf.nn.softmax(
                            tf.cast(logits[:, self.sot_index], tf.float32),
                            axis=-1
                        )
                        
                        # Extract no_speech probability
                        no_speech_probs = probs_at_sot[:, no_speech_token].numpy().tolist()
                
                # Consider only logits at last token position
                logits = logits[:, -1]
                
                if i == 0:
                    top_vals, top_idx = tf.nn.top_k(logits[0], k=10)
                    print(f"[DEBUG] Step 0 BEFORE filter: top10 tokens={top_idx.numpy()}, values={top_vals.numpy()}")
                
                # Apply logit filters
                for logit_filter in self.logit_filters:
                    logits = logit_filter.apply(logits, tokens)
                
                if i == 0:
                    top_vals, top_idx = tf.nn.top_k(logits[0], k=10)
                    print(f"[DEBUG] Step 0 AFTER filter: top10 tokens={top_idx.numpy()}, values={top_vals.numpy()}")
                
                # Select next tokens using decoder (greedy or beam search)
                tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)
                
                if i < 3:
                    print(f"[DEBUG main_loop] Step {i}: tokens[0]={tokens[0].numpy()}, completed={completed}")
                
                # Check termination conditions
                if completed or tf.shape(tokens)[-1] > self.n_ctx:
                    break
        
        finally:
            # Always cleanup caching, even if error occurs
            self.inference.cleanup_caching()
        
        return tokens, sum_logprobs, no_speech_probs
    
    def run(self, mel: tf.Tensor) -> List[DecodingResult]:
        """
        Main entry point for decoding
        
        Args:
            mel: [n_audio, n_mels, n_frames] - Mel spectrogram(s)
            
        Returns:
            List of DecodingResult for each audio
        """
        # Reset decoder state
        self.decoder.reset()
        
        tokenizer = self.tokenizer
        n_audio = mel.shape[0]
        
        audio_features = self._get_audio_features(mel)
        
        # Initialize tokens with initial_tokens for each audio
        tokens = tf.tile(
            tf.constant([self.initial_tokens], dtype=tf.int32),
            [n_audio, 1]
        )
        
        # Language detection (simplified - assume language is set in options)
        # For now, use provided language or default to "en"
        languages = [self.options.language or "en"] * n_audio
        language_probs = [None] * n_audio
        tokens = tf.repeat(tokens, repeats=self.n_group, axis=0)
        
        # Repeat audio features too
        audio_features = tf.repeat(audio_features, repeats=self.n_group, axis=0)
        
        # Call main sampling loop
        tokens, sum_logprobs, no_speech_probs = self._main_loop(audio_features, tokens)
        
        # Reshape to (n_audio, n_group, ...)
        audio_features = audio_features[::self.n_group]
        no_speech_probs = no_speech_probs[::self.n_group]
        
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio
        tokens = tf.reshape(tokens, [n_audio, self.n_group, -1])
        sum_logprobs = tf.reshape(sum_logprobs, [n_audio, self.n_group])
        
        # Finalize and get candidates
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        
        # Slice tokens between sample_begin and EOT
        processed_tokens = []
        for s in tokens:
            group_tokens = []
            for t in s:
                # Find first EOT position
                eot_positions = tf.where(t == tokenizer.eot)
                if len(eot_positions) > 0:
                    eot_idx = int(eot_positions[0][0])
                    group_tokens.append(t[self.sample_begin:eot_idx])
                else:
                    group_tokens.append(t[self.sample_begin:])
            processed_tokens.append(group_tokens)
        tokens = processed_tokens
        
        # Select top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens_list = [t[i].numpy().tolist() for i, t in zip(selected, tokens)]
        
        # Decode to text
        texts = [tokenizer.decode(t).strip() for t in tokens_list]
        
        # Get selected log probabilities
        sum_logprobs_list = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        
        # Calculate average log probabilities
        avg_logprobs = [lp / (len(t) + 1) for t, lp in zip(tokens_list, sum_logprobs_list)]
        
        # Verify all fields have same length
        fields = (texts, languages, tokens_list, avg_logprobs, no_speech_probs)
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")
        
        # Create DecodingResult objects
        results = []
        for i in range(n_audio):
            results.append(DecodingResult(
                audio_features=audio_features[i],
                language=languages[i],
                language_probs=language_probs[i],
                tokens=tokens_list[i],
                text=texts[i],
                avg_logprob=avg_logprobs[i],
                no_speech_prob=no_speech_probs[i],
                temperature=self.options.temperature,
                compression_ratio=compression_ratio(texts[i])
            ))
        
        return results

def compression_ratio(text: str) -> float:
    """
    Calculate text compression ratio (used in DecodingResult)
    Higher ratio suggests more repetitive text
    """
    import zlib
    text_bytes = text.encode('utf-8')
    return len(text_bytes) / len(zlib.compress(text_bytes))