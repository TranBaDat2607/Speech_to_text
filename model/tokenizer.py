import base64
import os
import string
from dataclasses import dataclass, field
from functools import lru_cache, cached_property
from typing import Dict, List, Optional, Tuple
import tensorflow as tf
import numpy as np
import re

LANGUAGES = {
    "en": "english", "zh": "chinese", "de": "german", "es": "spanish",
    "ru": "russian", "ko": "korean", "fr": "french", "ja": "japanese",
    "pt": "portuguese", "tr": "turkish", "pl": "polish", "ca": "catalan",
    "nl": "dutch", "ar": "arabic", "sv": "swedish", "it": "italian",
    "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese",
    "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay",
    "cs": "czech", "ro": "romanian", "da": "danish", "hu": "hungarian",
    "ta": "tamil", "no": "norwegian", "th": "thai", "ur": "urdu",
    "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian", "la": "latin",
    "mi": "maori", "ml": "malayalam", "cy": "welsh", "sk": "slovak",
    "te": "telugu", "fa": "persian", "lv": "latvian", "bn": "bengali",
    "sr": "serbian", "az": "azerbaijani", "sl": "slovenian", "kn": "kannada",
    "et": "estonian", "mk": "macedonian", "br": "breton", "eu": "basque",
    "is": "icelandic", "hy": "armenian", "ne": "nepali", "mn": "mongolian",
    "bs": "bosnian", "kk": "kazakh", "sq": "albanian", "sw": "swahili",
    "gl": "galician", "mr": "marathi", "pa": "punjabi", "si": "sinhala",
    "km": "khmer", "sn": "shona", "yo": "yoruba", "so": "somali",
    "af": "afrikaans", "oc": "occitan", "ka": "georgian", "be": "belarusian",
    "tg": "tajik", "sd": "sindhi", "gu": "gujarati", "am": "amharic",
    "yi": "yiddish", "lo": "lao", "uz": "uzbek", "fo": "faroese",
    "ht": "haitian creole", "ps": "pashto", "tk": "turkmen", "nn": "nynorsk",
    "mt": "maltese", "sa": "sanskrit", "lb": "luxembourgish", "my": "myanmar",
    "bo": "tibetan", "tl": "tagalog", "mg": "malagasy", "as": "assamese",
    "tt": "tatar", "haw": "hawaiian", "ln": "lingala", "ha": "hausa",
    "ba": "bashkir", "jw": "javanese", "su": "sundanese", "yue": "cantonese",
}

TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my", "valencian": "ca", "flemish": "nl", "haitian": "ht",
    "letzeburgesch": "lb", "pushto": "ps", "panjabi": "pa", "moldavian": "ro",
    "moldovan": "ro", "sinhalese": "si", "castilian": "es", "mandarin": "zh",
}

@dataclass
class Tokenizer:
    """A TensorFlow wrapper around BPE encoding providing quick access to special tokens"""
    
    encoding: 'TensorFlowEncoding'
    num_languages: int
    language: Optional[str] = None
    task: Optional[str] = None
    sot_sequence: Tuple[int] = field(default_factory=tuple)
    special_tokens: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        # Extract special tokens from encoding
        for special in self.encoding.special_tokens_set:
            special_token = self.encoding.encode_single_token(special)
            self.special_tokens[special] = special_token
            
        sot: int = self.special_tokens["<|startoftranscript|>"]
        translate: int = self.special_tokens["<|translate|>"]
        transcribe: int = self.special_tokens["<|transcribe|>"]
        
        langs = tuple(LANGUAGES.keys())[: self.num_languages]
        sot_sequence = [sot]
        if self.language is not None:
            sot_sequence.append(sot + 1 + langs.index(self.language))
        if self.task is not None:
            task_token: int = transcribe if self.task == "transcribe" else translate
            sot_sequence.append(task_token)
            
        self.sot_sequence = tuple(sot_sequence)
    
    def encode(self, text, **kwargs):
        return self.encoding.encode(text, **kwargs)
        
    def decode(self, token_ids: List[int], **kwargs) -> str:
        token_ids = [t for t in token_ids if t < self.timestamp_begin]
        return self.encoding.decode(token_ids, **kwargs)
        
    def decode_with_timestamps(self, token_ids: List[int], **kwargs) -> str:
        """
        Timestamp tokens are above other special tokens' id range and are ignored by `decode()`.
        This method decodes given tokens with timestamps tokens annotated, e.g. "<|1.08|>".
        """
        return self.encoding.decode(token_ids, **kwargs)
    
    
    @cached_property
    def eot(self) -> int:
        return self.encoding.eot_token
    
    @cached_property
    def transcribe(self) -> int:
        """Transcribe task token"""
        return self.special_tokens["<|transcribe|>"]
    
    @cached_property
    def translate(self) -> int:
        """Translate task token"""
        return self.special_tokens["<|translate|>"]
    
    @cached_property
    def sot(self) -> int:
        """Start of transcript token"""
        return self.special_tokens["<|startoftranscript|>"]
    
    @cached_property
    def sot_lm(self) -> int:
        """Start of language model token"""
        return self.special_tokens["<|startoflm|>"]
    
    @cached_property
    def sot_prev(self) -> int:
        """Start of previous token"""
        return self.special_tokens["<|startofprev|>"]
    
    @cached_property
    def no_speech(self) -> int:
        """No speech token"""
        return self.special_tokens["<|nospeech|>"]
    
    @cached_property
    def no_timestamps(self) -> int:
        """No timestamps token"""
        return self.special_tokens["<|notimestamps|>"]
    
    @cached_property
    def timestamp_begin(self) -> int:
        """First timestamp token"""
        return self.special_tokens["<|0.00|>"]
    
    @cached_property
    def language_token(self) -> int:
        """Language token for current language"""
        if self.language is None:
            raise ValueError("This tokenizer does not have language token configured")
        return self.to_language_token(self.language)
    
    def to_language_token(self, language):
        if token := self.special_tokens.get(f"<|{language}|>", None):
            return token
        
        raise KeyError(f"Language {language} not found in tokenizer.")
    
    @cached_property
    def all_language_tokens(self) -> Tuple[int]:
        """All language tokens"""
        result = []
        for token, token_id in self.special_tokens.items():
            if token.strip("<|>") in LANGUAGES:
                result.append(token_id)
        return tuple(result)[:self.num_languages]
    
    @cached_property
    def all_language_codes(self) -> Tuple[str]:
        """All language codes"""
        return tuple(token.strip("<|>") for token in self.special_tokens.keys() 
                    if token.strip("<|>") in LANGUAGES)[:self.num_languages]
    
    @cached_property
    def sot_sequence_including_notimestamps(self) -> Tuple[int]:
        """SOT sequence including no timestamps token"""
        return tuple(list(self.sot_sequence) + [self.no_timestamps])
    
    @cached_property
    def non_speech_tokens(self) -> Tuple[int]:
        """
        Returns the list of tokens to suppress in order to avoid any speaker tags or non-speech
        annotations, to prevent sampling texts that are not actually spoken in the audio, e.g.
        
        - ♪♪♪
        - ( SPEAKING FOREIGN LANGUAGE )
        - [DAVID] Hey there,
        
        keeping basic punctuations like commas, periods, question marks, exclamation points, etc.
        """
        symbols = list('"#()*+/:;<=>@[\\]^_`{|}~「」『』')
        symbols += (
            "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split()
        )
        
        # symbols that may be a single token or multiple tokens depending on the tokenizer.
        # In case they're multiple tokens, suppress the first token, which is safe because:
        # These are between U+2640 and U+267F miscellaneous symbols that are okay to suppress
        # in generations, and in the 3-byte UTF-8 representation they share the first two bytes.
        miscellaneous = set("♩♪♫♬♭♮♯")
        assert all(0x2640 <= ord(c) <= 0x267F for c in miscellaneous)
        
        # allow hyphens "-" and single quotes "'" between words, but not at the beginning of a word
        result = {self.encoding.encode(" -")[0], self.encoding.encode(" '")[0]}
        for symbol in symbols + list(miscellaneous):
            for tokens in [
                self.encoding.encode(symbol),
                self.encoding.encode(" " + symbol),
            ]:
                if len(tokens) == 1 or symbol in miscellaneous:
                    result.add(tokens[0])
                    
        return tuple(sorted(result))
    
    def split_to_word_tokens(self, tokens: List[int]):
        if self.language in {"zh", "ja", "th", "lo", "my", "yue"}:
            # These languages don't typically use spaces, so it is difficult to split words
            # without morpheme analysis. Here, we instead split words at any
            # position where the tokens are decoded as valid unicode points
            return self.split_tokens_on_unicode(tokens)
            
        return self.split_tokens_on_spaces(tokens)
        
    def split_tokens_on_unicode(self, tokens: List[int]):
        decoded_full = self.decode_with_timestamps(tokens)
        replacement_char = "\ufffd"
        
        words = []
        word_tokens = []
        current_tokens = []
        unicode_offset = 0
        
        for token in tokens:
            current_tokens.append(token)
            decoded = self.decode_with_timestamps(current_tokens)
            
            if (
                replacement_char not in decoded
                or decoded_full[unicode_offset + decoded.index(replacement_char)]
                == replacement_char
            ):
                words.append(decoded)
                word_tokens.append(current_tokens)
                current_tokens = []
                unicode_offset += len(decoded)
                
        return words, word_tokens
        
    def split_tokens_on_spaces(self, tokens: List[int]):
        subwords, subword_tokens_list = self.split_tokens_on_unicode(tokens)
        words = []
        word_tokens = []
        
        for subword, subword_tokens in zip(subwords, subword_tokens_list):
            special = subword_tokens[0] >= self.eot
            with_space = subword.startswith(" ")
            punctuation = subword.strip() in string.punctuation
            if special or with_space or punctuation or len(words) == 0:
                words.append(subword)
                word_tokens.append(subword_tokens)
            else:
                words[-1] = words[-1] + subword
                word_tokens[-1].extend(subword_tokens)
                
        return words, word_tokens


class TensorFlowEncoding:
    """TensorFlow implementation of tiktoken.Encoding"""
    
    def __init__(self, name: str, explicit_n_vocab: int, pat_str: str, mergeable_ranks: dict, special_tokens: dict):
        self.name = name
        self.explicit_n_vocab = explicit_n_vocab
        self.pat_str = pat_str
        self.mergeable_ranks = mergeable_ranks
        self.special_tokens = special_tokens
        self.special_tokens_set = set(special_tokens.keys())
        self.eot_token = special_tokens.get("<|endoftext|>", 50256)
        
        # Create reverse mapping
        self.id_to_token = {v: k for k, v in mergeable_ranks.items()}
        self.id_to_token.update({v: k for k, v in special_tokens.items()})
        
        # Compile regex pattern
        self.regex_pattern = re.compile(pat_str)
        
    def encode_single_token(self, token: str) -> int:
        """Encode a single special token"""
        if token in self.special_tokens:
            return self.special_tokens[token]
        raise KeyError(f"Token {token} not found")
        
    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode text to token IDs using BPE"""
        # Pre-tokenize using regex pattern
        tokens = self.regex_pattern.findall(text)
        token_ids = []
        
        for token in tokens:
            # Check if it's a special token first
            if token in self.special_tokens:
                token_ids.append(self.special_tokens[token])
                continue
                
            # Apply BPE encoding
            token_bytes = token.encode('utf-8')
            if token_bytes in self.mergeable_ranks:
                token_ids.append(self.mergeable_ranks[token_bytes])
            else:
                # Fall back to byte-level encoding
                for byte in token_bytes:
                    if bytes([byte]) in self.mergeable_ranks:
                        token_ids.append(self.mergeable_ranks[bytes([byte])])
                        
        return token_ids
        
    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode token IDs back to text"""
        text_parts = []
        
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if isinstance(token, bytes):
                    try:
                        text_parts.append(token.decode('utf-8'))
                    except UnicodeDecodeError:
                        text_parts.append(token.decode('utf-8', errors='replace'))
                else:
                    text_parts.append(token)
                    
        return ''.join(text_parts)


@lru_cache(maxsize=None)
def get_encoding(name: str = "gpt2", num_languages: int = 99):
    """TensorFlow implementation of tiktoken get_encoding"""
    # Simulate loading from tiktoken file
    vocab_path = os.path.join(os.path.dirname(__file__), "assets", f"{name}.tiktoken")
    
    # Create simulated BPE ranks (in practice, load from actual file)
    ranks = {}
    
    # Simulate base vocabulary (GPT-2 style)
    for i in range(256):  # Byte-level tokens
        ranks[bytes([i])] = i
        
    # Add some common BPE merges (simplified)
    common_merges = [
        b'ed', b'ing', b'er', b'est', b'ly', b'tion', b'ness',
        b'ment', b'able', b'ful', b'less', b'ive', b'ous'
    ]
    
    current_rank = 256
    for merge in common_merges:
        if current_rank < 50257:  # GPT-2 vocab size
            ranks[merge] = current_rank
            current_rank += 1
            
    # Fill remaining vocabulary slots
    while current_rank < 50257:
        fake_token = f"token_{current_rank}".encode('utf-8')
        ranks[fake_token] = current_rank
        current_rank += 1
        
    n_vocab = len(ranks)
    special_tokens = {}
    
    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in list(LANGUAGES.keys())[:num_languages]],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]
    
    for token in specials:
        special_tokens[token] = n_vocab
        n_vocab += 1
        
    return TensorFlowEncoding(
        name=os.path.basename(vocab_path) if os.path.exists(vocab_path) else name,
        explicit_n_vocab=n_vocab,
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-ZÀ-ÿ\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+""",
        mergeable_ranks=ranks,
        special_tokens=special_tokens,
    )


@lru_cache(maxsize=None)
def get_tokenizer(
    multilingual: bool,
    *,
    num_languages: int = 99,
    language: Optional[str] = None,
    task: Optional[str] = None,  # Literal["transcribe", "translate", None]
) -> Tokenizer:
    if language is not None:
        language = language.lower()
        if language not in LANGUAGES:
            if language in TO_LANGUAGE_CODE:
                language = TO_LANGUAGE_CODE[language]
            else:
                raise ValueError(f"Unsupported language: {language}")
                
    if multilingual:
        encoding_name = "multilingual"
        language = language or "en"
        task = task or "transcribe"
    else:
        encoding_name = "gpt2"
        language = None
        task = None
        
    encoding = get_encoding(name=encoding_name, num_languages=num_languages)
    
    return Tokenizer(
        encoding=encoding, num_languages=num_languages, language=language, task=task
    )