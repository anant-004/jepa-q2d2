"""Text tokenizers for KoeTTS.

Two tokenizers for two experiments:
  - CharTokenizer (Experiment A): character → integer ID, no dependencies
  - QwenTextTokenizer (Experiment B): thin wrapper around Qwen3's BPE tokenizer

Both expose the same interface:
    tokenizer.encode(text) → list[int]
    tokenizer.decode(ids) → str
    tokenizer.vocab_size → int
"""

from typing import List, Optional

from koe.config import TokenConfig, QwenTokenConfig


# ──────────────────────────────────────────────────────────────
# Accented / extended characters beyond ASCII printable (32-126)
# Slots 100-199 in the character range (IDs 105-204 in the vocab)
# ──────────────────────────────────────────────────────────────
_EXTENDED_CHARS = (
    "àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ"
    "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞŸ"
    "ßœŒšŠžŽ"
    "¡¿"
    # Common CJK/emoji are NOT included — this is for Latin-script TTS
)

# Build bidirectional lookup for extended chars
# Extended char → slot index (100, 101, 102, ...)
_EXTENDED_TO_SLOT = {ch: 100 + i for i, ch in enumerate(_EXTENDED_CHARS)}
# Slot index → extended char
_SLOT_TO_EXTENDED = {v: k for k, v in _EXTENDED_TO_SLOT.items()}

# Replacement character slot (for unknown characters)
_UNK_SLOT = 99  # slot 99, ID = 99 + 5 = 104


class CharTokenizer:
    """Character-level tokenizer for Experiment A (custom 97M model).

    Maps each character to an integer ID within the TokenConfig vocabulary:
        - ASCII printable (space=32 through ~=126): slots 0-94 → IDs 5-99
        - Extended/accented characters: slots 100-199 → IDs 105-204
        - Unknown characters: slot 99 → ID 104 (replacement)

    The char_id = slot + num_special (5), so:
        encode('H') → ord('H') - 32 + 5 = 72 - 32 + 5 = 45
        ... wait, let me be precise:
        slot = ord('H') - 32 = 40
        id = slot + 5 = 45

    Example:
        tok = CharTokenizer()
        tok.encode("Hi!")  → [45, 78, 0+5=6... no let me just show the logic]

    Actually, let's trace through concretely:
        'H' → ord('H')=72, slot=72-32=40, id=40+5=45
        'i' → ord('i')=105, slot=105-32=73, id=73+5=78
        '!' → ord('!')=33, slot=33-32=1, id=1+5=6
        encode("Hi!") → [45, 78, 6]
    """

    def __init__(self, config: Optional[TokenConfig] = None):
        self.config = config or TokenConfig()

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    def _char_to_id(self, ch: str) -> int:
        """Single character → token ID."""
        code = ord(ch)

        # ASCII printable: space (32) through ~ (126) → slots 0-94
        if 32 <= code <= 126:
            slot = code - 32  # 0 to 94
            return slot + self.config.num_special

        # Extended / accented characters → slots 100+
        if ch in _EXTENDED_TO_SLOT:
            slot = _EXTENDED_TO_SLOT[ch]
            return slot + self.config.num_special

        # Unknown → replacement slot
        return _UNK_SLOT + self.config.num_special

    def _id_to_char(self, token_id: int) -> str:
        """Token ID → single character."""
        slot = token_id - self.config.num_special

        # Out of character range
        if slot < 0 or slot >= self.config.num_chars:
            return "\ufffd"  # Unicode replacement character

        # ASCII printable: slots 0-94
        if slot <= 94:
            return chr(slot + 32)

        # Replacement slot
        if slot == _UNK_SLOT:
            return "\ufffd"

        # Extended characters: slots 100+
        if slot in _SLOT_TO_EXTENDED:
            return _SLOT_TO_EXTENDED[slot]

        # Unused slot (95-98)
        return "\ufffd"

    def encode(self, text: str) -> List[int]:
        """Encode text string to list of character token IDs.

        Does NOT add special tokens (bos, eos, text_sep) — that's
        the dataset's job when building the full training sequence.
        """
        return [self._char_to_id(ch) for ch in text]

    def decode(self, ids: List[int]) -> str:
        """Decode list of character token IDs back to string."""
        return "".join(self._id_to_char(i) for i in ids)


class QwenTextTokenizer:
    """Thin wrapper around Qwen3's BPE tokenizer for Experiment B.

    Loads the pretrained Qwen3-0.6B-Base tokenizer from HuggingFace
    and delegates encode/decode to it. The wrapper exists so both
    experiments share the same interface.

    The Qwen3 tokenizer handles subword BPE — "Hello" might become
    one token or multiple, depending on the learned merges.

    Audio tokens and new special tokens (text_sep, audio_sep) are
    NOT handled here — they're integer IDs that the dataset adds
    directly. This tokenizer only handles the TEXT portion.
    """

    def __init__(self, config: Optional[QwenTokenConfig] = None):
        self.config = config or QwenTokenConfig()

        # Lazy import — only needed for Experiment B
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B-Base",
            trust_remote_code=True,
        )

    @property
    def vocab_size(self) -> int:
        """Full extended vocab size (Qwen + audio + special)."""
        return self.config.vocab_size

    @property
    def qwen_vocab_size(self) -> int:
        """Original Qwen3 vocab size (before extension)."""
        return self.config.qwen_vocab_size

    def encode(self, text: str) -> List[int]:
        """Encode text string to list of BPE token IDs.

        Returns Qwen3 token IDs (0-151935). Does NOT add special
        tokens — that's the dataset's job.
        """
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: List[int]) -> str:
        """Decode list of BPE token IDs back to string.

        Only decodes IDs in the Qwen3 vocab range. Audio token IDs
        and new special token IDs are skipped.
        """
        # Filter to only Qwen3 vocab IDs for decoding
        text_ids = [i for i in ids if i < self.config.qwen_vocab_size]
        return self._tokenizer.decode(text_ids, skip_special_tokens=False)
