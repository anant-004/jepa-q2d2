"""Dataset and collation for KoeTTS AR model training.

Two dataset modes:
1. Pre-encoded: audio already converted to packed FSQ tokens (.npy files)
   Used for main training. Fast — no encoder forward pass per step.

2. On-the-fly: raw audio files encoded during training (for debugging/eval)
   Slower but doesn't require pre-encoding.

Input format (JSONL manifest):
    {"text": "Hello world", "audio_tokens": "/path/to/tokens.npy"}
    {"text": "Good morning", "audio_tokens": "/path/to/tokens.npy", "prompt_tokens": "/path/to/prompt.npy"}

Sequence layout:
    [bos] [text...] [text_sep] [prompt...] [audio_sep] [target...] [eos]

    segment_ids:   0=text  1=prompt_audio  2=target_audio
    group_pos_ids: 0 for text/special, cycles 0-18 for audio frames
    loss_mask:     1 for target_audio + eos, 0 elsewhere
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from koe.config import TokenConfig, QwenTokenConfig


class TTSDataset(Dataset):
    """TTS dataset from pre-encoded audio tokens.

    Each sample is a JSONL line with:
        - "text": the text transcript
        - "audio_tokens": path to .npy file with packed FSQ tokens [T_z, 19]
        - "prompt_tokens": (optional) path to .npy for voice prompt [T_p, 19]
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        text_tokenizer,
        token_config: Union[TokenConfig, QwenTokenConfig],
        max_seq_len: int = 2048,
        groups_per_frame: int = 19,
        cfg_dropout: float = 0.0,
    ):
        self.text_tokenizer = text_tokenizer
        self.token_config = token_config
        self.max_seq_len = max_seq_len
        self.groups_per_frame = groups_per_frame
        self.cfg_dropout = cfg_dropout

        # Load manifest
        self.samples = []
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Encode text
        text_ids = self.text_tokenizer.encode(sample["text"])

        # CFG dropout: with probability cfg_dropout, drop the text
        # This trains the model to generate audio without text conditioning
        # At inference, we can use classifier-free guidance
        drop_text = False
        if self.cfg_dropout > 0:
            import random
            if random.random() < self.cfg_dropout:
                drop_text = True
                text_ids = []

        # Load pre-encoded audio tokens
        audio_tokens = np.load(sample["audio_tokens"])  # [T_z, 19]
        audio_flat = audio_tokens.reshape(-1)  # [T_z * 19]
        # Offset into vocab
        audio_ids = (audio_flat + self.token_config.audio_offset).tolist()

        # Load prompt tokens if available
        prompt_ids = []
        if "prompt_tokens" in sample and sample["prompt_tokens"]:
            prompt_tokens = np.load(sample["prompt_tokens"])  # [T_p, 19]
            prompt_flat = prompt_tokens.reshape(-1)
            prompt_ids = (prompt_flat + self.token_config.audio_offset).tolist()

        # Build the full sequence
        return self._build_sequence(text_ids, prompt_ids, audio_ids)

    def _build_sequence(
        self,
        text_ids: List[int],
        prompt_ids: List[int],
        target_ids: List[int],
    ) -> Dict[str, torch.Tensor]:
        """Build training sequence with all metadata tensors.

        Layout with prompt:
            [bos] [text...] [text_sep] [prompt...] [audio_sep] [target...] [eos]

        Layout without prompt:
            [bos] [text...] [text_sep] [audio_sep] [target...] [eos]
        """
        tc = self.token_config
        gpf = self.groups_per_frame

        # Resolve special token IDs based on config type
        if isinstance(tc, QwenTokenConfig):
            bos_id = 151643  # Qwen3 <|im_start|>
            eos_id = 151645  # Qwen3 <|endoftext|>
        else:
            bos_id = tc.bos_id
            eos_id = tc.eos_id

        # Build token sequence
        tokens = [bos_id]
        segments = [0]
        group_pos = [0]

        # Text tokens (segment 0, group_pos 0)
        for tid in text_ids:
            tokens.append(tid)
            segments.append(0)
            group_pos.append(0)

        # text_sep
        tokens.append(tc.text_sep_id)
        segments.append(0)
        group_pos.append(0)

        # Prompt audio (segment 1, group_pos cycles 0-18)
        if prompt_ids:
            for i, pid in enumerate(prompt_ids):
                tokens.append(pid)
                segments.append(1)
                group_pos.append(i % gpf)

        # audio_sep
        tokens.append(tc.audio_sep_id)
        segments.append(1 if prompt_ids else 0)
        group_pos.append(0)

        # Mark where target audio starts (for loss_mask)
        target_start = len(tokens)

        # Target audio (segment 2, group_pos cycles 0-18)
        for i, tid in enumerate(target_ids):
            tokens.append(tid)
            segments.append(2)
            group_pos.append(i % gpf)

        # eos
        tokens.append(eos_id)
        segments.append(2)
        group_pos.append(0)

        # Truncate to max_seq_len
        tokens = tokens[:self.max_seq_len]
        segments = segments[:self.max_seq_len]
        group_pos = group_pos[:self.max_seq_len]

        # Build loss_mask: 1 for target audio + eos positions
        loss_mask = [0.0] * len(tokens)
        for i in range(min(target_start, len(tokens)), len(tokens)):
            loss_mask[i] = 1.0

        return {
            "token_ids": torch.tensor(tokens, dtype=torch.long),
            "segment_ids": torch.tensor(segments, dtype=torch.long),
            "group_pos_ids": torch.tensor(group_pos, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.float32),
        }


def collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """Collate variable-length sequences with padding.

    Pads all sequences to the max length in the batch.
    Padding tokens get segment_id=0, group_pos=0, loss_mask=0.

    Returns:
        dict with tensors [B, T_max] for token_ids, segment_ids,
        group_pos_ids, loss_mask, and attention_mask.
    """
    max_len = max(b["token_ids"].shape[0] for b in batch)

    token_ids = []
    segment_ids = []
    group_pos_ids = []
    loss_mask = []
    attention_mask = []

    for b in batch:
        T = b["token_ids"].shape[0]
        pad_len = max_len - T

        token_ids.append(F.pad(b["token_ids"], (0, pad_len), value=pad_id))
        segment_ids.append(F.pad(b["segment_ids"], (0, pad_len), value=0))
        group_pos_ids.append(F.pad(b["group_pos_ids"], (0, pad_len), value=0))
        loss_mask.append(F.pad(b["loss_mask"], (0, pad_len), value=0.0))
        # attention_mask: 1 for real tokens, 0 for padding
        attn = torch.ones(T, dtype=torch.float32)
        attention_mask.append(F.pad(attn, (0, pad_len), value=0.0))

    return {
        "token_ids": torch.stack(token_ids),
        "segment_ids": torch.stack(segment_ids),
        "group_pos_ids": torch.stack(group_pos_ids),
        "loss_mask": torch.stack(loss_mask),
        "attention_mask": torch.stack(attention_mask),
    }
