"""Transformer-based audio codec decoder (MOSS/CAT-inspired).

Replaces HiFi-GAN with a pure Transformer decoder that uses
patchify/unpatchify for temporal resampling — no ConvTranspose1d,
no checkerboard artifacts.

Architecture:
    Input: [B, code_dim, T@25Hz] quantized features

    Stage 1 (25Hz):  12 Transformer blocks, hidden=512
    Unpatch 2x → 50Hz, hidden=384

    Stage 2 (50Hz):  8 Transformer blocks, hidden=384
    Unpatch 2x → 100Hz, hidden=256

    Stage 3 (100Hz): 4 Transformer blocks, hidden=256
    Final projection → 240 samples per frame → 24kHz waveform

    Total upsample: 2 × 2 × 240 = 960x
    Decoder params: ~50M
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerDecoderConfig:
    code_dim: int = 32
    sample_rate: int = 24000
    hop_length: int = 960  # 25 Hz

    # Stage configs: (n_blocks, hidden_dim, n_heads, ffn_mult)
    stages: List[Tuple[int, int, int, int]] = field(default_factory=lambda: [
        (12, 512, 8, 4),   # Stage 1: 25 Hz
        (8, 384, 6, 4),    # Stage 2: 50 Hz
        (4, 256, 4, 4),    # Stage 3: 100 Hz
    ])
    unpatch_sizes: List[int] = field(default_factory=lambda: [2, 2])
    # Final output: 960 / (2*2) = 240 samples per frame at 100 Hz
    final_patch_size: int = 240

    dropout: float = 0.0
    rope_base: float = 10000.0


# ═══════════════════════════════════════════════════════════
# Rotary Positional Embedding
# ═══════════════════════════════════════════════════════════

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.outer(t, self.inv_freq.to(dtype))
        return torch.cos(freqs), torch.sin(freqs)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x: [B, H, T, D_head]."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos = cos[:x.shape[2], :d].unsqueeze(0).unsqueeze(0)  # [1, 1, T, d]
    sin = sin[:x.shape[2], :d].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ═══════════════════════════════════════════════════════════
# Transformer Block
# ═══════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with RoPE and optional sliding window."""

    def __init__(self, hidden_dim: int, n_heads: int, ffn_mult: int = 4,
                 dropout: float = 0.0, rope_base: float = 10000.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        assert hidden_dim % n_heads == 0

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.norm2 = nn.LayerNorm(hidden_dim)
        ffn_dim = hidden_dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim, bias=False),
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]"""
        B, T, D = x.shape

        # Self-attention with RoPE
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [B, T, H, D_head]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [B, H, T, D_head]

        cos, sin = self.rope(T, device=x.device, dtype=x.dtype)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # Scaled dot-product attention (uses Flash Attention when available)
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)  # [B, T, D]
        x = x + self.dropout(self.out_proj(attn_out))

        # FFN
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ═══════════════════════════════════════════════════════════
# Unpatchify (Temporal Upsampling)
# ═══════════════════════════════════════════════════════════

class Unpatchify(nn.Module):
    """Temporal upsampling via linear projection + reshape.

    [B, T, D_in] → Linear(D_in, patch_size * D_out) → [B, T*patch_size, D_out]

    No ConvTranspose — no checkerboard artifacts.
    """

    def __init__(self, in_dim: int, out_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.out_dim = out_dim
        self.proj = nn.Linear(in_dim, patch_size * out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D_in] → [B, T*patch_size, D_out]"""
        B, T, _ = x.shape
        x = self.proj(x)                          # [B, T, patch_size * D_out]
        x = x.view(B, T * self.patch_size, self.out_dim)  # [B, T*ps, D_out]
        x = self.norm(x)
        return x


# ═══════════════════════════════════════════════════════════
# Transformer Codec Decoder
# ═══════════════════════════════════════════════════════════

class TransformerCodecDecoder(nn.Module):
    """MOSS-inspired Transformer decoder for audio codec.

    Pure Transformer with unpatchify for temporal upsampling.
    No CNNs, no ConvTranspose, no checkerboard artifacts.
    """

    def __init__(self, cfg: TransformerDecoderConfig = None):
        super().__init__()
        if cfg is None:
            cfg = TransformerDecoderConfig()
        self.cfg = cfg

        stages = cfg.stages
        n_stages = len(stages)

        # Input projection: code_dim → first stage hidden
        first_hidden = stages[0][1]
        self.input_proj = nn.Linear(cfg.code_dim, first_hidden)
        self.input_norm = nn.LayerNorm(first_hidden)

        # Build stages
        self.stages = nn.ModuleList()
        for n_blocks, hidden, n_heads, ffn_mult in stages:
            blocks = nn.ModuleList([
                TransformerBlock(hidden, n_heads, ffn_mult,
                                 dropout=cfg.dropout, rope_base=cfg.rope_base)
                for _ in range(n_blocks)
            ])
            self.stages.append(blocks)

        # Unpatchify layers between stages
        self.unpatch_layers = nn.ModuleList()
        for i, ps in enumerate(cfg.unpatch_sizes):
            in_dim = stages[i][1]
            out_dim = stages[i + 1][1]
            self.unpatch_layers.append(Unpatchify(in_dim, out_dim, ps))

        # Final projection: last hidden → waveform samples
        last_hidden = stages[-1][1]
        self.final_proj = nn.Linear(last_hidden, cfg.final_patch_size)

        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"[TransformerDecoder] {n_params:.1f}M params, "
              f"stages={[(s[0], s[1]) for s in stages]}, "
              f"unpatch={cfg.unpatch_sizes}, final_patch={cfg.final_patch_size}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z_q: torch.Tensor, target_len: Optional[int] = None) -> torch.Tensor:
        """Decode quantized features to waveform.

        Args:
            z_q: [B, code_dim, T] quantized features (channel-first from Q2D2)
            target_len: desired output waveform length for trimming

        Returns:
            wav: [B, 1, T_wav] reconstructed waveform
        """
        # Channel-first → sequence-first: [B, D, T] → [B, T, D]
        x = z_q.transpose(1, 2)

        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)

        # Process through stages with unpatchify between them
        for i, blocks in enumerate(self.stages):
            for block in blocks:
                x = block(x)

            # Unpatchify after each stage except the last
            if i < len(self.unpatch_layers):
                x = self.unpatch_layers[i](x)

        # Final projection → waveform samples
        # x: [B, T_final, hidden] → [B, T_final, final_patch_size]
        wav = self.final_proj(x)

        # Reshape to waveform: [B, T_final * final_patch_size]
        B = wav.shape[0]
        wav = wav.reshape(B, -1)

        # Tanh to bound output
        wav = torch.tanh(wav)

        # Trim or pad to target length
        if target_len is not None:
            if wav.shape[-1] > target_len:
                wav = wav[:, :target_len]
            elif wav.shape[-1] < target_len:
                wav = F.pad(wav, (0, target_len - wav.shape[-1]))

        # [B, T_wav] → [B, 1, T_wav]
        return wav.unsqueeze(1)
