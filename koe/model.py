"""KoeTTS: Custom 97M AR transformer for Experiment A.

Architecture:
    12 layers, 768 dim, 12 heads, SwiGLU FFN (d_ffn=2048)
    RoPE position encoding, segment embeddings (text/prompt/target),
    group position embeddings (19 FSQ groups per frame)
    Weight-tied lm_head, RMSNorm, KV-cache for inference

Input sequence layout:
    [bos, ...text_ids..., text_sep, ...prompt_audio..., audio_sep, ...target_audio..., eos]
    segment:  0=text         1=prompt_audio           2=target_audio
    group_pos: 0 for text tokens, cycles 0-18 for audio tokens

Loss: CE on target_audio + eos positions only (text & prompt are conditioning)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from koe.config import ModelConfig, TokenConfig


# ──────────────────────────────────────────────────────────────
# RMSNorm — simpler LayerNorm without mean subtraction
# ──────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm(x) = x / sqrt(mean(x²) + eps) * gain

    No learnable bias. One gain parameter per dimension.
    Faster than LayerNorm (skips mean subtraction), works just as well.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.gain


# ──────────────────────────────────────────────────────────────
# RoPE — Rotary Position Embedding
# ──────────────────────────────────────────────────────────────

def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute complex exponentials for RoPE.

    Returns:
        freqs_cis: [max_seq_len, dim//2] complex tensor
    """
    # θ_i = 1 / (theta^(2i/dim)) for i = 0, 1, ..., dim//2 - 1
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    # positions 0, 1, 2, ..., max_seq_len-1
    t = torch.arange(max_seq_len)
    # outer product: [max_seq_len, dim//2]
    angles = torch.outer(t, freqs)
    # complex exponentials: e^(i·angle) = cos(angle) + i·sin(angle)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return freqs_cis


def apply_rope(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply rotary position embedding to Q or K.

    Args:
        x: [B, n_heads, T, d_head]
        freqs_cis: [max_seq_len, d_head//2] complex tensor
        position_ids: [B, T] or None (defaults to 0,1,2,...,T-1)

    Returns:
        rotated x, same shape
    """
    B, H, T, D = x.shape

    # Get the right frequency entries for these positions
    if position_ids is not None:
        # position_ids: [B, T] → gather from freqs_cis
        fc = freqs_cis[position_ids]  # [B, T, D//2]
        fc = fc.unsqueeze(1)  # [B, 1, T, D//2] for broadcasting over heads
    else:
        fc = freqs_cis[:T].unsqueeze(0).unsqueeze(0)  # [1, 1, T, D//2]

    # Reshape x as complex: [B, H, T, D] → [B, H, T, D//2] complex
    x_complex = torch.view_as_complex(x.float().reshape(B, H, T, D // 2, 2))
    # Rotate: multiply by e^(i·angle)
    x_rotated = x_complex * fc
    # Back to real: [B, H, T, D//2] complex → [B, H, T, D]
    return torch.view_as_real(x_rotated).reshape(B, H, T, D).to(x.dtype)


# ──────────────────────────────────────────────────────────────
# CausalSelfAttention — multi-head attention with RoPE + KV-cache
# ──────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and optional KV-cache.

    Q, K, V projections → RoPE on Q,K → scaled dot-product attention
    (with causal mask) → output projection.

    During generation, pass kv_cache=(past_k, past_v) to avoid
    recomputing attention over the full sequence each step.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model

        # Fused QKV projection (slightly faster than separate)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: [B, T, D]
            freqs_cis: precomputed RoPE frequencies
            position_ids: [B, T] absolute positions for RoPE
            kv_cache: (past_k, past_v) each [B, H, T_past, d_head]
            attention_mask: [B, 1, T, T_full] or None for default causal

        Returns:
            output: [B, T, D]
            new_kv_cache: (k, v) each [B, H, T_full, d_head]
        """
        B, T, D = x.shape

        # QKV projection: [B, T, D] → [B, T, 3D]
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=-1)

        # Reshape to heads: [B, T, D] → [B, H, T, d_head]
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE to Q and K
        q = apply_rope(q, freqs_cis, position_ids)
        k = apply_rope(k, freqs_cis, position_ids)

        # KV-cache: prepend past keys/values
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)  # [B, H, T_past+T, d_head]
            v = torch.cat([past_v, v], dim=2)
        new_kv_cache = (k, v)

        # Scaled dot-product attention (PyTorch 2.0+ uses FlashAttention automatically)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=(attention_mask is None and kv_cache is None),
        )
        # out: [B, H, T, d_head]

        # Merge heads: [B, H, T, d_head] → [B, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        return self.out_proj(out), new_kv_cache


# ──────────────────────────────────────────────────────────────
# SwiGLU — gated feed-forward network
# ──────────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    """SwiGLU feed-forward: gate * up, then down-project.

    SwiGLU(x) = (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down

    3 matrices: gate (d→d_ffn), up (d→d_ffn), down (d_ffn→d)
    Better than ReLU/GELU at same parameter count.
    """

    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ffn, bias=False)
        self.up = nn.Linear(d_model, d_ffn, bias=False)
        self.down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ──────────────────────────────────────────────────────────────
# TransformerBlock — one layer (pre-norm residual)
# ──────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """One transformer layer with pre-norm residuals.

    x = x + CausalSelfAttention(RMSNorm(x))
    x = x + SwiGLU(RMSNorm(x))
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.d_ffn)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Self-attention with residual
        h, new_kv = self.attn(
            self.attn_norm(x), freqs_cis, position_ids, kv_cache, attention_mask,
        )
        x = x + h
        # FFN with residual
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_kv


# ──────────────────────────────────────────────────────────────
# KoeTTS — the full 97M AR model
# ──────────────────────────────────────────────────────────────

class KoeTTS(nn.Module):
    """Autoregressive speech language model for Experiment A.

    Input: sequence of token IDs (text chars + audio tokens + special tokens)
    Output: next-token logits over the full vocabulary

    Embeddings:
        token_emb:      maps token ID → d_model vector (weight-tied with lm_head)
        segment_emb:    3 types — 0=text, 1=prompt_audio, 2=target_audio
        group_pos_emb:  19 positions — which FSQ group within a frame (0 for text)

    The three embeddings are summed, then passed through 12 transformer blocks.
    """

    def __init__(self, config: ModelConfig, token_config: Optional[TokenConfig] = None):
        super().__init__()
        self.config = config
        self.token_config = token_config or TokenConfig()

        # Token embedding (shared with lm_head via weight tying)
        self.token_emb = nn.Embedding(self.token_config.vocab_size, config.d_model)

        # Segment embedding: which part of the sequence is this token from?
        # 0=text, 1=prompt_audio, 2=target_audio
        self.segment_emb = nn.Embedding(config.num_segments, config.d_model)

        # Group position embedding: which of the 19 FSQ groups is this token?
        # Text tokens and special tokens use group_pos=0
        self.group_pos_emb = nn.Embedding(config.groups_per_frame, config.d_model)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final norm before logits
        self.final_norm = RMSNorm(config.d_model)

        # lm_head: weight-tied with token_emb (saves ~12.6M params)
        # No separate nn.Linear — we'll use F.linear with token_emb.weight

        # Precompute RoPE frequencies (not a parameter, just a buffer)
        freqs_cis = precompute_rope_freqs(
            config.d_head, config.max_seq_len, config.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following GPT-2 / LLaMA conventions."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Scale output projection of attention and FFN by 1/sqrt(2*n_layers)
        # (reduces contribution of each layer at init, stabilizes training)
        scale = 1.0 / math.sqrt(2 * self.config.n_layers)
        for layer in self.layers:
            torch.nn.init.normal_(layer.attn.out_proj.weight, mean=0.0, std=0.02 * scale)
            torch.nn.init.normal_(layer.ffn.down.weight, mean=0.0, std=0.02 * scale)

    def forward(
        self,
        token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        group_pos_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[list] = None,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            token_ids:    [B, T] token indices (0 to vocab_size-1)
            segment_ids:  [B, T] segment type (0=text, 1=prompt, 2=target)
            group_pos_ids:[B, T] group position within frame (0-18, 0 for text)
            position_ids: [B, T] absolute positions for RoPE (default: 0,1,...,T-1)
            attention_mask:[B, 1, T, T_full] custom attention mask (None=causal)
            kv_cache:     list of (past_k, past_v) per layer, or None
            labels:       [B, T] target token IDs for loss computation
            loss_mask:    [B, T] binary mask — 1 where loss should be computed
                          (target audio + eos positions only)

        Returns:
            dict with:
                logits: [B, T, vocab_size]
                loss: scalar (if labels provided)
                kv_cache: list of (k, v) per layer
        """
        B, T = token_ids.shape

        # Sum the three embeddings
        x = self.token_emb(token_ids) + self.segment_emb(segment_ids) + self.group_pos_emb(group_pos_ids)
        # x: [B, T, d_model]

        # Default position IDs: 0, 1, 2, ..., T-1
        if position_ids is None:
            if kv_cache is not None and kv_cache[0] is not None:
                # During generation with cache: position = past_len
                past_len = kv_cache[0][0].shape[2]
                position_ids = torch.arange(past_len, past_len + T, device=x.device).unsqueeze(0).expand(B, T)
            else:
                position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)

        # Get RoPE frequencies on the right device
        freqs_cis = self.freqs_cis.to(x.device)

        # Pass through transformer blocks
        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            layer_kv = kv_cache[i] if kv_cache is not None else None
            x, new_kv = layer(x, freqs_cis, position_ids, layer_kv, attention_mask)
            new_kv_cache.append(new_kv)

        # Final norm
        x = self.final_norm(x)

        # Logits via weight-tied lm_head
        logits = F.linear(x, self.token_emb.weight)
        # logits: [B, T, vocab_size]

        result = {"logits": logits, "kv_cache": new_kv_cache}

        # Compute loss if labels provided
        if labels is not None:
            # Shift: predict position t+1 from position t
            # logits[:, :-1] predicts labels[:, 1:]
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            )
            # loss: [B * (T-1)]

            if loss_mask is not None:
                # Shift loss_mask to match shifted labels
                shift_mask = loss_mask[:, 1:].contiguous().view(-1)
                loss = (loss * shift_mask).sum() / shift_mask.sum().clamp(min=1)
            else:
                loss = loss.mean()

            result["loss"] = loss

        return result

    def estimate_params(self) -> int:
        """Count actual parameters (not the config estimate)."""
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate_next_token(
        self,
        token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        group_pos_ids: torch.Tensor,
        kv_cache: Optional[list] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Tuple[torch.Tensor, list]:
        """Generate a single next token (for use in autoregressive loop).

        Args:
            token_ids:    [B, 1] current token (or [B, T] for prefill)
            segment_ids:  [B, 1] or [B, T]
            group_pos_ids:[B, 1] or [B, T]
            kv_cache:     previous cache or None
            temperature:  sampling temperature
            top_k:        top-k filtering
            top_p:        nucleus sampling threshold

        Returns:
            next_token: [B, 1] sampled token ID
            new_kv_cache: updated cache
        """
        result = self.forward(
            token_ids, segment_ids, group_pos_ids,
            kv_cache=kv_cache,
        )
        logits = result["logits"][:, -1, :]  # [B, vocab_size]
        kv_cache = result["kv_cache"]

        # Temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # Top-k filtering
        if top_k is not None:
            topk_vals, _ = torch.topk(logits, top_k)
            logits[logits < topk_vals[:, -1:]] = float("-inf")

        # Top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above the threshold
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float("-inf")
            # Scatter back to original indices
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

        return next_token, kv_cache
