"""KoeTTSQwen: Qwen3-0.6B-Base fine-tuned for TTS (Experiment B).

Loads the pretrained Qwen3-0.6B-Base model and extends its vocabulary
with 16384 audio tokens + 2 new special tokens (text_sep, audio_sep).

Architecture (from pretrained, not configurable):
    28 layers, 1024 dim, 16 Q heads / 8 KV heads (GQA)
    QK-Norm, SwiGLU, RMSNorm, RoPE (theta=1M)
    head_dim=64, intermediate_size=3072

Vocab extension:
    Qwen3 vocab (151,936) + audio tokens (16,384) + text_sep + audio_sep = 168,322
    New token embeddings initialized randomly (std=0.02)
    lm_head resized to match (Qwen3 uses tied embeddings)

Text loss preservation:
    CE on text positions with weight 0.5 to prevent "text space destruction"
    (validated by Sesame Discord community)

Exposes the same interface as KoeTTS:
    model.forward(token_ids, segment_ids, group_pos_ids, ...) → dict with logits, loss
    model.generate_next_token(...) → next_token, kv_cache
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from koe.config import QwenModelConfig, QwenTokenConfig


class KoeTTSQwen(nn.Module):
    """Qwen3-0.6B-Base wrapper for Experiment B.

    Wraps HuggingFace Qwen3 model with:
    - Extended vocabulary (audio tokens + new specials)
    - Segment embeddings (text/prompt/target)
    - Group position embeddings (19 FSQ groups per frame)
    - Text loss preservation (weight 0.5 on text positions)
    - Same forward/generate interface as KoeTTS
    """

    def __init__(
        self,
        config: QwenModelConfig,
        token_config: Optional[QwenTokenConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.token_config = token_config or QwenTokenConfig()

        from transformers import AutoModelForCausalLM

        # Load pretrained Qwen3-0.6B-Base
        self.qwen = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Extend vocabulary: add audio tokens + new special tokens
        self._extend_vocab()

        # Segment embeddings: 0=text, 1=prompt_audio, 2=target_audio
        # Added to Qwen's hidden states after the embedding layer
        hidden_size = self.qwen.config.hidden_size  # 1024
        self.segment_emb = nn.Embedding(config.num_segments, hidden_size)
        nn.init.normal_(self.segment_emb.weight, std=0.02)

        # Group position embeddings: 19 positions within each audio frame
        self.group_pos_emb = nn.Embedding(config.groups_per_frame, hidden_size)
        nn.init.normal_(self.group_pos_emb.weight, std=0.02)

    def _extend_vocab(self):
        """Extend Qwen3's vocabulary with audio tokens and new specials.

        Adds 16,386 new tokens:
        - 16,384 audio tokens (IDs 151936-168319)
        - text_sep (ID 168320)
        - audio_sep (ID 168321)

        New embeddings initialized with small random values.
        """
        tc = self.token_config
        old_vocab = self.qwen.config.vocab_size

        # Resize embeddings (handles both input embeddings and lm_head)
        self.qwen.resize_token_embeddings(tc.vocab_size)

        # Initialize new token embeddings with small random values
        with torch.no_grad():
            embed_weight = self.qwen.get_input_embeddings().weight
            new_tokens_start = old_vocab
            embed_weight[new_tokens_start:].normal_(mean=0.0, std=0.02)

    @property
    def groups_per_frame(self) -> int:
        return self.config.groups_per_frame

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
        """Forward pass matching KoeTTS interface.

        Args:
            token_ids:     [B, T] token indices
            segment_ids:   [B, T] segment type (0=text, 1=prompt, 2=target)
            group_pos_ids: [B, T] group position within frame (0-18, 0 for text)
            position_ids:  [B, T] absolute positions (None for auto)
            attention_mask:[B, T] 1/0 mask for padding (NOT the causal mask)
            kv_cache:      HuggingFace past_key_values format
            labels:        [B, T] target token IDs for loss computation
            loss_mask:     [B, T] binary mask — 1 for target audio + eos

        Returns:
            dict with logits, loss (if labels), kv_cache
        """
        B, T = token_ids.shape

        # Get input embeddings from Qwen's embedding layer
        inputs_embeds = self.qwen.get_input_embeddings()(token_ids)

        # Add segment + group position embeddings
        inputs_embeds = inputs_embeds + self.segment_emb(segment_ids) + self.group_pos_emb(group_pos_ids)

        # Build HuggingFace-style attention mask (1 = attend, 0 = ignore)
        hf_attention_mask = attention_mask
        if hf_attention_mask is None:
            hf_attention_mask = torch.ones(B, T, dtype=torch.long, device=token_ids.device)

        # Forward through Qwen3
        outputs = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=hf_attention_mask,
            position_ids=position_ids,
            past_key_values=kv_cache,
            use_cache=True,
        )

        logits = outputs.logits  # [B, T, vocab_size]
        new_kv_cache = outputs.past_key_values

        result = {"logits": logits, "kv_cache": new_kv_cache}

        # Compute loss if labels provided
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            )
            # loss: [B * (T-1)]

            if loss_mask is not None:
                shift_mask = loss_mask[:, 1:].contiguous().view(-1)

                # Text loss preservation for Experiment B:
                # Apply text_loss_weight to text positions, full weight to audio
                if self.config.text_loss_weight > 0:
                    shift_segments = segment_ids[:, 1:].contiguous().view(-1)
                    is_text = (shift_segments == 0).float()
                    is_audio = shift_mask  # loss_mask already selects target audio

                    # Weighted loss: full weight on audio, text_loss_weight on text
                    weights = is_audio + self.config.text_loss_weight * is_text * (1 - is_audio)
                    loss = (loss * weights).sum() / weights.sum().clamp(min=1)
                else:
                    loss = (loss * shift_mask).sum() / shift_mask.sum().clamp(min=1)
            else:
                loss = loss.mean()

            result["loss"] = loss

        return result

    def estimate_params(self) -> int:
        """Count actual parameters."""
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
        """Generate a single next token (same interface as KoeTTS).

        Args:
            token_ids:     [B, 1] current token (or [B, T] for prefill)
            segment_ids:   [B, 1] or [B, T]
            group_pos_ids: [B, 1] or [B, T]
            kv_cache:      HuggingFace past_key_values
            temperature:   sampling temperature
            top_k:         top-k filtering
            top_p:         nucleus sampling threshold

        Returns:
            next_token: [B, 1] sampled token ID
            new_kv_cache: updated cache
        """
        result = self.forward(
            token_ids, segment_ids, group_pos_ids, kv_cache=kv_cache,
        )
        logits = result["logits"][:, -1, :]  # [B, vocab_size]
        kv_cache = result["kv_cache"]

        # Temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Top-k
        if top_k is not None:
            topk_vals, _ = torch.topk(logits, top_k)
            logits[logits < topk_vals[:, -1:]] = float("-inf")

        # Top-p (nucleus)
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

        return next_token, kv_cache
