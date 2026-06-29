"""Configuration dataclasses for KoeTTS.

Single source of truth for all hyperparameters:
- TokenConfig: vocabulary layout for custom 97M model (special + characters + audio)
- QwenTokenConfig: vocabulary layout for Qwen3-0.6B experiment (extended Qwen3 vocab)
- ModelConfig: custom AR transformer architecture (~97M params)
- QwenModelConfig: Qwen3-0.6B-Base fine-tuning config
- TrainConfig: training hyperparameters (optimizers, schedule, DDP)
- TokenizerTrainConfig: JEPA tokenizer training hyperparameters
- CodecConfig: pretrained JEPA tokenizer settings
"""

from dataclasses import dataclass, field
from typing import List, Optional
import math


@dataclass
class TokenConfig:
    """Vocabulary layout for Experiment A (custom 97M model).

    IDs 0-4: special tokens
    IDs 5-204: characters (200 slots — ASCII printable + accented + punctuation)
    IDs 205-16588: audio tokens (16384 possible FSQ packed values)

    Total vocab size: 16589
    """

    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    text_sep_id: int = 3  # separates text from audio
    audio_sep_id: int = 4  # separates prompt audio from target audio

    num_special: int = 5
    num_chars: int = 200  # ASCII printable + accented chars + punctuation
    num_audio_tokens: int = 16384  # max packed FSQ value + 1 (4^7 = 16384)

    @property
    def audio_offset(self) -> int:
        """First audio token ID in vocab."""
        return self.num_special + self.num_chars  # 205

    @property
    def vocab_size(self) -> int:
        return self.num_special + self.num_chars + self.num_audio_tokens  # 16589


@dataclass
class QwenTokenConfig:
    """Vocabulary layout for Experiment B (Qwen3-0.6B fine-tuned).

    Uses Qwen3's existing BPE vocab (151,936 tokens) extended with:
    - 16384 audio tokens (IDs 151936-168319)
    - 2 new special tokens: text_sep (168320), audio_sep (168321)

    Total vocab size: 168322
    """

    qwen_vocab_size: int = 151936
    num_audio_tokens: int = 16384

    @property
    def audio_offset(self) -> int:
        """First audio token ID (appended after Qwen3 vocab)."""
        return self.qwen_vocab_size  # 151936

    @property
    def text_sep_id(self) -> int:
        return self.qwen_vocab_size + self.num_audio_tokens  # 168320

    @property
    def audio_sep_id(self) -> int:
        return self.qwen_vocab_size + self.num_audio_tokens + 1  # 168321

    @property
    def vocab_size(self) -> int:
        return self.qwen_vocab_size + self.num_audio_tokens + 2  # 168322


@dataclass
class ModelConfig:
    """AR transformer architecture (~97M params).

    12 layers, 768 dim, 12 heads, SwiGLU FFN.
    RoPE for position encoding, segment embeddings for text/prompt/target,
    group position embeddings for the 19 FSQ groups within each audio frame.
    """

    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    d_ffn: int = 2048  # SwiGLU hidden dim (gate + up projections)
    max_seq_len: int = 2048
    dropout: float = 0.0  # no dropout — rely on data/regularization

    # FSQ frame structure: 19 groups per 2.5Hz frame
    groups_per_frame: int = 19

    # Segment types: 0=text, 1=prompt_audio, 2=target_audio
    num_segments: int = 3

    # RoPE base frequency
    rope_theta: float = 10000.0

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def estimate_params(self) -> int:
        """Rough parameter count estimate."""
        # Embedding: vocab_size * d_model (weight-tied, counted once)
        embed = 16589 * self.d_model
        # Segment + group_pos embeddings
        aux_embed = (self.num_segments + self.groups_per_frame) * self.d_model
        # Per transformer block:
        #   attention: 4 * d_model^2 (Q, K, V, O)
        #   SwiGLU FFN: 3 * d_model * d_ffn (gate, up, down)
        #   RMSNorm: 2 * d_model (attn_norm + ffn_norm)
        per_block = 4 * self.d_model**2 + 3 * self.d_model * self.d_ffn + 2 * self.d_model
        blocks = self.n_layers * per_block
        # Final norm
        final_norm = self.d_model
        return embed + aux_embed + blocks + final_norm


@dataclass
class QwenModelConfig:
    """Qwen3-0.6B-Base fine-tuning config for Experiment B.

    Architecture (from pretrained, not configurable):
    - 28 layers, 1024 dim, 16 Q heads / 8 KV heads (GQA)
    - QK-Norm, SwiGLU, RMSNorm, RoPE (theta=1M)
    - head_dim=64, intermediate_size=3072

    We extend the vocab with audio tokens and fine-tune.
    """

    model_name: str = "Qwen/Qwen3-0.6B-Base"
    max_seq_len: int = 2048

    # FSQ frame structure (same as custom model)
    groups_per_frame: int = 19
    num_segments: int = 3

    # Training strategy for Experiment B
    # Include loss on text positions to preserve LLM knowledge
    # (validated by Sesame community: prevents "text space destruction")
    text_loss_weight: float = 0.5  # weight for text token CE loss
    text_only_data_ratio: float = 0.0  # fraction of batches with pure text (no audio)

    # Freeze/unfreeze schedule (None = train everything from start)
    freeze_layers_until_step: Optional[int] = None


@dataclass
class TrainConfig:
    """Training hyperparameters for the AR model."""

    # Muon optimizer (2D transformer weights)
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_nesterov: bool = True

    # AdamW optimizer (embeddings, norms, head, biases)
    adam_lr: float = 3e-4
    adam_betas: tuple = (0.9, 0.95)

    # Shared
    weight_decay: float = 0.01
    grad_clip: float = 1.0  # applied to AdamW params only; Muon has built-in norm control
    warmup_steps: int = 1000
    lr_decay_ratio: float = 0.1  # cosine decay to this fraction of max LR

    # Batch
    per_gpu_batch_size: int = 64
    gradient_accumulation_steps: int = 1

    # Precision
    dtype: str = "bfloat16"

    # CFG: probability of dropping text during training
    cfg_dropout: float = 0.1

    # Checkpointing
    save_every: int = 1000
    keep_last_n: int = 3

    # W&B
    wandb_project: str = "koe-tts"
    log_every: int = 10
    eval_every: int = 1000
    token_viz_every: int = 5000

    # Training duration
    max_steps: Optional[int] = None
    max_epochs: Optional[int] = None


@dataclass
class TokenizerTrainConfig:
    """Training hyperparameters for the JEPA tokenizer."""

    # Stage 1: JEPA encoder
    stage1_steps: int = 24000
    stage1_batch_size: int = 256  # 64 per GPU × 4 GPUs
    stage1_muon_lr: float = 0.02  # 2D encoder/conformer weights
    stage1_adam_lr: float = 1.5e-4  # 1D params (norms, SnakeBeta, DAAM)
    stage1_mask_ratio: float = 0.5
    stage1_ema_decay: float = 0.996

    # Stage 2: decoder + FSQ + discriminators
    stage2_steps: int = 29000
    stage2_batch_size: int = 24  # reduced from 32 — GAN backward OOMs on H200 at batch 32
    stage2_muon_lr: float = 0.02  # 2D decoder weights
    stage2_adam_lr_gen: float = 1.5e-4  # 1D generator params
    stage2_adam_lr_disc: float = 7.5e-5  # all discriminator params (AdamW only)
    stage2_disc_warmup: int = 5000
    stage2_lambda_stft: float = 2.0
    stage2_lambda_gan: float = 0.1

    # Shared
    adam_betas: tuple = (0.8, 0.99)  # paper's values
    weight_decay: float = 1e-3
    max_audio_seconds: float = 15.0

    # Checkpointing / logging (same infra as AR model)
    save_every: int = 500
    keep_last_n: int = 3
    wandb_project: str = "koe-tokenizer"
    log_every: int = 10
    eval_every: int = 500
    token_viz_every: int = 2000


@dataclass
class CodecConfig:
    """Pretrained JEPA tokenizer settings.

    Mirrors the architecture in Density-Adaptive-JEPA/train_fsqvae_jepa.py
    with the paper's hyperparameters (from argparse defaults).
    """

    sample_rate: int = 24000
    code_dim: int = 128
    channels: List[int] = field(default_factory=lambda: [64, 128, 256, 384, 512, 512])
    strides: List[int] = field(default_factory=lambda: [8, 8, 5, 5, 6])
    fsq_levels: List[int] = field(default_factory=lambda: [4, 4, 4, 4])
    group_size: int = 7  # dims packed per token

    # Conformer
    n_conformer: int = 8
    conformer_heads: int = 16
    n_res_blocks: int = 8  # reference uses 8; was 3 in v3

    # HiFi-GAN decoder
    hifi_kernels: List[int] = field(default_factory=lambda: [3, 7, 11, 15, 23, 32])
    use_decoder_gaatn: bool = False

    @property
    def hop_length(self) -> int:
        """Total downsampling factor."""
        result = 1
        for s in self.strides:
            result *= s
        return result  # 8*8*5*5*6 = 9600

    @property
    def frame_rate(self) -> float:
        """Frames per second."""
        return self.sample_rate / self.hop_length  # 24000/9600 = 2.5 Hz

    @property
    def num_groups(self) -> int:
        """Number of packed token groups per frame."""
        return math.ceil(self.code_dim / self.group_size)  # ceil(128/7) = 19

    @property
    def tokens_per_second(self) -> float:
        """Total tokens per second of audio."""
        return self.frame_rate * self.num_groups  # 2.5 * 19 = 47.5

    @property
    def max_packed_value(self) -> int:
        """Maximum value of a packed token (for a full group with fsq_levels[0] levels)."""
        return self.fsq_levels[0] ** self.group_size - 1

    def audio_length_to_tokens(self, seconds: float) -> int:
        """Estimate total tokens for a given audio duration."""
        frames = int(seconds * self.frame_rate)
        return frames * self.num_groups
