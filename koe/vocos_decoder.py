"""Vocos-style decoder: JEPA encoder features -> STFT coefficients -> iSTFT -> waveform.

Instead of predicting mel spectrograms and using an external vocoder, this module
directly predicts STFT magnitude and phase, then reconstructs the waveform via iSTFT.

Architecture:
  1. Input projection: 128 -> 512
  2. Upsample from 12.5 Hz to ~93.75 Hz (7.5x): ConvTranspose1d [5, 3] then Conv1d stride-2 downsample
  3. ConvNeXt backbone: 8 blocks
  4. Dual output heads: log-magnitude + phase -> complex STFT -> iSTFT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class VocosDecoderConfig:
    jepa_dim: int = 128
    hidden_dim: int = 512
    n_fft: int = 1024
    hop_length: int = 256
    sample_rate: int = 24000
    n_convnext_blocks: int = 8
    convnext_kernel: int = 7
    convnext_mult: int = 4
    jepa_hop: int = 1920  # strides product for v9


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style 1D block: depthwise conv + LayerNorm + pointwise MLP."""

    def __init__(self, dim: int, kernel_size: int = 7, mult: int = 4):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, dim * mult)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim * mult, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # [B, T, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)  # [B, C, T]
        return residual + x


class VocosDecoder(nn.Module):
    """Vocos-style STFT decoder for JEPA encoder features.

    Input:  [B, jepa_dim, T_jepa]
    Output: [B, T_wav] waveform at sample_rate

    Upsampling ratio is computed from jepa_hop / stft_hop to support any frame rate.
    """

    def __init__(self, cfg: VocosDecoderConfig = None):
        super().__init__()
        if cfg is None:
            cfg = VocosDecoderConfig()
        self.cfg = cfg

        n_freqs = cfg.n_fft // 2 + 1  # 513 for n_fft=1024

        # Input projection
        self.input_proj = nn.Conv1d(cfg.jepa_dim, cfg.hidden_dim, kernel_size=1)

        # Compute upsampling ratio: jepa_hop / stft_hop
        # e.g. 12.5Hz: 1920/256=7.5x, 25Hz: 960/256=3.75x
        target_ratio = cfg.jepa_hop / cfg.hop_length

        # Upsampling: ConvTranspose1d [5, 3] = 15x, then learned downsample
        upsample_factors = [5, 3]
        upsample_total = 1
        for f in upsample_factors:
            upsample_total *= f  # 15

        # Downsample stride to hit the target ratio: 15 / stride = target_ratio
        downsample_stride = max(1, round(upsample_total / target_ratio))
        self._downsample_stride = downsample_stride

        upsample_layers = []
        for factor in upsample_factors:
            upsample_layers.extend([
                nn.ConvTranspose1d(
                    cfg.hidden_dim, cfg.hidden_dim,
                    kernel_size=factor * 2,
                    stride=factor,
                    padding=factor // 2,
                ),
                nn.GELU(),
            ])
        self.upsample = nn.Sequential(*upsample_layers)

        # Learned downsample: 15x -> target_ratio
        self.downsample = nn.Sequential(
            nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim,
                      kernel_size=2 * downsample_stride + 1,
                      stride=downsample_stride,
                      padding=downsample_stride),
            nn.GELU(),
        )

        # ConvNeXt backbone
        self.backbone = nn.Sequential(*[
            ConvNeXtBlock(cfg.hidden_dim, kernel_size=cfg.convnext_kernel, mult=cfg.convnext_mult)
            for _ in range(cfg.n_convnext_blocks)
        ])

        # Output heads
        # Log-magnitude head: predict n_freqs channels, apply exp to get magnitude
        self.mag_head = nn.Conv1d(cfg.hidden_dim, n_freqs, kernel_size=1)

        # Phase head: predict n_freqs channels (unconstrained angle in radians)
        self.phase_head = nn.Conv1d(cfg.hidden_dim, n_freqs, kernel_size=1)

        # Register hann window as buffer
        self.register_buffer("window", torch.hann_window(cfg.n_fft), persistent=False)

    def forward(self, z: torch.Tensor, target_len: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            z: [B, 128, T_jepa] JEPA encoder features
            target_len: desired output waveform length for trimming
        Returns:
            wav: [B, T_wav] reconstructed waveform
        """
        cfg = self.cfg

        # Input projection
        x = self.input_proj(z)  # [B, hidden, T_jepa]

        # Upsample 15x then downsample 2x -> 7.5x
        x = self.upsample(x)   # [B, hidden, ~T_jepa * 15]
        x = self.downsample(x)  # [B, hidden, ~T_jepa * 7.5]

        # ConvNeXt backbone
        x = self.backbone(x)    # [B, hidden, T_stft]

        # Predict STFT coefficients (keep native dtype through conv, float32 for STFT math)
        log_mag = self.mag_head(x)        # [B, n_freqs, T_stft]
        phase = self.phase_head(x)        # [B, n_freqs, T_stft]

        # Float32 for exp/polar numerical stability
        mag = torch.exp(log_mag.float())
        complex_spec = torch.polar(mag, phase.float())  # [B, n_freqs, T_stft]

        # iSTFT to reconstruct waveform (transpose to [B, n_freqs, T_stft] already correct)
        window = self.window.float()
        wav = torch.istft(
            complex_spec,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.n_fft,
            window=window,
        )  # [B, T_wav]

        # Trim to target length
        if target_len is not None:
            if wav.size(-1) > target_len:
                wav = wav[:, :target_len]
            elif wav.size(-1) < target_len:
                wav = F.pad(wav, (0, target_len - wav.size(-1)))

        return wav
