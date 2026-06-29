"""BigVGAN adapter: JEPA encoder features → mel spectrogram → BigVGAN vocoder.

Instead of training a HiFi-GAN decoder from scratch (Stage 2), this module:
1. Takes frozen JEPA encoder output (128-dim @ 2.5 Hz)
2. Upsamples + projects to 100-band mel spectrogram @ 93.75 Hz (BigVGAN's input)
3. Feeds predicted mels into frozen pretrained BigVGAN for waveform synthesis

The adapter is small (~5M params) and trains with simple L1 mel loss against
ground truth mel spectrograms. No GAN training, no discriminators.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


def _compute_upsample_factors(jepa_hop: int, mel_hop: int = 256):
    """Compute upsample factors and subsample rate from hop length ratio.

    Returns (upsample_factors, subsample) such that:
        product(upsample_factors) / subsample == jepa_hop / mel_hop
    """
    ratio = jepa_hop / mel_hop  # e.g. 9600/256=37.5 or 1920/256=7.5
    # Strategy: upsample by 2*ratio (integer), then subsample by 2
    int_ratio = int(ratio * 2)  # 75 for 37.5x, 15 for 7.5x
    subsample = 2

    # Factorize int_ratio into small factors for ConvTranspose1d layers
    factors = []
    remaining = int_ratio
    for p in [5, 3, 2]:
        while remaining % p == 0 and remaining > 1:
            factors.append(p)
            remaining //= p
    if remaining > 1:
        factors.append(remaining)
    factors.sort(reverse=True)
    return tuple(factors), subsample


@dataclass
class BigVGANAdapterConfig:
    """Config for the mel prediction adapter."""
    jepa_dim: int = 128           # JEPA encoder output dim
    n_mels: int = 100             # BigVGAN expects 100 mel bands
    hidden_dim: int = 512         # intermediate channels
    jepa_hop: int = 9600          # JEPA encoder hop length (product of strides)
    mel_hop: int = 256            # BigVGAN mel hop length
    kernel_size: int = 7
    n_conv_layers: int = 3        # conv layers after upsampling for refinement

    # BigVGAN mel extraction params (must match pretrained model)
    sample_rate: int = 24000
    n_fft: int = 1024
    hop_size: int = 256
    win_size: int = 1024
    fmin: float = 0.0
    fmax: float = 12000.0

    @property
    def upsample_factors(self):
        factors, _ = _compute_upsample_factors(self.jepa_hop, self.mel_hop)
        return factors

    @property
    def subsample(self):
        _, sub = _compute_upsample_factors(self.jepa_hop, self.mel_hop)
        return sub


class ResBlock1d(nn.Module):
    """Residual block with dilated convolutions."""
    def __init__(self, dim, kernel_size=7, dilation=1, dropout=0.0):
        super().__init__()
        pad = (kernel_size * dilation - dilation) // 2
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding=pad, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class MelAdapter(nn.Module):
    """Predicts mel spectrograms from JEPA encoder features.

    Input: [B, 128, T_jepa] JEPA encoder output
    Output: [B, 100, T_mel] log-mel spectrogram for BigVGAN

    Upsampling ratio is computed from jepa_hop / mel_hop.
    """

    def __init__(self, cfg: BigVGANAdapterConfig = None, dropout: float = 0.1):
        super().__init__()
        if cfg is None:
            cfg = BigVGANAdapterConfig()
        self.cfg = cfg
        self.subsample = cfg.subsample

        # Project JEPA dims to hidden
        self.input_proj = nn.Conv1d(cfg.jepa_dim, cfg.hidden_dim, 1)

        # Learnable upsampling via transposed convolutions with residual blocks
        upsample_layers = []
        in_ch = cfg.hidden_dim
        for i, factor in enumerate(cfg.upsample_factors):
            out_ch = cfg.hidden_dim
            upsample_layers.extend([
                nn.ConvTranspose1d(
                    in_ch, out_ch,
                    kernel_size=factor * 2,
                    stride=factor,
                    padding=factor // 2,
                ),
                nn.GELU(),
                nn.Dropout(dropout),
                ResBlock1d(out_ch, kernel_size=7, dilation=1, dropout=dropout),
                ResBlock1d(out_ch, kernel_size=7, dilation=3, dropout=dropout),
            ])
            in_ch = out_ch
        self.upsample = nn.Sequential(*upsample_layers)

        # Learned anti-aliased downsampling (replaces naive x[:, :, ::2])
        if self.subsample > 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=2 * self.subsample + 1,
                          stride=self.subsample, padding=self.subsample),
                nn.GELU(),
            )
        else:
            self.downsample = None

        # Refinement with residual blocks at multiple dilations
        refine_layers = []
        for d in [1, 2, 4, 1, 2, 4]:
            refine_layers.append(ResBlock1d(cfg.hidden_dim, kernel_size=cfg.kernel_size, dilation=d, dropout=dropout))
        self.refine = nn.Sequential(*refine_layers)

        # Output projection to mel bands
        self.output_proj = nn.Conv1d(cfg.hidden_dim, cfg.n_mels, 1)

    def forward(self, z: torch.Tensor, target_len: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            z: [B, 128, T_jepa] JEPA encoder features
            target_len: expected output mel length (for precise trimming)
        Returns:
            mel: [B, 100, T_mel] predicted log-mel spectrogram
        """
        x = self.input_proj(z)          # [B, hidden, T_jepa]
        x = self.upsample(x)           # [B, hidden, T_jepa * upsample_product]

        # Learned downsampling to exact ratio (e.g. 15x -> 7.5x)
        if self.downsample is not None:
            x = self.downsample(x)

        if target_len is not None:
            if x.size(-1) > target_len:
                x = x[:, :, :target_len]
            elif x.size(-1) < target_len:
                x = F.pad(x, (0, target_len - x.size(-1)))

        x = self.refine(x)             # [B, hidden, T_mel]
        mel = self.output_proj(x)       # [B, 100, T_mel]
        return mel


class MelDiscriminator(nn.Module):
    """Multi-scale 2D convolutional discriminator on mel spectrograms.

    Operates on [B, 1, n_mels, T_mel] patches at multiple scales.
    Returns list of per-scale (logits, [features]) for hinge loss + feature matching.
    """

    def __init__(self, n_mels: int = 100, n_scales: int = 3):
        super().__init__()
        self.discriminators = nn.ModuleList()
        for _ in range(n_scales):
            self.discriminators.append(self._make_disc(n_mels))
        # Downsampling for multi-scale
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

    @staticmethod
    def _make_disc(n_mels: int) -> nn.Module:
        """Single-scale discriminator: 4 strided conv layers + output."""
        return nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(3, 9), stride=(1, 1), padding=(1, 4)),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.LeakyReLU(0.2),
            ),
            nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ])

    def forward(self, mel: torch.Tensor):
        """
        Args:
            mel: [B, n_mels, T] mel spectrogram
        Returns:
            results: list of (logits, [features]) per scale
        """
        x = mel.unsqueeze(1)  # [B, 1, n_mels, T]
        results = []
        for disc_layers in self.discriminators:
            feats = []
            h = x
            for layer in disc_layers[:-1]:
                h = layer(h)
                feats.append(h)
            logits = disc_layers[-1](h)
            results.append((logits, feats))
            x = self.pool(x)  # downsample for next scale
        return results


def mel_disc_loss(disc_real, disc_fake):
    """Hinge loss for discriminator."""
    loss = 0.0
    for (real_logits, _), (fake_logits, _) in zip(disc_real, disc_fake):
        loss += torch.mean(F.relu(1.0 - real_logits))
        loss += torch.mean(F.relu(1.0 + fake_logits))
    return loss / len(disc_real)


def mel_gen_loss(disc_fake):
    """Hinge generator loss + feature matching."""
    adv_loss = 0.0
    for fake_logits, _ in disc_fake:
        adv_loss += -torch.mean(fake_logits)
    adv_loss = adv_loss / len(disc_fake)
    return adv_loss


def mel_feat_match_loss(disc_real, disc_fake):
    """L1 feature matching loss across all scales and layers."""
    loss = 0.0
    n = 0
    for (_, real_feats), (_, fake_feats) in zip(disc_real, disc_fake):
        for rf, ff in zip(real_feats, fake_feats):
            loss += F.l1_loss(ff, rf.detach())
            n += 1
    return loss / max(n, 1)


def extract_mel(wav: torch.Tensor, cfg: BigVGANAdapterConfig) -> torch.Tensor:
    """Extract log-mel spectrogram matching BigVGAN's expected input.

    Args:
        wav: [B, T] waveform at 24kHz
        cfg: adapter config with mel params
    Returns:
        mel: [B, 100, T_mel] log-mel spectrogram
    """
    import torchaudio

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_size,
        win_length=cfg.win_size,
        n_mels=cfg.n_mels,
        f_min=cfg.fmin,
        f_max=cfg.fmax,
        power=1.0,
        norm=None,
        mel_scale="slaney",
    ).to(wav.device)

    mel = mel_transform(wav)
    # Log mel (clamp for numerical stability)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel


def load_bigvgan(device="cuda"):
    """Load pretrained BigVGAN v2 24kHz vocoder.

    Downloads from HuggingFace Hub and loads manually to avoid
    huggingface_hub API version incompatibilities.
    """
    import json
    from huggingface_hub import hf_hub_download

    repo_id = "nvidia/bigvgan_v2_24khz_100band_256x"

    # Download config and weights
    config_path = hf_hub_download(repo_id, "config.json")
    weights_path = hf_hub_download(repo_id, "bigvgan_generator.pt")

    with open(config_path) as f:
        config = json.load(f)

    # Create AttrDict-like config
    class AttrDict(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                raise AttributeError(key)

    h = AttrDict(config)

    import bigvgan as bigvgan_module
    model = bigvgan_module.BigVGAN(h, use_cuda_kernel=False)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    # Handle state_dict wrapper
    if "generator" in state_dict:
        state_dict = state_dict["generator"]
    model.load_state_dict(state_dict)
    model.remove_weight_norm()
    model.h = h  # attach config for mel extraction
    model = model.eval().to(device)

    for p in model.parameters():
        p.requires_grad = False

    return model


@torch.no_grad()
def bigvgan_synthesize(bigvgan_model, mel: torch.Tensor) -> torch.Tensor:
    """Synthesize waveform from mel spectrogram using frozen BigVGAN.

    Args:
        bigvgan_model: pretrained BigVGAN model
        mel: [B, 100, T_mel] log-mel spectrogram
    Returns:
        wav: [B, 1, T_time] waveform
    """
    return bigvgan_model(mel)
