"""Conditional flow matching decoder: JEPA encoder features -> mel spectrograms.

Architecture:
  JEPA features (128-dim, 12.5 Hz)
    -> ConditionEncoder (upsample to mel rate ~93.75 Hz)
    -> VelocityNetwork predicts flow v(x_t, t, condition)
    -> Euler ODE solve at inference
    -> mel spectrogram

Training uses optimal transport conditional flow matching:
  x_t = (1-t)*noise + t*mel_target
  v_target = mel_target - noise
  loss = MSE(v_pred, v_target)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class FlowDecoderConfig:
    jepa_dim: int = 128
    mel_dim: int = 100
    hidden_dim: int = 512
    n_blocks: int = 8
    kernel_size: int = 7
    jepa_hop: int = 1920
    mel_hop: int = 256
    sample_rate: int = 24000
    n_fft: int = 1024
    fmin: float = 0.0
    fmax: float = 12000.0


class ConditionEncoder(nn.Module):
    """Upsample JEPA features from 12.5 Hz to mel frame rate (~93.75 Hz).

    Upsampling strategy:
      ConvTranspose1d factors [5, 3] = 15x upsample
      Then Conv1d stride 2 = learned downsampling by 2
      Net effect: 15 / 2 = 7.5x = jepa_hop / mel_hop = 1920 / 256
    """

    def __init__(self, cfg: FlowDecoderConfig):
        super().__init__()
        self.cfg = cfg

        # Project JEPA dim to hidden
        self.input_proj = nn.Conv1d(cfg.jepa_dim, cfg.hidden_dim, kernel_size=1)

        # Transposed conv upsampling: 5x then 3x = 15x
        upsample_factors = [5, 3]
        layers = []
        for factor in upsample_factors:
            layers.append(
                nn.ConvTranspose1d(
                    cfg.hidden_dim, cfg.hidden_dim,
                    kernel_size=factor * 2,
                    stride=factor,
                    padding=factor // 2,
                )
            )
            layers.append(nn.GELU())
        self.upsample = nn.Sequential(*layers)

        # Learned stride-2 downsampling: 15x -> 7.5x effective
        self.downsample = nn.Conv1d(
            cfg.hidden_dim, cfg.hidden_dim,
            kernel_size=5, stride=2, padding=2,
        )

        # Residual refinement blocks
        self.refine = nn.Sequential(
            _PlainResBlock(cfg.hidden_dim, cfg.kernel_size),
            _PlainResBlock(cfg.hidden_dim, cfg.kernel_size),
        )

    def forward(self, z_e: torch.Tensor, target_len: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            z_e: [B, jepa_dim, T_jepa] JEPA encoder features
            target_len: desired output length T_mel (for precise trimming/padding)
        Returns:
            cond: [B, hidden_dim, T_mel]
        """
        x = self.input_proj(z_e)       # [B, hidden_dim, T_jepa]
        x = self.upsample(x)           # [B, hidden_dim, T_jepa * 15]
        x = self.downsample(x)         # [B, hidden_dim, ~T_jepa * 7.5]

        if target_len is not None:
            if x.size(-1) > target_len:
                x = x[:, :, :target_len]
            elif x.size(-1) < target_len:
                x = F.pad(x, (0, target_len - x.size(-1)))

        x = self.refine(x)             # [B, hidden_dim, T_mel]
        return x


class _PlainResBlock(nn.Module):
    """Simple residual conv block (no time conditioning)."""

    def __init__(self, dim: int, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.GroupNorm(8, dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size, padding=pad),
            nn.GroupNorm(8, dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size, padding=pad),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for flow matching timestep t."""

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] or scalar, timestep in [0, 1]
        Returns:
            emb: [B, hidden_dim]
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)

        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, embed_dim]

        return self.mlp(emb)


class ResBlock(nn.Module):
    """Residual block with time conditioning for the velocity network."""

    def __init__(self, dim: int, kernel_size: int = 7, time_dim: int = 512):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=pad)
        self.time_proj = nn.Linear(time_dim, dim)
        self.norm1 = nn.GroupNorm(8, dim)
        self.norm2 = nn.GroupNorm(8, dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, dim, T]
            t_emb: [B, time_dim]
        Returns:
            [B, dim, T]
        """
        h = self.norm1(x)
        h = F.gelu(h)
        h = self.conv1(h)
        h = h + self.time_proj(t_emb).unsqueeze(-1)  # add time conditioning
        h = self.norm2(h)
        h = F.gelu(h)
        h = self.conv2(h)
        return x + h


class VelocityNetwork(nn.Module):
    """1D ResNet that predicts the velocity field v(x_t, t, condition).

    Input: concatenation of x_t (mel_dim) and condition (hidden_dim) along channels.
    Output: predicted velocity in mel space (mel_dim channels).
    """

    def __init__(self, cfg: FlowDecoderConfig):
        super().__init__()
        in_channels = cfg.mel_dim + cfg.hidden_dim

        # Project concatenated input to hidden_dim
        self.input_proj = nn.Conv1d(in_channels, cfg.hidden_dim, kernel_size=1)

        # Time embedding
        self.time_embed = TimeEmbedding(embed_dim=cfg.hidden_dim, hidden_dim=cfg.hidden_dim)

        # Stack of time-conditioned residual blocks
        self.blocks = nn.ModuleList([
            ResBlock(cfg.hidden_dim, kernel_size=cfg.kernel_size, time_dim=cfg.hidden_dim)
            for _ in range(cfg.n_blocks)
        ])

        # Output projection to mel_dim
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, cfg.hidden_dim),
            nn.GELU(),
            nn.Conv1d(cfg.hidden_dim, cfg.mel_dim, kernel_size=1),
        )

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_t: [B, mel_dim, T_mel] noisy mel at time t
            t: [B] timestep in [0, 1]
            condition: [B, hidden_dim, T_mel] encoded JEPA features
        Returns:
            v: [B, mel_dim, T_mel] predicted velocity
        """
        # Concatenate noisy mel and condition
        h = torch.cat([x_t, condition], dim=1)  # [B, mel_dim + hidden_dim, T_mel]
        h = self.input_proj(h)                   # [B, hidden_dim, T_mel]

        # Time embedding
        t_emb = self.time_embed(t)               # [B, hidden_dim]

        # Residual blocks with time conditioning
        for block in self.blocks:
            h = block(h, t_emb)

        # Project to mel_dim
        v = self.output_proj(h)                  # [B, mel_dim, T_mel]
        return v


class FlowMatchingDecoder(nn.Module):
    """Conditional flow matching decoder: JEPA features -> mel spectrograms.

    Training: Learns a velocity field that transports Gaussian noise to mel spectrograms,
    conditioned on JEPA encoder features.

    Inference: Integrates the learned ODE from noise to mel using Euler steps.
    """

    def __init__(self, cfg: FlowDecoderConfig = None):
        super().__init__()
        if cfg is None:
            cfg = FlowDecoderConfig()
        self.cfg = cfg

        self.condition_encoder = ConditionEncoder(cfg)
        self.velocity_network = VelocityNetwork(cfg)

    def _mel_len_from_jepa(self, T_jepa: int) -> int:
        """Compute expected mel length from JEPA sequence length."""
        return math.ceil(T_jepa * self.cfg.jepa_hop / self.cfg.mel_hop)

    def compute_loss(
        self, z_e: torch.Tensor, mel_target: torch.Tensor
    ) -> torch.Tensor:
        """Compute conditional flow matching training loss.

        Args:
            z_e: [B, jepa_dim, T_jepa] JEPA encoder features
            mel_target: [B, mel_dim, T_mel] ground truth log-mel spectrogram
        Returns:
            loss: scalar MSE loss between predicted and target velocity
        """
        B = z_e.size(0)
        T_mel = mel_target.size(-1)
        device = z_e.device

        # Encode condition (upsample JEPA features to mel rate)
        condition = self.condition_encoder(z_e, target_len=T_mel)  # [B, hidden_dim, T_mel]

        # Sample timestep t ~ U(0, 1) per batch element
        t = torch.rand(B, device=device, dtype=torch.float32)  # [B]

        # Sample noise
        noise = torch.randn_like(mel_target, dtype=torch.float32)

        # Ensure mel_target is float32 for flow computations
        mel_f32 = mel_target.float()

        # Optimal transport interpolation: x_t = (1 - t) * noise + t * mel
        t_expand = t[:, None, None]  # [B, 1, 1]
        x_t = (1.0 - t_expand) * noise + t_expand * mel_f32

        # Target velocity: v = mel - noise
        v_target = mel_f32 - noise

        # Predict velocity
        v_pred = self.velocity_network(x_t, t, condition)

        # MSE loss
        loss = F.mse_loss(v_pred.float(), v_target)
        return loss

    @torch.no_grad()
    def sample(
        self,
        z_e: torch.Tensor,
        n_steps: int = 32,
        target_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate mel spectrogram via Euler ODE integration.

        Args:
            z_e: [B, jepa_dim, T_jepa] JEPA encoder features
            n_steps: number of Euler integration steps
            target_len: output mel length; if None, computed from JEPA length
        Returns:
            mel: [B, mel_dim, T_mel] predicted log-mel spectrogram
        """
        B = z_e.size(0)
        T_jepa = z_e.size(-1)
        device = z_e.device

        if target_len is None:
            target_len = self._mel_len_from_jepa(T_jepa)

        # Compute condition once (reused across all ODE steps)
        condition = self.condition_encoder(z_e, target_len=target_len)  # [B, hidden_dim, T_mel]

        # Start from Gaussian noise
        x = torch.randn(B, self.cfg.mel_dim, target_len, device=device, dtype=torch.float32)

        dt = 1.0 / n_steps

        # Euler integration from t=0 to t=1
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=device, dtype=torch.float32)
            v = self.velocity_network(x, t, condition)
            x = x + v.float() * dt

        return x


def extract_mel(wav: torch.Tensor, cfg: FlowDecoderConfig) -> torch.Tensor:
    """Extract log-mel spectrogram matching BigVGAN's expected input.

    Args:
        wav: [B, T] waveform at 24kHz
        cfg: FlowDecoderConfig with mel extraction params
    Returns:
        mel: [B, mel_dim, T_mel] log-mel spectrogram
    """
    import torchaudio

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.mel_hop,
        win_length=cfg.n_fft,
        n_mels=cfg.mel_dim,
        f_min=cfg.fmin,
        f_max=cfg.fmax,
        power=1.0,
        norm=None,
        mel_scale="slaney",
    ).to(wav.device)

    mel = mel_transform(wav)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel
