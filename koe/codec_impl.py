"""Forked model classes from Density-Adaptive-JEPA/train_fsqvae_jepa.py.

All bug fixes documented in FIXES.md are applied here.
Original line references are annotated with # [orig Lxxx].

Components:
- Utilities: FSQ packing/unpacking, mask creation, loss functions
- DAAM: GaussianAdaptiveAttention, GAttnGateG
- MR-STFT Loss
- SnakeBeta activation
- Encoder/Decoder blocks with optional DAAM gating
- ConformerBlock
- JEPAEncoder (Stage 1)
- WaveformJEPAFSQVAE (Stage 2: frozen encoder + FSQ + HiFi-GAN decoder)
- FiniteScalarQuantizer (FSQ)
- Discriminators (MPD + MSD)
"""

import math
import random
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================================
# Utilities
# ======================================================================

@torch.no_grad()
def _fsq_dim_radices(D: int, levels: List[int], device=None) -> torch.Tensor:
    """Build per-dimension radix vector for mixed-radix packing. [orig L57-63]"""
    assert D % len(levels) == 0, f"D={D} must be divisible by len(levels)={len(levels)}"
    per = D // len(levels)
    r = []
    for L in levels:
        r += [int(L)] * per
    return torch.tensor(r, dtype=torch.long, device=device)


@torch.no_grad()
def fsq_pack_indices(
    indices: torch.Tensor,
    levels: List[int],
    group_size: int = 7,
) -> torch.Tensor:
    """Pack FSQ indices into mixed-radix tokens via Horner's method. [orig L66-86]

    indices: [B, T, D] with values in [0, L-1] per dim
    Returns: [B, T, G] packed tokens, max value = prod(levels)^group_size - 1
    """
    B, T, D = indices.shape
    device = indices.device
    rad = _fsq_dim_radices(D, levels, device=device)
    G = (D + group_size - 1) // group_size

    pad = G * group_size - D
    if pad > 0:
        indices = torch.cat(
            [indices, torch.zeros(B, T, pad, dtype=indices.dtype, device=device)], dim=2
        )
        rad = torch.cat([rad, torch.ones(pad, dtype=rad.dtype, device=device)], dim=0)

    toks = torch.zeros(B, T, 0, dtype=torch.long, device=device)
    for g in range(G):
        s, e = g * group_size, (g + 1) * group_size
        chunk = indices[:, :, s:e].long()
        rchunk = rad[s:e].long()
        tok = torch.zeros(B, T, dtype=torch.long, device=device)
        for k in range(rchunk.numel() - 1, -1, -1):
            tok = chunk[:, :, k] + tok * rchunk[k]
        toks = torch.cat([toks, tok.unsqueeze(-1)], dim=-1)
    return toks


@torch.no_grad()
def fsq_unpack_indices(
    packed: torch.Tensor,
    levels: List[int],
    code_dim: int,
    group_size: int = 7,
) -> torch.Tensor:
    """Unpack mixed-radix tokens back to per-dim FSQ indices.

    packed: [B, T, G]
    Returns: [B, T, D] with values in [0, L-1] per dim
    """
    B, T, G = packed.shape
    device = packed.device
    D_padded = G * group_size
    rad = _fsq_dim_radices(code_dim, levels, device=device)

    pad = D_padded - code_dim
    if pad > 0:
        rad = torch.cat([rad, torch.ones(pad, dtype=rad.dtype, device=device)], dim=0)

    indices = torch.zeros(B, T, D_padded, dtype=torch.long, device=device)
    for g in range(G):
        s = g * group_size
        rchunk = rad[s : s + group_size].long()
        tok = packed[:, :, g].clone()
        for k in range(group_size):
            indices[:, :, s + k] = tok % rchunk[k]
            tok = tok // rchunk[k]

    return indices[:, :, :code_dim]


def create_jepa_mask(
    batch_size: int,
    seq_len: int,
    mask_ratio: float = 0.5,
    min_span: int = 4,
    max_span: int = 16,
    device: str = "cuda",
) -> torch.Tensor:
    """Create JEPA-style block masks for temporal sequences. [orig L137-160]

    Returns: mask [B, T] where 1=keep, 0=mask.

    FIX #6: De-duplicate masked indices before counting to achieve accurate
    mask ratios. Original code counted overlapping spans multiple times,
    yielding ~40-45% actual mask vs target 50%.
    """
    masks = torch.ones(batch_size, seq_len, device=device)

    for b in range(batch_size):
        num_to_mask = int(seq_len * mask_ratio)
        # FIX #6: Track actual unique masked positions
        masked_positions = set()

        while len(masked_positions) < num_to_mask:
            span_len = random.randint(min_span, max_span)
            start = random.randint(0, max(1, seq_len - span_len))
            end = min(start + span_len, seq_len)
            for pos in range(start, end):
                masked_positions.add(pos)

        for pos in masked_positions:
            masks[b, pos] = 0

    return masks


def feature_loss(fmap_r: List, fmap_g: List) -> torch.Tensor:
    """Feature matching loss between real and generated feature maps. [orig L1077-1082]"""
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += F.l1_loss(gl, rl.detach())
    return loss * 2


def discriminator_loss(dr_list: List, dg_list: List) -> torch.Tensor:
    """Hinge-style discriminator loss. [orig L1084-1089]"""
    loss = 0
    for dr, dg in zip(dr_list, dg_list):
        loss += F.mse_loss(dr, torch.ones_like(dr))
        loss += F.mse_loss(dg, torch.zeros_like(dg))
    return loss


def generator_loss(dg_list: List) -> torch.Tensor:
    """Generator adversarial loss. [orig L1091-1095]"""
    loss = 0
    for dg in dg_list:
        loss += F.mse_loss(dg, torch.ones_like(dg))
    return loss


# ======================================================================
# DAAM: Density Adaptive Attention Mechanism
# ======================================================================


class GaussianAdaptiveAttention(nn.Module):
    """Gaussian mixture gating — soft feature selection via density estimation. [orig L166-211]

    NOTE: self.eps and self.padding_value are stored but unused in forward().
    The forward method uses purpose-specific epsilon values instead:
      - 1e-6 for variance clamping (prevents sqrt(0) → NaN)
      - 1e-3 added to sigma (prevents infinitely narrow Gaussians)
      - 1e-8 in denominator (prevents division by zero)
    These are kept as-is from the original code; they serve different numerical
    stability roles and shouldn't all be the same value.
    """

    def __init__(
        self,
        norm_axis: int,
        num_heads: int,
        num_gaussians: int,
        padding_value=None,
        mean_offset_init: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.norm_axis = norm_axis
        self.num_heads = num_heads
        self.num_gaussians = num_gaussians
        self.padding_value = padding_value
        self.eps = eps

        self.mean_offsets = nn.Parameter(
            torch.full((num_gaussians,), float(mean_offset_init))
        )
        self.log_sigma = nn.Parameter(torch.full((num_gaussians,), math.log(0.5)))
        self.register_buffer(
            "_log_sqrt_2pi",
            torch.tensor(0.5 * math.log(2.0 * math.pi)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, return_attention_details: bool = False):
        with torch.cuda.amp.autocast(enabled=False):
            xf = x.float()
            mean_offsets = self.mean_offsets.float()

            mean = xf.mean(dim=self.norm_axis, keepdim=True)
            var = xf.var(dim=self.norm_axis, keepdim=True, unbiased=False)
            std = var.clamp_min(1e-6).sqrt()
            sigma = F.softplus(self.log_sigma.float()) + 1e-3

            log_terms = []
            for k in range(self.num_gaussians):
                z = (xf - (mean + mean_offsets[k])) / (std * sigma[k] + 1e-8)
                log_terms.append(
                    -0.5 * (z * z)
                    - torch.log(sigma[k])
                    - self._log_sqrt_2pi  # use pre-computed buffer instead of allocating each call
                )

            # log_terms -> log probabilities under each kth gaussian
            log_G = torch.stack(log_terms, dim=-1) # (B, 1, T, 4)
            log_gate = torch.logsumexp(log_G, dim=-1) - math.log(self.num_gaussians) # (B, 1, T) : mixture probability
            gate32 = torch.exp(log_gate)
            out32 = xf * gate32

        out = out32.to(x.dtype)
        if return_attention_details:
            return out, gate32.to(x.dtype)
        return out # (B, 1, T)



# Concrete example: batch=2, time=5 (tiny for illustration)
# Input to GAttnGateG: x = [2, 512, 5]  (512-channel feature map)

# Step 1: Project to 1 channel
# a = self.to_attn(x)  # Conv1d(512, 1, 1) → [2, 1, 5]

# Say a = [[[-0.3, 0.8, -0.1, 0.5, 1.2]],   ← sample 0
#           [[ 0.2, 0.0, -0.4, 0.1, 0.3]]]   ← sample 1

# Step 2: GAA forward
# For sample 0: mean = 0.42, std = 0.56
# For sample 1: mean = 0.04, std = 0.24

# With 4 Gaussians (offsets start at 0, sigmas start at ~0.6):
# Each Gaussian says "how likely is this time step under my distribution?"

# Gaussian 0 (offset=0.0):  peaks at the sample mean
#   sample 0, time 2 (val=-0.1): z = (-0.1 - 0.42) / (0.56*0.6) = -1.55 → low prob
#   sample 0, time 4 (val=1.2):  z = (1.2 - 0.42) / (0.56*0.6) = 2.32  → very low prob

# After logsumexp across 4 Gaussians and exp:
# gate might look like: [0.4, 0.8, 0.5, 0.7, 0.3]
#   ← high gate for time steps near Gaussian peaks
#   ← low gate for outlier time steps

# Step 3: Back in GAttnGateG
# scale = 1.0 + 0.05 * gate
#        = [1.02, 1.04, 1.025, 1.035, 1.015]
#
# y = x * scale   ← ALL 512 channels at time=1 get scaled by 1.04
#                  ← ALL 512 channels at time=4 get scaled by 1.015
#
# The effect is SUBTLE (alpha=0.05) — a ~2-4% boost/suppression
# But over 8 Conformer blocks × 5 encoder blocks, it adds up

class GAttnGateG(nn.Module):
    """DAAM gating wrapper with learnable alpha. [orig L213-227]"""

    def __init__(self, in_ch: int, num_gaussians: int = 4, cap: float = 0.2):
        super().__init__()
        self.to_attn = nn.Conv1d(in_ch, 1, 1)
        self.gaa = GaussianAdaptiveAttention(
            norm_axis=2, num_heads=1, num_gaussians=num_gaussians
        )
        self.alpha = nn.Parameter(torch.tensor(0.05))
        self.cap = cap

    def forward(self, x: torch.Tensor):
        a = self.to_attn(x) # Conv1d(in_ch, 1, 1) — project to 1 channel -> (B, 1, T)
        _, gate = self.gaa(a, return_attention_details=True) # (B, 1, T)
        scale = 1.0 + self.alpha * gate # (B, 1, T)
        # x -> (B, C, T)
        y = x * scale # (B, C, T) each channel scaled by same gate
        return y, gate



# ======================================================================
# MR-STFT Loss
# ======================================================================


class MRSTFTLoss(nn.Module):
    """Multi-resolution STFT loss for waveform reconstruction. [orig L233-285]"""

    def __init__(
        self,
        fft_sizes=(2048, 1024, 512, 256, 128),
        hop_sizes=(512, 256, 128, 64, 32),
        win_lengths=(2048, 1024, 512, 256, 128),
        mag_weight: float = 1.0,
        log_mag_weight: float = 1.0,
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.mag_weight = mag_weight
        self.log_mag_weight = log_mag_weight
        for w in win_lengths:
            self.register_buffer(f"window_{w}", torch.hann_window(w), persistent=False)

    def stft(self, x, fft_size, hop_size, win_length):
        window = getattr(self, f"window_{win_length}")
        x32 = x.float()
        w32 = window.to(device=x.device, dtype=torch.float32)
        return torch.stft(
            x32,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=w32,
            return_complex=True,
        )

    def forward(self, pred, target, lengths=None):
        if lengths is not None:
            B = pred.shape[0]
            tot = 0.0
            for b in range(B):
                L = lengths[b]
                p = pred[b : b + 1, :, :L]
                t = target[b : b + 1, :, :L]
                ssum = 0.0
                used = 0
                for n, h, w in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
                    if L < n:
                        continue
                    used += 1
                    ps = self.stft(p.squeeze(1), n, h, w)
                    ts = self.stft(t.squeeze(1), n, h, w)
                    pm, tm = ps.abs(), ts.abs()
                    ssum += self.mag_weight * F.l1_loss(pm, tm)
                    ssum += self.log_mag_weight * F.l1_loss(
                        (pm + 1e-5).log(), (tm + 1e-5).log()
                    )
                tot += ssum / max(1, used)
            return tot / B

        loss = 0.0
        for n, h, w in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            ps = self.stft(pred.squeeze(1), n, h, w)
            ts = self.stft(target.squeeze(1), n, h, w)
            pm, tm = ps.abs(), ts.abs()
            loss += self.mag_weight * F.l1_loss(pm, tm)
            loss += self.log_mag_weight * F.l1_loss(
                (pm + 1e-5).log(), (tm + 1e-5).log()
            )
        return loss / len(self.fft_sizes)


# ======================================================================
# SnakeBeta Activation
# ======================================================================


class SnakeBeta(nn.Module):
    """Snake activation with learnable frequency. [orig L464-474]"""

    def __init__(self, in_features: int, min_alpha: float = 1e-2, max_inv: float = 10.0):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(1, in_features, 1))
        self.min_alpha = min_alpha
        self.max_inv = max_inv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = F.softplus(self.raw) + self.min_alpha
        inv = (1.0 / alpha).clamp_max(self.max_inv)
        return x + inv * (torch.sin(alpha * x) ** 2) # (1, in_features, 1)


# ======================================================================
# Anti-Aliased Activation (BigVGAN-style)
# ======================================================================


def _kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    """Build a Kaiser-windowed sinc lowpass filter."""
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    time = (torch.arange(-half_size, half_size) + 0.5) if even else (torch.arange(kernel_size) - half_size)
    if cutoff == 0:
        filt = torch.zeros_like(time)
    else:
        filt = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
        filt = filt / filt.sum()
    return filt.view(1, 1, kernel_size)


class _LowPassFilter1d(nn.Module):
    def __init__(self, cutoff: float, half_width: float, stride: int, kernel_size: int = 12):
        super().__init__()
        self.stride = stride
        self.pad_left = kernel_size // 2 - int(kernel_size % 2 == 0)
        self.pad_right = kernel_size // 2
        self.register_buffer("filter", _kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)


class _UpSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.ratio = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * ratio + (kernel_size - ratio) // 2
        self.pad_right = self.pad * ratio + (kernel_size - ratio + 1) // 2
        self.register_buffer(
            "filter",
            _kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, kernel_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(
            x, self.filter.expand(C, -1, -1), stride=self.ratio, groups=C
        )
        return x[..., self.pad_left:-self.pad_right]


class _DownSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.lowpass = _LowPassFilter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio,
            stride=ratio, kernel_size=kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)


class AntiAliasedActivation(nn.Module):
    """Wraps any activation in upsample→activate→downsample (BigVGAN-style).

    Prevents aliasing from nonlinear activations by evaluating at 2x resolution
    and filtering back down with a Kaiser sinc lowpass.

    Runs in float32 for numerical stability (Kaiser filter coefficients
    lose precision in bf16). Casts back to input dtype after.
    """

    def __init__(self, activation: nn.Module, up_ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.act = activation
        self.upsample = _UpSample1d(up_ratio, kernel_size)
        self.downsample = _DownSample1d(up_ratio, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        # Cast upsample/downsample filters to float32 alongside input
        self.upsample.filter.data = self.upsample.filter.data.float()
        self.downsample.lowpass.filter.data = self.downsample.lowpass.filter.data.float()
        x = self.upsample(x.float())
        x = self.act(x)
        x = self.downsample(x)
        return x.to(dtype)


# ======================================================================
# Encoder / Decoder Blocks
# ======================================================================


class ResBlock(nn.Module):
    """Residual block with dilated convolutions. [orig L432-451]"""

    def __init__(self, channels: int, kernel_size: int = 3, dilation=(1, 3, 5),
                 use_weight_norm: bool = False):
        super().__init__()
        _wn = nn.utils.weight_norm if use_weight_norm else (lambda x: x)
        self.convs1 = nn.ModuleList(
            [
                _wn(nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=d,
                    padding=(kernel_size * d - d) // 2,
                ))
                for d in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                _wn(nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=(kernel_size - 1) // 2,
                ))
                for _ in dilation
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class MRFBlock(nn.Module):
    """Multi-receptive-field fusion block. [orig L453-462]"""

    def __init__(
        self,
        channels: int,
        kernels=(3, 7, 11),
        dilations=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        use_weight_norm: bool = False,
    ):
        super().__init__()
        self.resblocks = nn.ModuleList(
            [ResBlock(channels, k, d, use_weight_norm=use_weight_norm)
             for k, d in zip(kernels, dilations)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.resblocks[0](x)
        for b in self.resblocks[1:]:
            out = out + b(x)
        return out / len(self.resblocks)


class EncoderBlock(nn.Module):
    """Strided conv + residual blocks + optional DAAM gating. [orig L476-499]"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        n_res: int = 2,
        use_gaatn: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=2 * stride, stride=stride, padding=stride // 2
        )
        self.res_blocks = nn.ModuleList(
            [
                ResBlock(out_channels, kernel_size=3, dilation=(1, 3**i, 5**i))
                for i in range(n_res)   
            ]
        )
        self.snake = SnakeBeta(out_channels)
        self.use_gaatn = use_gaatn
        if use_gaatn:
            self.gaatn_gate = GAttnGateG(in_ch=out_channels, num_gaussians=4)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.snake(x)
        for b in self.res_blocks:
            x = b(x)
        gate = None
        if self.use_gaatn:
            x, gate = self.gaatn_gate(x)
        return x, gate


class HiFiDecoderBlock(nn.Module):
    """Transposed conv + MRF + optional DAAM gating. [orig L501-520]"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        kernels=(3, 7, 11, 15, 23, 32),
        use_gaatn: bool = True,
        use_weight_norm: bool = False,
        anti_alias: bool = False,
    ):
        super().__init__()
        _wn = nn.utils.weight_norm if use_weight_norm else (lambda x: x)
        snake = SnakeBeta(in_channels)
        self.snake = AntiAliasedActivation(snake) if anti_alias else snake
        self.deconv = _wn(nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=2 * stride, stride=stride, padding=stride // 2
        ))
        self.mrf = MRFBlock(out_channels, kernels, use_weight_norm=use_weight_norm)
        self.use_gaatn = use_gaatn
        if use_gaatn:
            self.gaatn_gate = GAttnGateG(in_ch=out_channels, num_gaussians=4)

    def forward(self, x: torch.Tensor):
        x = self.snake(x)
        x = self.deconv(x)
        x = self.mrf(x)
        gate = None
        if self.use_gaatn:
            x, gate = self.gaatn_gate(x)
        return x, gate


# ======================================================================
# ConformerBlock
# ======================================================================


class ConformerBlock(nn.Module):
    """Conformer block: FFN1 → MHSA → Conv → FFN2 → Norm. [orig L524-623]

    Operates on [B, D, T] (channel-first). Internally transposes for attention.
    """

    def __init__(self, dim: int, heads: int = 8, ff_mult: int = 4, conv_kernel: int = 31, dropout: float = 0.1):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.attn_drop_p = dropout

        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

        self.norm_attn = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

        self.conv = nn.Sequential(
            nn.GroupNorm(1, dim),
            nn.Conv1d(dim, 2 * dim, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(dim, dim, kernel_size=conv_kernel, padding=conv_kernel // 2, groups=dim),
            nn.GroupNorm(1, dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Dropout(dropout),
        )

        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

        self.norm_final = nn.LayerNorm(dim)

    def _shape_qkv(self, x: torch.Tensor):
        B, T, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(t):
            return t.view(B, T, self.heads, self.head_dim).transpose(1, 2).contiguous()

        return split_heads(q), split_heads(k), split_heads(v)

    def _merge_heads(self, x: torch.Tensor):
        B, H, T, Hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Hd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D, T]
        s = x.transpose(1, 2).contiguous()  # [B, T, D]

        s = s + 0.5 * self.ff1(s)

        s_norm = self.norm_attn(s)
        q, k, v = self._shape_qkv(s_norm)
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=False,
        )
        attn = self._merge_heads(attn)
        attn = self.out_proj(attn)
        s = s + attn

        c = s.transpose(1, 2)  # [B, D, T]
        c = c + self.conv(c)

        s = c.transpose(1, 2)  # [B, T, D]
        s = s + 0.5 * self.ff2(s)

        s = self.norm_final(s)
        return s.transpose(1, 2).contiguous()  # [B, D, T]


# ======================================================================
# FSQ Quantizer
# ======================================================================


class FiniteScalarQuantizer(nn.Module):
    """Finite Scalar Quantization with STE. [orig L373-426]

    levels=[4,4,4,4], dim=128, normalized=True.
    Boundaries: linspace(-1+1/L, 1-1/L, L) = [-0.75, -0.25, 0.25, 0.75] for L=4.

    Uses LayerNorm to normalize encoder outputs into the FSQ boundary range,
    avoiding tanh saturation that causes codebook collapse. The STE operates
    on the normalized representation so gradients flow through pre_norm/pre_scale.
    """

    def __init__(
        self,
        levels: List[int],
        dim: int,
        normalized: bool = True,
        use_tanh: bool = False,
        temperature: float = 1.0,
        entropy_weight: float = 0.0,
    ):
        super().__init__()
        assert dim % len(levels) == 0
        self.levels = levels
        self.dim = dim
        self.normalized = normalized
        self.use_tanh = use_tanh
        self.temperature = temperature
        self.entropy_weight = entropy_weight
        self.dims_per_level = dim // len(levels)
        self.boundaries = nn.ModuleList()
        for L in levels:
            if normalized:
                bounds = torch.linspace(-1 + 1 / L, 1 - 1 / L, L)
            else:
                bounds = torch.linspace(1 / (2 * L), 1 - 1 / (2 * L), L)
            mod = nn.Module()
            mod.register_buffer("bounds", bounds)
            self.boundaries.append(mod)
        self.implicit_codebook_size = math.prod(levels)

        # LayerNorm to normalize encoder outputs into boundary range
        self.pre_norm = nn.LayerNorm(dim)
        # Learnable scale to map normalized outputs into [-1, 1] range
        self.pre_scale = nn.Parameter(torch.ones(1) * 0.5)

    def normalize(self, z_e: torch.Tensor) -> torch.Tensor:
        """Normalize encoder outputs into FSQ boundary range.

        z_e: [B, D, T] (channel-first, raw encoder output)
        Returns: z_norm [B, D, T] (normalized, in boundary range)
        """
        B, D, T = z_e.shape
        z = z_e.permute(0, 2, 1).contiguous()  # [B, T, D]
        z_flat = z.view(-1, D)
        z_flat = self.pre_norm(z_flat) * self.pre_scale
        if self.use_tanh:
            z_flat = torch.tanh(z_flat / self.temperature) * self.temperature
        return z_flat.view(B, -1, D).permute(0, 2, 1).contiguous()  # [B, D, T]

    def quantize(self, z_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize already-normalized features to nearest boundary.

        z_norm: [B, D, T] (channel-first, already normalized)
        Returns: z_q [B, D, T], indices [B, T, D]
        """
        B, D, T = z_norm.shape
        assert D == self.dim
        z = z_norm.permute(0, 2, 1).contiguous()  # [B, T, D]
        z_flat = z.view(-1, D)
        z_q_list, indices_list = [], []
        for i, L in enumerate(self.levels):
            s = i * self.dims_per_level
            e = s + self.dims_per_level
            z_group = z_flat[:, s:e]
            bounds = self.boundaries[i].bounds
            dist = (z_group.unsqueeze(-1) - bounds.view(1, 1, L)).abs()
            idx = torch.argmin(dist, dim=-1)
            z_q_group = bounds[idx]
            z_q_list.append(z_q_group)
            indices_list.append(idx)
        z_q_flat = torch.cat(z_q_list, dim=1)
        all_idx = torch.cat(indices_list, dim=1)
        z_q = z_q_flat.view(B, -1, D).permute(0, 2, 1).contiguous()  # [B, D, T]
        return z_q, all_idx.view(B, -1, D)  # indices: [B, T, D]

    def entropy_metric(self, indices: torch.Tensor) -> float:
        """Compute codebook entropy as a monitoring metric (non-differentiable).

        indices: [B, T, D] with values in [0, L-1]
        Returns: float in [0, 1], where 1 = perfectly uniform
        """
        B, T, D = indices.shape
        total_entropy = 0.0
        offset = 0
        for i, L in enumerate(self.levels):
            s = offset
            e = offset + self.dims_per_level
            idx_group = indices[:, :, s:e].reshape(-1)
            counts = torch.zeros(L, device=indices.device, dtype=torch.float32)
            counts.scatter_add_(0, idx_group.long(), torch.ones_like(idx_group, dtype=torch.float32))
            probs = counts / counts.sum().clamp(min=1)
            entropy = -(probs * (probs + 1e-8).log()).sum().item()
            max_entropy = math.log(L)
            total_entropy += entropy / max_entropy
            offset = e
        return total_entropy / len(self.levels)

    def forward(
        self, z_e: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize with straight-through estimator.

        STE operates on the normalized representation so gradients flow
        through pre_norm and pre_scale to the reconstruction loss.

        z_e: [B, D, T] (raw encoder output)
        Returns: z_q [B, D, T], indices [B, T, D], aux_loss (always 0 for FSQ)
        """
        # Normalize into FSQ range (differentiable)
        z_norm = self.normalize(z_e)
        # Quantize (non-differentiable)
        z_q, indices = self.quantize(z_norm)
        # STE on normalized representation — gradients flow to pre_norm/pre_scale
        z_q = z_norm + (z_q - z_norm).detach()
        aux_loss = torch.tensor(0.0, device=z_e.device, dtype=z_e.dtype)
        return z_q, indices, aux_loss

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert indices back to quantized values.

        indices: [B, T, D] with values in [0, L-1]
        Returns: z_q [B, D, T]
        """
        B, T, D = indices.shape
        assert D == self.dim
        z_q_list = []
        for i, L in enumerate(self.levels):
            s = i * self.dims_per_level
            e = s + self.dims_per_level
            idx = indices[:, :, s:e]
            bounds = self.boundaries[i].bounds
            z_q_list.append(bounds[idx])
        z_q = torch.cat(z_q_list, dim=-1)  # [B, T, D]
        return z_q.permute(0, 2, 1).contiguous()  # [B, D, T]


# ======================================================================
# JEPA Encoder (Stage 1)
# ======================================================================


def _conv1d_out_len(L: int, k: int, s: int, p: int, d: int = 1) -> int:
    return (L + 2 * p - d * (k - 1) - 1) // s + 1


def jepa_time_len_from_wav(T_wav: int, strides: List[int]) -> int:
    """Compute encoder output time length from waveform length."""
    L = _conv1d_out_len(T_wav, k=7, s=1, p=3)
    for s in strides:
        k = 2 * s
        p = s // 2
        L = _conv1d_out_len(L, k=k, s=s, p=p)
    return L


class JEPAEncoder(nn.Module):
    """JEPA self-supervised encoder with EMA target network. [orig L646-844]

    Stage 1: Trains context encoder + predictor to predict EMA target features
    at masked positions. After training, only the context encoder is used.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        code_dim: int = 128,
        channels: List[int] = (32, 64, 128, 256, 512),
        strides: List[int] = (4, 4, 5, 4, 4),
        n_res_blocks: int = 2,
        n_conformer: int = 2,
        conformer_heads: int = 4,
        use_gaatn: bool = True,
    ):
        super().__init__()
        channels = list(channels)
        strides = list(strides)
        assert len(channels) == len(strides) + 1
        self.sample_rate = sample_rate
        self.strides = strides
        self.hop_length = math.prod(strides)
        self.code_dim = code_dim

        # Context encoder (online, trainable)
        self.input_conv = nn.Conv1d(1, channels[0], kernel_size=7, padding=3)
        self.encoder = nn.ModuleList(
            [
                EncoderBlock(channels[i], channels[i + 1], strides[i], n_res_blocks, use_gaatn)
                for i in range(len(strides))
            ]
        )
        self.bottleneck_proj = nn.Conv1d(channels[-1], code_dim, 1)
        self.conformer_blocks = nn.ModuleList(
            [
                ConformerBlock(code_dim, heads=conformer_heads, dropout=0.1)
                for _ in range(n_conformer)
            ]
        )

        # Learnable mask tokens
        self.mask_token = nn.Parameter(torch.zeros(1, code_dim, 1))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Predictor
        self.predictor = nn.Sequential(
            nn.Conv1d(code_dim, code_dim * 2, 1),
            nn.GELU(),
            ConformerBlock(code_dim * 2, heads=conformer_heads, dropout=0.1),
            nn.Conv1d(code_dim * 2, code_dim * 2, 1),
            nn.GELU(),
            ConformerBlock(code_dim * 2, heads=conformer_heads, dropout=0.1),
            nn.Conv1d(code_dim * 2, code_dim, 1),
        )

        self.apply(self._init_weights)

        # EMA target encoder (frozen copy)
        import copy

        self.ema_decay = 0.996
        self.target_encoder = nn.ModuleDict(
            {
                "input_conv": copy.deepcopy(self.input_conv),
                "encoder": copy.deepcopy(self.encoder),
                "bottleneck_proj": copy.deepcopy(self.bottleneck_proj),
                "conformer_blocks": copy.deepcopy(self.conformer_blocks),
            }
        )
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.target_encoder.eval()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def _target_encode(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.target_encoder["input_conv"](wav)
        for enc in self.target_encoder["encoder"]:
            x, _ = enc(x)
        z = self.target_encoder["bottleneck_proj"](x)
        for conf in self.target_encoder["conformer_blocks"]:
            if z.shape[-1] < 2:
                break
            z = conf(z)
        return z

    @torch.no_grad()
    def update_target_encoder(self, decay: Optional[float] = None):
        d = self.ema_decay if decay is None else decay

        def ema_update(tgt_mod, src_mod):
            for (_, p_t), (_, p_s) in zip(
                tgt_mod.named_parameters(), src_mod.named_parameters()
            ):
                p_t.data.mul_(d).add_(p_s.data, alpha=1.0 - d)
            for (_, b_t), (_, b_s) in zip(
                tgt_mod.named_buffers(), src_mod.named_buffers()
            ):
                b_t.data.copy_(b_s.data)

        ema_update(self.target_encoder["input_conv"], self.input_conv)
        ema_update(self.target_encoder["encoder"], self.encoder)
        ema_update(self.target_encoder["bottleneck_proj"], self.bottleneck_proj)
        for tb, sb in zip(
            self.target_encoder["conformer_blocks"], self.conformer_blocks
        ):
            ema_update(tb, sb)

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """Encode waveform to representations (online encoder).

        wav: [B, 1, T_wav]
        Returns: z [B, code_dim, T_z]
        """
        x = self.input_conv(wav)
        for enc in self.encoder:
            x, _ = enc(x)
        z = self.bottleneck_proj(x)
        for conf in self.conformer_blocks:
            if z.shape[-1] < 2:
                break
            z = conf(z)
        return z

    def forward(
        self, wav: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """JEPA forward pass.

        Args:
            wav: [B, 1, T_wav]
            mask: [B, T_z] binary mask (1=visible, 0=masked)

        Returns:
            z_context, z_pred, mask, z_target
        """
        if mask is None:
            return self.encode(wav), None, None, None

        z_context = self.encode(wav)

        with torch.no_grad():
            z_target = self._target_encode(wav)

        B, C, Tz = z_target.shape
        mask_3d = mask.unsqueeze(1).to(device=z_context.device, dtype=z_context.dtype)
        mask_tokens = self.mask_token.expand(B, -1, Tz).to(
            device=z_context.device, dtype=z_context.dtype
        )
        z_masked = z_context * mask_3d + mask_tokens * (1 - mask_3d)
        z_pred = self.predictor(z_masked)

        return z_context, z_pred, mask, z_target


# ======================================================================
# WaveformJEPAFSQVAE (Stage 2: Frozen Encoder + FSQ + HiFi-GAN Decoder)
# ======================================================================

class WaveformJEPAFSQVAE(nn.Module):
    """Full codec: JEPA encoder → FSQ → HiFi-GAN decoder. [orig L850-960]

    In Stage 2, the encoder is frozen. Only FSQ + decoder + discriminators train.

    FIX #4: When freeze_encoder=True, we set requires_grad_(False) on encoder
    and exclude it from the optimizer entirely, saving ~240MB optimizer memory.
    """

    def __init__(
        self,
        jepa_encoder: Optional[JEPAEncoder] = None,
        fsq_levels: List[int] = (8, 8, 8, 8),
        channels: List[int] = (32, 64, 128, 256, 512),
        strides: List[int] = (2, 2, 4, 5, 8),
        use_tanh: bool = True,
        temperature: float = 1.0,
        hifi_kernels: List[int] = (3, 7, 11, 15, 23, 32),
        use_decoder_gaatn: bool = False,
        freeze_encoder: bool = False,
        code_dim: int = 128,
        sample_rate: int = 24000,
        n_res_blocks: int = 2,
        n_conformer: int = 2,
        conformer_heads: int = 8,
    ):
        super().__init__()
        channels = list(channels)
        strides = list(strides)

        if jepa_encoder is None:
            self.encoder = JEPAEncoder(
                sample_rate=sample_rate,
                code_dim=code_dim,
                channels=channels,
                strides=strides,
                n_res_blocks=n_res_blocks,
                n_conformer=n_conformer,
                conformer_heads=conformer_heads,
                use_gaatn=True,
            )
        else:
            self.encoder = jepa_encoder

        code_dim = self.encoder.code_dim
        self.sample_rate = self.encoder.sample_rate
        self.strides = strides
        self.hop_length = self.encoder.hop_length
        self.code_dim = code_dim

        # FIX #4: Actually freeze encoder properly
        self._freeze_encoder = freeze_encoder
        if freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

        self.fsq = FiniteScalarQuantizer(
            levels=list(fsq_levels),
            dim=code_dim,
            normalized=True,
            use_tanh=use_tanh,
            temperature=temperature,
        )

        self.bottleneck_unproj = nn.Conv1d(code_dim, channels[-1], 1)
        self.decoder = nn.ModuleList(
            [
                HiFiDecoderBlock(
                    channels[i + 1], channels[i], strides[i], list(hifi_kernels), use_decoder_gaatn
                )
                for i in range(len(strides) - 1, -1, -1)
            ]
        )
        self.output_conv = nn.Conv1d(channels[0], 1, kernel_size=7, padding=3)
        self.final_activation = nn.Tanh()

        self.fsq_levels = list(fsq_levels)
        self._last_dec_attn_maps: List[torch.Tensor] = []

        # Only init decoder weights
        for m in [self.bottleneck_unproj, self.output_conv, *self.decoder]:
            self._init_weights(m)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Module):
            for subm in m.modules():
                if isinstance(subm, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                    nn.init.trunc_normal_(subm.weight, std=0.02)
                    if subm.bias is not None:
                        nn.init.zeros_(subm.bias)

    def train(self, mode=True):
        """Override to keep encoder in eval mode when frozen."""
        super().train(mode)
        if self._freeze_encoder:
            self.encoder.eval()
        return self

    def encode(
        self, wav: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode waveform to quantized representations.

        wav: [B, 1, T_wav]
        Returns: z_q [B, D, T_z], z_e [B, D, T_z], indices [B, T_z, D], aux_loss
        """
        z_e = self.encoder.encode(wav)
        z_q, indices, aux_loss = self.fsq(z_e)
        return z_q, z_e, indices, aux_loss

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized representations to waveform.

        z_q: [B, D, T_z]  (D = code_dim = 128)
        Returns: wav [B, 1, T_wav]
        """
        x = self.bottleneck_unproj(z_q)
        self._last_dec_attn_maps = []
        for dec in self.decoder:
            x, gate = dec(x)
            if not self.training and gate is not None:
                self._last_dec_attn_maps.append(gate.detach())
        wav = self.output_conv(x)
        wav = self.final_activation(wav)
        return wav

    def forward(self, wav: torch.Tensor):
        original_length = wav.shape[-1]
        z_q, z_e, indices, aux_loss = self.encode(wav)
        rec = self.decode(z_q)

        if rec.shape[-1] > original_length:
            rec = rec[..., :original_length]
        elif rec.shape[-1] < original_length:
            rec = F.pad(rec, (0, original_length - rec.shape[-1]))

        return rec, indices, aux_loss, z_e

    def forward_from_z_e(self, z_e: torch.Tensor, wav_length: int):
        """Forward pass from pre-computed encoder outputs (skip encoder).

        z_e: [B, D, T_z] pre-cached encoder outputs
        wav_length: original waveform length for output length matching
        """
        z_q, indices, aux_loss = self.fsq(z_e)
        rec = self.decode(z_q)

        if rec.shape[-1] > wav_length:
            rec = rec[..., :wav_length]
        elif rec.shape[-1] < wav_length:
            rec = F.pad(rec, (0, wav_length - rec.shape[-1]))

        return rec, indices, aux_loss, z_e

    def trainable_params(self) -> List[nn.Parameter]:
        """Return only parameters that should be optimized.

        When freeze_encoder=True, excludes encoder params entirely.
        When freeze_encoder=False (fine-tuning), includes all params.
        """
        params = []
        for name, module in self.named_children():
            if name == "encoder" and self._freeze_encoder:
                continue  # Frozen in Stage 2
            params.extend(p for p in module.parameters() if p.requires_grad)
        return params


# ======================================================================
# Discriminators
# ======================================================================


class PeriodDiscriminator(nn.Module):
    """Multi-period discriminator sub-network. [orig L966-992]"""

    def __init__(self, period: int):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)),
                nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
                nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
                nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
                nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
            ]
        )
        self.post = nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))
        self.use_grad_ckpt = False

    def _conv_block(self, x, conv):
        return F.leaky_relu(conv(x), 0.1)

    def forward(self, x: torch.Tensor):
        from torch.utils.checkpoint import checkpoint
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t += n_pad
        x = x.view(b, c, t // self.period, self.period)
        fmaps = []
        for conv in self.convs:
            if self.use_grad_ckpt and x.requires_grad:
                x = checkpoint(self._conv_block, x, conv, use_reentrant=False)
            else:
                x = F.leaky_relu(conv(x), 0.1)
            fmaps.append(x)
        x = self.post(x)
        fmaps.append(x)
        return x.flatten(1), fmaps


class ScaleDiscriminator(nn.Module):
    """Single-scale 1D conv discriminator. [orig L994-1014]"""

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(1, 16, 15, 1, padding=7),
                nn.Conv1d(16, 64, 41, 4, groups=4, padding=20),
                nn.Conv1d(64, 256, 41, 4, groups=16, padding=20),
                nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20),
                nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20),
                nn.Conv1d(1024, 1024, 5, 1, padding=2),
            ]
        )
        self.post = nn.Conv1d(1024, 1, 3, 1, padding=1)
        self.use_grad_ckpt = False

    def _conv_block(self, x, conv):
        return F.leaky_relu(conv(x), 0.1)

    def forward(self, x: torch.Tensor):
        from torch.utils.checkpoint import checkpoint
        fmaps = []
        for c in self.convs:
            if self.use_grad_ckpt and x.requires_grad:
                x = checkpoint(self._conv_block, x, c, use_reentrant=False)
            else:
                x = F.leaky_relu(c(x), 0.1)
            fmaps.append(x)
        x = self.post(x)
        fmaps.append(x)
        return x.flatten(1), fmaps


class MultiPeriodDiscriminator(nn.Module):
    """Multi-period discriminator ensemble. [orig L1016-1030]"""

    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.ds = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        rs, gs, fr, fg = [], [], [], []
        for d in self.ds:
            r, fr_ = d(y)
            g, fg_ = d(y_hat)
            rs.append(r)
            gs.append(g)
            fr.append(fr_)
            fg.append(fg_)
        return rs, gs, fr, fg


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator with cascaded pooling. [orig L1032-1051]

    FIX #1: Original code applied pooling independently (pool(audio) for each scale
    beyond the first). Standard MSD from HiFi-GAN cascades: scale 0 sees raw,
    scale 1 sees pool(raw), scale 2 sees pool(pool(raw)). The original had
    `yy = p(y)` always pooling from the original `y`, making scales 1 and 2
    see the same temporal resolution when they have the same pool config.

    Fix: Apply cumulative pooling so each scale sees progressively downsampled audio.
    """

    def __init__(self):
        super().__init__()
        self.ds = nn.ModuleList(
            [ScaleDiscriminator(), ScaleDiscriminator(), ScaleDiscriminator()]
        )
        self.pools = nn.ModuleList(
            [
                nn.Identity(),
                nn.AvgPool1d(4, 2, padding=2),
                nn.AvgPool1d(4, 2, padding=2),
            ]
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        rs, gs, fr, fg = [], [], [], []
        # FIX #1: Cumulative pooling — each scale sees progressively downsampled audio
        yy, gg = y, y_hat
        for d, p in zip(self.ds, self.pools):
            yy = p(yy)
            gg = p(gg)
            r, fr_ = d(yy)
            g, fg_ = d(gg)
            rs.append(r)
            gs.append(g)
            fr.append(fr_)
            fg.append(fg_)
        return rs, gs, fr, fg


class STFTDiscriminator(nn.Module):
    """2D convolutional discriminator on magnitude spectrograms with spectral normalization."""

    def __init__(self, n_fft=1024, hop_length=256, win_length=1024, channels=16):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))
        sn = nn.utils.spectral_norm
        self.convs = nn.ModuleList([
            sn(nn.Conv2d(1, channels, (7, 5), (2, 2), padding=(3, 2))),
            sn(nn.Conv2d(channels, channels, (5, 3), (2, 1), padding=(2, 1))),
            sn(nn.Conv2d(channels, channels, (5, 3), (2, 1), padding=(2, 1))),
            sn(nn.Conv2d(channels, channels, 3, 1, padding=1)),
        ])
        self.post = sn(nn.Conv2d(channels, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        with torch.amp.autocast("cuda", enabled=False):
            xf = x.squeeze(1).float()
            spec = torch.stft(
                xf, self.n_fft, self.hop_length, self.win_length,
                window=self.window, return_complex=True,
            )
            x = spec.abs().unsqueeze(1).to(dtype)
        fmaps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmaps.append(x)
        x = self.post(x)
        fmaps.append(x)
        return x.flatten(1), fmaps


class MultiScaleSTFTDiscriminator(nn.Module):
    """MS-STFT discriminator — complementary to MPD+MSD for frequency-domain discrimination."""

    def __init__(self, n_ffts=(2048, 1024, 512)):
        super().__init__()
        self.ds = nn.ModuleList([
            STFTDiscriminator(n_fft=n, hop_length=n // 4, win_length=n)
            for n in n_ffts
        ])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        rs, gs, fr, fg = [], [], [], []
        for d in self.ds:
            r, fr_ = d(y)
            g, fg_ = d(y_hat)
            rs.append(r)
            gs.append(g)
            fr.append(fr_)
            fg.append(fg_)
        return rs, gs, fr, fg


# ======================================================================
# Collate / Data Utilities
# ======================================================================


def make_collate_fn(sample_rate: int, hop_length: int):
    """Create a collate function that ensures minimum sequence length. [orig L1053-1075]"""

    def collate_fn(batch):
        if not batch:
            return None
        T = max(x.shape[0] for x in batch)
        min_samples = max(int(sample_rate * 0.5), 4 * hop_length)
        T = max(T, min_samples)
        T = ((T + hop_length - 1) // hop_length) * hop_length
        xs = torch.stack([F.pad(x, (0, T - x.shape[0])) for x in batch], dim=0)
        return xs.unsqueeze(1)  # [B, 1, T]

    return collate_fn


def set_requires_grad(module: nn.Module, requires_grad: bool):
    """Toggle gradient computation for all parameters in a module.

    FIX #5: Used to disable discriminator gradients during generator step
    and vice versa, saving memory and compute.
    """
    for p in module.parameters():
        p.requires_grad = requires_grad
