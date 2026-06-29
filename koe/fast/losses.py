"""Loss functions for Stage 2 decoder training.

WavLMPerceptualLoss: L1 on frozen WavLM-Base intermediate features.
    Based on StableCodec (ICLR 2025), which found perceptual SSL losses
    essential for intelligible speech when using FSQ quantization.

PhaseAwareSTFTLoss: Multi-resolution STFT magnitude + instantaneous frequency loss.
    Extends standard magnitude-only STFT loss with L1 on the phase derivative
    (instantaneous frequency), improving temporal alignment.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# ---------------------------------------------------------------------------
# WavLM Perceptual Loss
# ---------------------------------------------------------------------------


class WavLMPerceptualLoss(nn.Module):
    """L1 loss on frozen WavLM-Base intermediate hidden states.

    WavLM-Base has 12 transformer layers. We extract hidden states from
    the specified layers and compute mean L1 distance between pred and target
    representations. This captures perceptual similarity that spectral losses miss.

    The model is lazy-loaded on first forward call to avoid blocking at init time
    (useful when the loss may not be used in every code path).

    Args:
        model_name: HuggingFace model identifier for WavLM.
        layers: Which transformer layer outputs to extract (1-indexed, up to 12).
        device: Device to place the WavLM model on.
    """

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base",  # ~95M params, ~190MB bf16
        layers: tuple = (6, 12),  # WavLM-Base has 12 layers; extract from these
        device: str = "cuda",
    ):
        super().__init__()
        self.model_name = model_name
        self.layers = layers
        self.device = device
        self._model: Optional[nn.Module] = None

    def _load_model(self) -> None:
        """Lazy-load and freeze WavLM on first use."""
        from transformers import WavLMModel

        self._model = WavLMModel.from_pretrained(
            self.model_name,
            output_hidden_states=True,
        )
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad_(False)
        self._model.to(device=self.device, dtype=torch.float32)

    def _extract_features(self, wav: torch.Tensor) -> list[torch.Tensor]:
        """Run WavLM and return hidden states from the configured layers.

        WavLM params are frozen (requires_grad=False) so no gradients accumulate
        on them, but we do NOT use torch.no_grad() because we need gradients to
        flow through the input waveform back to the decoder.

        Args:
            wav: [B, T_16k] mono waveform at 16 kHz, float32.

        Returns:
            List of hidden state tensors, one per requested layer.
        """
        outputs = self._model(wav, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of (B, T_frames, D), len = 13 (embedding + 12 layers)
        return [hidden_states[i] for i in self.layers]

    def forward(self, pred_wav: torch.Tensor, target_wav: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss between predicted and target waveforms.

        Gradients flow through pred_wav → resample → WavLM → L1 loss, allowing
        the decoder to optimize for perceptual quality. WavLM weights are frozen
        but act as a fixed feature extractor that gradients pass through.

        Target features are computed without gradients (detached).

        Args:
            pred_wav: [B, 1, T] predicted waveform at 24 kHz (any dtype).
            target_wav: [B, 1, T] target waveform at 24 kHz (any dtype).

        Returns:
            Scalar loss tensor on the same device and in the same dtype as pred_wav.
        """
        if self._model is None:
            self._load_model()

        input_device = pred_wav.device
        input_dtype = pred_wav.dtype

        # [B, 1, T] -> [B, T] and cast to float32 for WavLM
        pred_mono = pred_wav.squeeze(1).float()
        target_mono = target_wav.squeeze(1).float()

        # Resample 24kHz -> 16kHz (WavLM's expected sample rate)
        pred_16k = torchaudio.functional.resample(pred_mono, orig_freq=24000, new_freq=16000)
        target_16k = torchaudio.functional.resample(target_mono, orig_freq=24000, new_freq=16000)

        # Move to model device if needed
        pred_16k = pred_16k.to(self.device)
        target_16k = target_16k.to(self.device)

        # Extract pred features WITH grad (so decoder gets gradients)
        pred_features = self._extract_features(pred_16k)

        # Extract target features WITHOUT grad (no need to backprop through target)
        with torch.no_grad():
            target_features = self._extract_features(target_16k)
            # Detach target features so they're treated as fixed targets
            target_features = [tf.detach() for tf in target_features]

        # Mean L1 across layers
        loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        for pf, tf in zip(pred_features, target_features):
            loss = loss + F.l1_loss(pf, tf)
        loss = loss / len(self.layers)

        return loss.to(device=input_device, dtype=input_dtype)


# ---------------------------------------------------------------------------
# Phase-Aware Multi-Resolution STFT Loss
# ---------------------------------------------------------------------------


def _wrap_phase(phase: torch.Tensor) -> torch.Tensor:
    """Wrap phase values to [-pi, pi]."""
    return (phase + math.pi) % (2.0 * math.pi) - math.pi


class PhaseAwareSTFTLoss(nn.Module):
    """Multi-resolution STFT loss with magnitude and instantaneous frequency terms.

    Magnitude loss follows the standard formulation: L1 on linear magnitude plus
    L1 on log magnitude, averaged across resolutions.

    Phase loss uses instantaneous frequency (IF), defined as the time-derivative
    of the phase spectrum. IF is computed as torch.diff(angle(STFT), dim=-1) along
    the time axis, wrapped to [-pi, pi]. L1 between pred and target IF captures
    temporal alignment information that magnitude alone cannot.

    Args:
        fft_sizes: FFT sizes for each resolution.
        hop_sizes: Hop sizes for each resolution.
        win_lengths: Window lengths for each resolution.
        mag_weight: Weight for the magnitude loss component.
        phase_weight: Weight for the IF loss relative to magnitude loss.
    """

    def __init__(
        self,
        fft_sizes: tuple = (2048, 1024, 512, 256),
        hop_sizes: tuple = (512, 256, 128, 64),
        win_lengths: tuple = (2048, 1024, 512, 256),
        mag_weight: float = 1.0,
        phase_weight: float = 0.5,  # Weight for IF loss relative to magnitude
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths), (
            "fft_sizes, hop_sizes, and win_lengths must have the same length"
        )
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.mag_weight = mag_weight
        self.phase_weight = phase_weight

        # Register Hann windows as persistent=False buffers (not saved in state_dict)
        for w in win_lengths:
            self.register_buffer(f"window_{w}", torch.hann_window(w), persistent=False)

    def _stft(
        self, x: torch.Tensor, fft_size: int, hop_size: int, win_length: int
    ) -> torch.Tensor:
        """Compute complex STFT in float32.

        Args:
            x: [B, T] waveform.
            fft_size: FFT size.
            hop_size: Hop size.
            win_length: Window length.

        Returns:
            Complex STFT tensor [B, F, T_frames].
        """
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

    def _magnitude_loss(
        self, pred_stft: torch.Tensor, target_stft: torch.Tensor
    ) -> torch.Tensor:
        """L1 on magnitude + L1 on log-magnitude."""
        pred_mag = pred_stft.abs()
        target_mag = target_stft.abs()

        lin_loss = F.l1_loss(pred_mag, target_mag)
        log_loss = F.l1_loss(
            torch.log(pred_mag + 1e-5),
            torch.log(target_mag + 1e-5),
        )
        return lin_loss + log_loss

    def _if_loss(
        self, pred_stft: torch.Tensor, target_stft: torch.Tensor
    ) -> torch.Tensor:
        """L1 on instantaneous frequency (phase derivative along time)."""
        pred_phase = torch.angle(pred_stft)    # [B, F, T_frames]
        target_phase = torch.angle(target_stft)

        # Instantaneous frequency: difference of phase along time axis
        pred_if = torch.diff(pred_phase, dim=-1)    # [B, F, T_frames - 1]
        target_if = torch.diff(target_phase, dim=-1)

        # Wrap to [-pi, pi] before computing L1
        pred_if = _wrap_phase(pred_if)
        target_if = _wrap_phase(target_if)

        return F.l1_loss(pred_if, target_if)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute multi-resolution phase-aware STFT loss.

        Args:
            pred: [B, 1, T] reconstructed waveform (any dtype).
            target: [B, 1, T] original waveform (any dtype).

        Returns:
            Scalar loss = mean over resolutions of (mag_weight * mag_loss + phase_weight * if_loss).
        """
        input_dtype = pred.dtype

        # [B, 1, T] -> [B, T]
        pred_flat = pred.squeeze(1)
        target_flat = target.squeeze(1)

        mag_total = torch.tensor(0.0, device=pred.device, dtype=torch.float32)
        phase_total = torch.tensor(0.0, device=pred.device, dtype=torch.float32)

        for fft_size, hop_size, win_length in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths
        ):
            pred_stft = self._stft(pred_flat, fft_size, hop_size, win_length)
            target_stft = self._stft(target_flat, fft_size, hop_size, win_length)

            mag_total = mag_total + self._magnitude_loss(pred_stft, target_stft)
            phase_total = phase_total + self._if_loss(pred_stft, target_stft)

        n = len(self.fft_sizes)
        loss = self.mag_weight * (mag_total / n) + self.phase_weight * (phase_total / n)

        return loss.to(dtype=input_dtype)
