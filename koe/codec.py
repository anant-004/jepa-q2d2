"""Clean codec wrapper — the public API for encoding/decoding audio.

Usage:
    codec = KoeCodec.from_pretrained("path/to/checkpoint.pt")

    # Encode: waveform → packed token IDs
    tokens = codec.encode(wav)          # [B, T_z, 19]

    # Decode: packed token IDs → waveform
    wav = codec.decode(tokens)          # [B, 1, T_wav]

    # File helpers
    tokens = codec.encode_file("speech.wav")
    codec.decode_to_file(tokens, "output.wav")

All internals (FSQ, packing, encoder/decoder networks) are hidden.
"""

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torchaudio

from koe.config import CodecConfig
from koe.codec_impl import (
    WaveformJEPAFSQVAE,
    JEPAEncoder,
    fsq_pack_indices,
    fsq_unpack_indices,
)


class KoeCodec(nn.Module):
    """Encode audio → packed FSQ tokens, decode tokens → audio.

    Wraps WaveformJEPAFSQVAE with a simple two-method API.
    The underlying model has:
      - Encoder: waveform → continuous latents [B, 128, T_z]
      - FSQ: continuous → discrete indices [B, T_z, 128], each in {0,1,2,3}
      - Packing: 128 indices → 19 packed tokens [B, T_z, 19], each in [0, 16383]
      - Decoder: quantized latents → waveform [B, 1, T_wav]
    """

    def __init__(self, config: CodecConfig):
        super().__init__()
        self.config = config

        # Build encoder
        # channels list needs input channel prepended for the encoder
        # CodecConfig.channels = [64, 128, 256, 384, 512] — these are the
        # encoder block output channels. JEPAEncoder expects len(channels) == len(strides) + 1
        # where channels[0] is the input_conv output.
        encoder = JEPAEncoder(
            sample_rate=config.sample_rate,
            code_dim=config.code_dim,
            channels=config.channels,
            strides=config.strides,
            n_res_blocks=config.n_res_blocks,
            n_conformer=config.n_conformer,
            conformer_heads=config.conformer_heads,
            use_gaatn=True,
        )

        # Build full model (encoder + FSQ + decoder)
        self.model = WaveformJEPAFSQVAE(
            jepa_encoder=encoder,
            fsq_levels=config.fsq_levels,
            channels=config.channels,
            strides=config.strides,
            use_tanh=False,
            hifi_kernels=config.hifi_kernels,
            use_decoder_gaatn=config.use_decoder_gaatn,
            freeze_encoder=True,  # always frozen for inference
        )

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """Encode waveform to packed token IDs.

        Args:
            wav: [B, 1, T_wav] mono audio at self.config.sample_rate

        Returns:
            tokens: [B, T_z, 19] packed FSQ tokens, each in [0, 16383]
        """
        z_q, z_e, indices, aux_loss = self.model.encode(wav)
        # indices: [B, T_z, 128], values in {0,1,2,3}
        packed = fsq_pack_indices(
            indices,
            levels=self.config.fsq_levels,
            group_size=self.config.group_size,
        )
        # packed: [B, T_z, 19], values in [0, 16383]
        return packed

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode packed token IDs to waveform.

        Args:
            tokens: [B, T_z, 19] packed FSQ tokens

        Returns:
            wav: [B, 1, T_wav] reconstructed audio
        """
        # Unpack: [B, T_z, 19] → [B, T_z, 128]
        indices = fsq_unpack_indices(
            tokens,
            levels=self.config.fsq_levels,
            code_dim=self.config.code_dim,
            group_size=self.config.group_size,
        )
        # Dequantize: indices → boundary values [B, 128, T_z]
        z_q = self.model.fsq.dequantize(indices)
        # Decode: [B, 128, T_z] → [B, 1, T_wav]
        wav = self.model.decode(z_q)
        return wav

    @torch.no_grad()
    def encode_file(
        self,
        path: Union[str, Path],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Load an audio file and encode to tokens.

        Handles resampling to the codec's sample rate and mono conversion.

        Returns:
            tokens: [1, T_z, 19] packed FSQ tokens
        """
        if device is None:
            device = next(self.parameters()).device

        wav, sr = torchaudio.load(str(path))
        # Convert to mono
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        # Resample if needed
        if sr != self.config.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.config.sample_rate)
        # Add batch dim: [1, T] → [1, 1, T]
        wav = wav.unsqueeze(0).to(device)
        return self.encode(wav)

    @torch.no_grad()
    def decode_to_file(
        self,
        tokens: torch.Tensor,
        path: Union[str, Path],
    ) -> None:
        """Decode tokens and save to audio file."""
        wav = self.decode(tokens)
        # wav: [B, 1, T] → take first sample, squeeze channel
        torchaudio.save(
            str(path),
            wav[0].cpu(),  # [1, T]
            self.config.sample_rate,
        )

    @torch.no_grad()
    def roundtrip(self, wav: torch.Tensor) -> torch.Tensor:
        """Encode then decode — useful for evaluating codec quality.

        Args:
            wav: [B, 1, T_wav]
        Returns:
            reconstructed: [B, 1, T_wav] (length-matched to input)
        """
        tokens = self.encode(wav)
        rec = self.decode(tokens)
        # Match output length to input
        T = wav.shape[-1]
        if rec.shape[-1] > T:
            rec = rec[..., :T]
        elif rec.shape[-1] < T:
            rec = torch.nn.functional.pad(rec, (0, T - rec.shape[-1]))
        return rec

    def save(self, path: Union[str, Path]) -> None:
        """Save codec checkpoint."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
            },
            str(path),
        )

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
    ) -> "KoeCodec":
        """Load codec from checkpoint.

        Args:
            path: path to checkpoint .pt file
            device: device to load onto

        Returns:
            KoeCodec instance with loaded weights
        """
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
        config = ckpt["config"]
        codec = cls(config)
        codec.model.load_state_dict(ckpt["model_state_dict"])
        codec.eval()
        codec.to(device)
        return codec
