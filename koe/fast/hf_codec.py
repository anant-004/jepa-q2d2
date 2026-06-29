"""Load JEPA-Q2D2 codecs directly from the HuggingFace Hub.

Public weights: https://huggingface.co/Andy004/jepa-q2d2

Three released models (pick by short name or full subfolder):
    "main"    -> jepa-q2d2-cd64-12.5hz       (Q2D2, 1.6 kbps, 100 tok/s)   PESQ 2.53
    "sigreg"  -> jepa-q2d2-sigreg-cd32-25hz  (Q2D2 + SIGReg, 1.6 kbps)     ESTOI 0.79
    "teacher" -> teacher-cd128-fsq-12.5hz    (FSQ, ~2.85 kbps, 237.5 tok/s) PESQ ~2.91

Usage:
    from koe.fast.hf_codec import load_codec_from_hf
    model, info = load_codec_from_hf("teacher", device="cuda")
    # model.encode(wav) -> (z_q, z_pre, indices, aux);  model.decode(z_q) -> wav
    # model.encoder.encode(wav) -> [B, 128, T]  (pre-quantization features)

Requires the `koe` package to be importable (the model code lives here) and
`huggingface_hub`. No HF token is needed for these public weights.
"""
import json
import math
from typing import Dict, Tuple

import torch

HF_REPO = "Andy004/jepa-q2d2"
MODELS = {
    "main": "jepa-q2d2-cd64-12.5hz",
    "sigreg": "jepa-q2d2-sigreg-cd32-25hz",
    "teacher": "teacher-cd128-fsq-12.5hz",
}
CHANNELS = [64, 128, 256, 384, 512, 512]


def _build_model(cfg: Dict):
    """Construct the right architecture from a release config.json."""
    strides = [int(s) for s in cfg.get("strides", [4, 4, 4, 5, 6])]
    code_dim = int(cfg.get("code_dim", 128))
    quantizer = cfg.get("quantizer", "fsq")

    if quantizer == "q2d2":
        from koe.codec_impl import JEPAEncoder
        from koe.fast.train_v2_stage2 import V2Codec
        from koe.fast.q2d2 import Q2D2Quantizer

        encoder = JEPAEncoder(
            sample_rate=24000, code_dim=128, channels=CHANNELS,
            strides=strides, n_res_blocks=8, n_conformer=8,
            conformer_heads=16, use_gaatn=True,
        )
        model = V2Codec(
            encoder=encoder, code_dim=code_dim, decoder_type="hifigan",
            channels=CHANNELS, strides=strides,
        )
        # K=4 rhombic / commit 0.25 is the project-wide Q2D2 setting (cd64 + cd32).
        model.quantizer = Q2D2Quantizer(
            dim=code_dim, num_levels=4, grid_type="rhombic", commitment_weight=0.25,
        )
    else:  # finite scalar quantization (cd128 teacher)
        from koe.codec_impl import WaveformJEPAFSQVAE

        fsq_levels = [int(x) for x in cfg.get("fsq_levels", [8, 8, 8, 8])]
        model = WaveformJEPAFSQVAE(
            sample_rate=24000, code_dim=code_dim, channels=CHANNELS,
            strides=strides, n_res_blocks=int(cfg.get("n_res_blocks", 8)),
            n_conformer=8, conformer_heads=16,
            fsq_levels=fsq_levels, hifi_kernels=[3, 7, 11, 15, 23, 32],
        )
    return model, strides, code_dim, quantizer


def load_codec_from_hf(
    model: str = "teacher",
    repo_id: str = HF_REPO,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.nn.Module, Dict]:
    """Download a released checkpoint + config from HF and build a ready model.

    Returns (model, info). `info` carries hop, frame_rate, code_dim, quantizer,
    tokens_per_second, bitrate_kbps, and the load report (missing/unexpected key
    counts — both should be ~0 for a clean load).
    """
    from huggingface_hub import hf_hub_download

    subfolder = MODELS.get(model, model)
    cfg_path = hf_hub_download(repo_id, f"{subfolder}/config.json")
    pt_path = hf_hub_download(repo_id, f"{subfolder}/pytorch_model.pt")

    with open(cfg_path) as f:
        cfg = json.load(f)
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    net, strides, code_dim, quantizer = _build_model(cfg)
    incompatible = net.load_state_dict(state_dict, strict=False)
    net.eval().to(device, dtype=dtype)

    hop = math.prod(strides)
    info = {
        "subfolder": subfolder,
        "hop": hop,
        "frame_rate_hz": 24000 / hop,
        "code_dim": code_dim,
        "quantizer": quantizer,
        "tokens_per_second": cfg.get("tokens_per_second"),
        "bitrate_kbps": cfg.get("bitrate_kbps"),
        "step": cfg.get("step"),
        "metrics": cfg.get("metrics"),
        "missing_keys": len(incompatible.missing_keys),
        "unexpected_keys": len(incompatible.unexpected_keys),
    }
    return net, info
