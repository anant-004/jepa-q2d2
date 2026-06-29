"""Reconstruct audio with a JEPA-Q2D2 codec loaded from HuggingFace.

Defaults to the 237.5 tok/s FSQ teacher (cd128, 12.5 Hz); use --model to pick
any released codec. Weights: https://huggingface.co/Andy004/jepa-q2d2

Usage:
    python scripts/infer_from_hf.py --input in.wav --output recon.wav
    python scripts/infer_from_hf.py --model main   --input in.wav --output recon_cd64.wav
    python scripts/infer_from_hf.py --model sigreg  --input in.wav --output recon_cd32.wav

Run from the repo root so the `koe` package is importable.
"""
import argparse

import numpy as np
import soundfile as sf
import torch
import torchaudio

from koe.fast.hf_codec import load_codec_from_hf, MODELS

SAMPLE_RATE = 24000


def load_wav_24k(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if sr != SAMPLE_RATE:
        t = torch.from_numpy(wav).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, SAMPLE_RATE)
        wav = t.squeeze(0).numpy()
    return np.ascontiguousarray(wav)


@torch.no_grad()
def reconstruct(model, wav_24k: np.ndarray, hop: int, device: str, dtype) -> np.ndarray:
    n = len(wav_24k)
    rem = n % hop
    if rem:
        wav_24k = np.pad(wav_24k, (0, hop - rem))
    x = torch.from_numpy(wav_24k).view(1, 1, -1).to(device, dtype=dtype)
    z_q = model.encode(x)[0]        # encode -> (z_q, ...)
    rec = model.decode(z_q)         # decode -> [B, 1, T_wav]
    if isinstance(rec, (tuple, list)):
        rec = rec[0]
    return rec.view(-1).float().cpu().numpy()[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="teacher",
                    help=f"one of {list(MODELS)} or a repo subfolder name")
    ap.add_argument("--input", required=True, help="input wav/flac")
    ap.add_argument("--output", required=True, help="output wav path")
    ap.add_argument("--repo_id", default="Andy004/jepa-q2d2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    print(f"Loading '{args.model}' from {args.repo_id} on {args.device} ...")
    model, info = load_codec_from_hf(args.model, repo_id=args.repo_id,
                                     device=args.device, dtype=dtype)
    print(f"  {info['subfolder']}: {info['frame_rate_hz']:.1f} Hz, "
          f"{info['tokens_per_second']} tok/s, {info['bitrate_kbps']} kbps, "
          f"quantizer={info['quantizer']}")
    print(f"  load report: missing={info['missing_keys']} unexpected={info['unexpected_keys']}")

    wav = load_wav_24k(args.input)
    rec = reconstruct(model, wav, info["hop"], args.device, dtype)
    sf.write(args.output, rec, SAMPLE_RATE)
    print(f"Wrote {args.output}  ({len(rec) / SAMPLE_RATE:.2f}s @ {SAMPLE_RATE} Hz)")


if __name__ == "__main__":
    main()
