"""Extract real encoder feature pairs for Paper A hero figure panel (b).

For each of the two matched codecs, run audio through the model and capture
`pairs_rot` -- the post-normalize, post-affine 2D pairs that the Q2D2 lattice
actually snaps to (see koe/fast/q2d2.py: _apply_affine output, step 3 of forward).
We capture by wrapping Q2D2Quantizer._apply_affine, so we do not depend on
V2Codec internals.

Matched pair (verified 2026-05-14):
  SIGReg ON  = q2d2_sigreg/v2_latest.pt          (encoder sigreg_200k, lambda=0.05)
  SIGReg OFF = q2d2_ema200k_control/v2_latest.pt (encoder ema_200k,    lambda=0.0)
  Both: 25 Hz, code_dim 32, Q2D2 K=4 rhombic, 100k steps.

All P=16 pairs share the same lattice, so we pool pairs across all positions.

Run on the VM:
  CUDA_VISIBLE_DEVICES=0 python scripts/extract_sigreg_feature_dist.py \
      --data_dir /local_data/data/librilight/librilight_9k \
      --out /tmp/sigreg_feature_dist.npz
"""
import argparse
import glob
import os
import random
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.expanduser("~/koe"))
SAMPLE_RATE = 24000

MODELS = {
    "on":  "/local_data/checkpoints/q2d2_sigreg/v2_latest.pt",
    "off": "/local_data/checkpoints/q2d2_ema200k_control/v2_latest.pt",
}


def load_v2(ckpt_path, device):
    """Load a V2Codec. Mirrors scripts/run_wer_eval.py load_v2_model."""
    from koe.codec_impl import JEPAEncoder
    from koe.fast.train_v2_stage2 import V2Codec
    from koe.fast.q2d2 import Q2D2Quantizer

    import time
    t0 = time.time()
    print(f"  torch.load({os.path.basename(ckpt_path)})...", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # drop the heavy optimizer/disc states we don't need (keeps RAM low)
    state_dict = ckpt["state_dict"]
    cfg = ckpt.get("config", {})
    del ckpt
    print(f"  torch.load done in {time.time()-t0:.1f}s", flush=True)
    strides = cfg["strides"]
    code_dim = cfg["code_dim"]
    qtype = cfg["quantizer"]
    assert qtype == "q2d2", f"expected q2d2, got {qtype}"

    print("  building JEPAEncoder...", flush=True)
    encoder = JEPAEncoder(
        sample_rate=SAMPLE_RATE, code_dim=128,
        channels=[64, 128, 256, 384, 512, 512],
        strides=strides, n_res_blocks=8,
        n_conformer=8, conformer_heads=16, use_gaatn=True,
    )
    print(f"  building V2Codec... ({time.time()-t0:.1f}s elapsed)", flush=True)
    model = V2Codec(encoder=encoder, code_dim=code_dim, strides=strides)
    model.quantizer = Q2D2Quantizer(
        dim=code_dim, num_levels=4, grid_type="rhombic", commitment_weight=0.25,
    )
    print(f"  load_state_dict... ({time.time()-t0:.1f}s elapsed)", flush=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  state_dict loaded  cd={code_dim}  "
          f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    assert len(missing) <= 5 and len(unexpected) <= 5, "suspicious load"
    model = model.to(device, dtype=torch.float32).eval()
    print(f"  model on {next(model.parameters()).device}", flush=True)
    return model, int(np.prod(strides))


def load_audio(data_dir, n_files, max_seconds=8.0, seed=42):
    """Mirrors scripts/run_wer_eval.py load_eval_samples: resamples on sr
    mismatch (LibriLight is not 24 kHz), skips unreadable files, early-exits."""
    files = []
    for ext in ("*.flac", "*.wav"):
        files.extend(sorted(glob.glob(os.path.join(data_dir, "**", ext), recursive=True)))
    random.Random(seed).shuffle(files)
    wavs, maxs = [], int(max_seconds * SAMPLE_RATE)
    for f in files:
        if len(wavs) >= n_files:
            break
        try:
            data, sr = sf.read(f, dtype="float32")
            wav = data if data.ndim == 1 else data.mean(axis=1)
            if sr != SAMPLE_RATE:
                import torchaudio
                wav = torchaudio.functional.resample(
                    torch.from_numpy(wav).unsqueeze(0), sr, SAMPLE_RATE
                ).squeeze(0).numpy()
            if len(wav) < SAMPLE_RATE:
                continue
            wavs.append(wav[:maxs])
        except Exception:
            continue
    print(f"  loaded {len(wavs)} audio files", flush=True)
    return wavs


def capture_pairs(model, hop, wavs, device):
    """Run audio, capture pairs_rot from Q2D2._apply_affine. Returns (M,2)."""
    captured = []
    q = model.quantizer
    orig_fn = q._apply_affine

    def wrapped(pairs):
        out = orig_fn(pairs)
        captured.append(out.detach().reshape(-1, 2).float().cpu())
        return out

    q._apply_affine = wrapped
    with torch.no_grad():
        for wav_np in wavs:
            wav = torch.from_numpy(wav_np)
            rem = wav.shape[0] % hop
            if rem:
                wav = torch.nn.functional.pad(wav, (0, hop - rem))
            wav_in = wav.unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
            model(wav_in)
    q._apply_affine = orig_fn
    return torch.cat(captured, dim=0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/local_data/data/librilight/librilight_9k")
    ap.add_argument("--out", default="/tmp/sigreg_feature_dist.npz")
    ap.add_argument("--n_files", type=int, default=12)
    ap.add_argument("--n_plot", type=int, default=6000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] cuda_available={torch.cuda.is_available()} -> using {device}",
          flush=True)
    print("[load_audio] starting...", flush=True)
    wavs = load_audio(args.data_dir, args.n_files)

    out = {}
    grid = None
    for tag, ckpt in MODELS.items():
        print(f"[{tag}] {ckpt}")
        model, hop = load_v2(ckpt, device)
        if grid is None:
            grid = model.quantizer.grid.detach().cpu().numpy()  # (K^2, 2)
        pts = capture_pairs(model, hop, wavs, device)
        print(f"  captured {pts.shape[0]} pair-points; "
              f"x range [{pts[:,0].min():.2f},{pts[:,0].max():.2f}] "
              f"y range [{pts[:,1].min():.2f},{pts[:,1].max():.2f}]")
        # subsample for a clean scatter
        if pts.shape[0] > args.n_plot:
            sel = np.random.RandomState(0).choice(pts.shape[0], args.n_plot, replace=False)
            pts = pts[sel]
        out[f"{tag}_xy"] = pts
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # axis limit: cover both clouds and the grid, with a small margin
    allpts = np.concatenate([out["on_xy"], out["off_xy"], grid], axis=0)
    lim = float(np.percentile(np.abs(allpts), 99.5)) * 1.15
    lim = max(lim, float(np.abs(grid).max()) * 1.25)

    np.savez(args.out, on_xy=out["on_xy"], off_xy=out["off_xy"],
             grid=grid, lim=lim)
    print(f"saved {args.out}")
    print(f"  on_xy={out['on_xy'].shape}  off_xy={out['off_xy'].shape}  "
          f"grid={grid.shape}  lim={lim:.3f}")


if __name__ == "__main__":
    main()
