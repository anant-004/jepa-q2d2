"""Quick UTMOS evaluation on a Stage 2 checkpoint.

Usage:
    python -m koe.fast.eval_utmos \
        --checkpoint /checkpoints/tokenizer_v9_ll/stage2_step66000.pt \
        --data_dir /data/librilight_9k \
        --n_samples 20
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--strides", type=str, default="4,4,4,5,6")
    parser.add_argument("--n_res_blocks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    strides = [int(s) for s in args.strides.split(",")]
    hop_length = 1
    for s in strides:
        hop_length *= s
    sr = 24000

    print(f"[UTMOS Eval] Loading checkpoint: {args.checkpoint}")
    print(f"[UTMOS Eval] Strides: {strides}, hop: {hop_length}")

    # Load model
    from koe.codec_impl import WaveformJEPAFSQVAE
    from koe.config import CodecConfig

    cfg = CodecConfig(
        strides=strides,
        n_res_blocks=args.n_res_blocks,
        fsq_levels=[8, 8, 8, 8],
    )

    model = WaveformJEPAFSQVAE(
        sample_rate=sr,
        code_dim=128,
        channels=cfg.channels,
        strides=strides,
        n_res_blocks=args.n_res_blocks,
        n_conformer=cfg.n_conformer,
        conformer_heads=cfg.conformer_heads,
        fsq_levels=cfg.fsq_levels,
        hifi_kernels=cfg.hifi_kernels,
        use_decoder_gaatn=cfg.use_decoder_gaatn,
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
    if isinstance(sd, dict) and "state_dict" not in sd and "model" not in sd:
        # It might be the state_dict itself
        pass
    model.load_state_dict(sd, strict=False)
    model.eval()
    model = model.to(device, dtype=torch.bfloat16)
    print(f"[UTMOS Eval] Model loaded, step {ckpt.get('step', '?')}")

    # Load UTMOS predictor
    print("[UTMOS Eval] Loading UTMOS model via torch.hub...")
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    utmos.eval()
    print("[UTMOS Eval] UTMOS model loaded")

    # Also load PESQ/STOI for comparison
    from pesq import pesq
    from pystoi import stoi
    import librosa

    # Find audio files
    data_path = Path(args.data_dir)
    files = sorted(data_path.rglob("*.flac"))
    if not files:
        files = sorted(data_path.rglob("*.wav"))
    print(f"[UTMOS Eval] Found {len(files)} audio files")

    # Sample random files
    import random
    random.seed(args.seed)
    sample_files = random.sample(files, min(args.n_samples, len(files)))

    utmos_scores = []
    pesq_scores = []
    stoi_scores = []

    for i, fpath in enumerate(sample_files):
        try:
            import torchaudio
            wav, file_sr = torchaudio.load(str(fpath))
            if file_sr != sr:
                wav = torchaudio.functional.resample(wav, file_sr, sr)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            wav = wav.squeeze(0)

            # Crop to max 10s
            max_samples = sr * 10
            if wav.shape[0] > max_samples:
                start = random.randint(0, wav.shape[0] - max_samples)
                wav = wav[start:start + max_samples]

            # Align to hop length
            rem = wav.shape[0] % hop_length
            if rem:
                wav = F.pad(wav, (0, hop_length - rem))

            # Roundtrip through codec
            wav_in = wav.unsqueeze(0).unsqueeze(0).to(device, dtype=torch.bfloat16)
            with torch.no_grad():
                rec, indices, aux_loss, z_e = model(wav_in)
                wav_out = rec

            orig_np = wav.numpy()
            rec_np = wav_out[0, 0].float().cpu().numpy()

            # Match lengths
            min_len = min(len(orig_np), len(rec_np))
            orig_np = orig_np[:min_len]
            rec_np = rec_np[:min_len]

            # UTMOS (on reconstructed only — it predicts MOS of the output)
            rec_tensor = torch.from_numpy(rec_np).unsqueeze(0).float()
            with torch.no_grad():
                utmos_score = utmos(rec_tensor, sr).item()
            utmos_scores.append(utmos_score)

            # Also get UTMOS of original for reference
            orig_tensor = torch.from_numpy(orig_np).unsqueeze(0).float()
            with torch.no_grad():
                utmos_orig = utmos(orig_tensor, sr).item()

            # PESQ
            orig_16k = librosa.resample(orig_np, orig_sr=sr, target_sr=16000)
            rec_16k = librosa.resample(rec_np, orig_sr=sr, target_sr=16000)
            min_16k = min(len(orig_16k), len(rec_16k))
            try:
                pesq_score = pesq(16000, orig_16k[:min_16k], rec_16k[:min_16k], "wb")
                pesq_scores.append(pesq_score)
            except Exception:
                pesq_score = float("nan")

            # STOI
            try:
                stoi_score = stoi(orig_np, rec_np, sr, extended=True)
                stoi_scores.append(stoi_score)
            except Exception:
                stoi_score = float("nan")

            print(f"  [{i+1}/{len(sample_files)}] UTMOS={utmos_score:.3f} (orig={utmos_orig:.3f}) | "
                  f"PESQ={pesq_score:.3f} | STOI={stoi_score:.3f} | {fpath.name}")

        except Exception as e:
            print(f"  [{i+1}/{len(sample_files)}] ERROR: {e} | {fpath.name}")

    print(f"\n{'='*60}")
    print(f"[UTMOS Eval] Results on {len(utmos_scores)} samples:")
    print(f"  UTMOS:  {np.mean(utmos_scores):.3f} ± {np.std(utmos_scores):.3f}  (range: {np.min(utmos_scores):.3f} - {np.max(utmos_scores):.3f})")
    if pesq_scores:
        print(f"  PESQ:   {np.mean(pesq_scores):.3f} ± {np.std(pesq_scores):.3f}")
    if stoi_scores:
        print(f"  STOI:   {np.mean(stoi_scores):.3f} ± {np.std(stoi_scores):.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
