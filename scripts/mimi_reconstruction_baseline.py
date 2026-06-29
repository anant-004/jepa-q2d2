"""Run Mimi codec reconstruction baseline on our eval set.

Uses the SAME eval set as run_evals.py for fair PESQ/STOI comparison.
Mimi runs at 24 kHz, 12.5 Hz frame rate, 1.1 kbps (8 codebooks × 12.5 Hz × 11 bits).

Usage:
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/mimi_reconstruction_baseline.py \
        --data_dir /local_data/data/librilight/librilight_9k \
        --output_dir /local_data/eval_results/mimi_baseline \
        --n_samples 50
"""

import argparse
import glob
import json
import math
import os
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.expanduser("~/koe"))

SAMPLE_RATE = 24000


def load_eval_samples(data_dir, n_samples, max_seconds=10.0, seed=42):
    """Load fixed eval set — matches run_evals.py exactly."""
    files = []
    for ext in ["*.flac", "*.wav"]:
        files.extend(sorted(glob.glob(os.path.join(data_dir, "**", ext), recursive=True)))

    rng = random.Random(seed)
    rng.shuffle(files)

    wavs = []
    max_samples = int(max_seconds * SAMPLE_RATE)
    for f in files:
        if len(wavs) >= n_samples:
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
            wavs.append(wav[:max_samples])
        except Exception:
            continue

    print(f"[data] Loaded {len(wavs)} eval samples")
    return wavs


def load_mimi(device):
    """Load Mimi from Kyutai's moshi library."""
    from moshi.models import loaders

    weight_path = loaders.hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    print(f"[mimi] Loaded from {weight_path}")
    mimi = loaders.get_mimi(weight_path, device=device)
    mimi.eval()
    # Set num_codebooks for the default codebook configuration
    if hasattr(mimi, "set_num_codebooks"):
        mimi.set_num_codebooks(8)  # default Mimi = 8 codebooks
    print(f"[mimi] Frame rate: {mimi.frame_rate} Hz, sample rate: {mimi.sample_rate} Hz")
    return mimi


def reconstruct_mimi(mimi, wavs, device):
    """Encode → decode each waveform through Mimi."""
    recs = []
    hop = int(SAMPLE_RATE / mimi.frame_rate)  # 1920 at 12.5 Hz
    with torch.no_grad():
        for i, wav_np in enumerate(wavs):
            wav = torch.from_numpy(wav_np).float()
            orig_len = wav.shape[0]
            rem = orig_len % hop
            if rem:
                wav = torch.nn.functional.pad(wav, (0, hop - rem))
            wav_in = wav.unsqueeze(0).unsqueeze(0).to(device)

            # Mimi encode → returns codes (tokens) [B, K, T]
            codes = mimi.encode(wav_in)
            # Mimi decode → returns reconstructed waveform
            rec = mimi.decode(codes)
            rec_np = rec[0, 0, :orig_len].float().cpu().numpy()
            recs.append(rec_np)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(wavs)}")
    return recs


def compute_metrics(origs, recs):
    """Compute PESQ, STOI, mel L1 per sample, then average."""
    import librosa
    from pesq import pesq
    from pystoi import stoi

    sr = SAMPLE_RATE
    pesq_scores, stoi_scores, mel_l1_scores = [], [], []

    for orig, rec in zip(origs, recs):
        n = min(len(orig), len(rec))
        o = orig[:n].astype(np.float64)
        r = rec[:n].astype(np.float64)

        # PESQ at 16kHz
        try:
            o16 = librosa.resample(o, orig_sr=sr, target_sr=16000)
            r16 = librosa.resample(r, orig_sr=sr, target_sr=16000)
            p = pesq(16000, o16, r16, "wb")
            pesq_scores.append(p)
        except Exception as e:
            print(f"  PESQ fail: {e}")

        # STOI at 16kHz
        try:
            s = stoi(o16, r16, 16000, extended=False)
            stoi_scores.append(s)
        except Exception:
            pass

        # Mel L1 at 24kHz
        try:
            mel_o = librosa.feature.melspectrogram(y=o.astype(np.float32), sr=sr,
                                                   n_mels=80, n_fft=1024, hop_length=256)
            mel_r = librosa.feature.melspectrogram(y=r.astype(np.float32), sr=sr,
                                                   n_mels=80, n_fft=1024, hop_length=256)
            mel_o = np.log(np.clip(mel_o, 1e-5, None))
            mel_r = np.log(np.clip(mel_r, 1e-5, None))
            min_t = min(mel_o.shape[1], mel_r.shape[1])
            mel_l1 = float(np.abs(mel_o[:, :min_t] - mel_r[:, :min_t]).mean())
            mel_l1_scores.append(mel_l1)
        except Exception as e:
            print(f"  Mel fail: {e}")

    return {
        "pesq": {"mean": float(np.mean(pesq_scores)), "std": float(np.std(pesq_scores)),
                 "n": len(pesq_scores)},
        "stoi": {"mean": float(np.mean(stoi_scores)), "std": float(np.std(stoi_scores)),
                 "n": len(stoi_scores)},
        "mel_l1": {"mean": float(np.mean(mel_l1_scores)), "std": float(np.std(mel_l1_scores)),
                   "n": len(mel_l1_scores)},
    }


def compute_utmos(recs, device):
    """Compute UTMOS scores via speechmos."""
    try:
        from speechmos import utmos
    except ImportError:
        print("[utmos] speechmos not installed, skipping")
        return None
    import librosa
    scores = []
    for r in recs:
        try:
            r16 = librosa.resample(r.astype(np.float32), orig_sr=SAMPLE_RATE, target_sr=16000)
            res = utmos.run(r16, 16000)
            scores.append(float(res["utmos"]))
        except Exception as e:
            print(f"  UTMOS fail: {e}")
    if not scores:
        return None
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "n": len(scores)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/local_data/data/librilight/librilight_9k")
    parser.add_argument("--output_dir", default="/local_data/eval_results/mimi_baseline")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save_audio", action="store_true",
                        help="Save reconstructed audio for listening")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t_start = time.time()

    print("=== Mimi Reconstruction Baseline ===")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Eval samples: {args.n_samples}")

    # Load eval samples
    origs = load_eval_samples(args.data_dir, args.n_samples)

    # Load Mimi
    mimi = load_mimi(args.device)

    # Mimi config + bitrate calculation
    n_codebooks = 8
    frame_rate = mimi.frame_rate  # 12.5
    # Mimi codebook size is 2048 (11 bits per codebook)
    bits_per_code = 11
    bitrate = n_codebooks * bits_per_code * frame_rate
    print(f"[mimi] Bitrate: {bitrate} bps ({bitrate/1000:.2f} kbps) "
          f"= {n_codebooks} codebooks × {bits_per_code} bits × {frame_rate} Hz")

    # Reconstruct
    print("\n[recon] Encoding + decoding...")
    t0 = time.time()
    recs = reconstruct_mimi(mimi, origs, args.device)
    recon_time = time.time() - t0
    print(f"[recon] Done in {recon_time:.1f}s ({recon_time/len(origs):.2f}s/sample)")

    # Save audio samples
    if args.save_audio:
        audio_dir = os.path.join(args.output_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        for i in range(min(10, len(origs))):
            sf.write(os.path.join(audio_dir, f"orig_{i:02d}.wav"), origs[i], SAMPLE_RATE)
            sf.write(os.path.join(audio_dir, f"mimi_recon_{i:02d}.wav"), recs[i], SAMPLE_RATE)
        print(f"[audio] Saved 10 sample pairs to {audio_dir}")

    # Compute metrics
    print("\n[metrics] Computing PESQ, STOI, mel L1...")
    metrics = compute_metrics(origs, recs)
    print(f"  PESQ: {metrics['pesq']['mean']:.3f} ± {metrics['pesq']['std']:.3f}")
    print(f"  STOI: {metrics['stoi']['mean']:.3f} ± {metrics['stoi']['std']:.3f}")
    print(f"  Mel L1: {metrics['mel_l1']['mean']:.3f} ± {metrics['mel_l1']['std']:.3f}")

    # UTMOS
    print("\n[metrics] Computing UTMOS...")
    utmos_scores = compute_utmos(recs, args.device)
    if utmos_scores:
        print(f"  UTMOS: {utmos_scores['mean']:.3f} ± {utmos_scores['std']:.3f}")
        metrics["utmos"] = utmos_scores

    metrics["config"] = {
        "n_samples": len(origs),
        "n_codebooks": n_codebooks,
        "frame_rate": frame_rate,
        "bitrate_bps": bitrate,
        "bitrate_kbps": bitrate / 1000,
        "sample_rate": SAMPLE_RATE,
    }

    # Save
    with open(os.path.join(args.output_dir, "mimi_baseline_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\nCOMPLETE in {elapsed:.1f}s")
    print(f"\n=== Comparison to our codec at 1.6 kbps ===")
    print(f"Mimi (1.1 kbps): PESQ {metrics['pesq']['mean']:.3f}, STOI {metrics['stoi']['mean']:.3f}")
    print(f"Q2D2 cd64 (1.6 kbps): PESQ 2.55, STOI 0.856")


if __name__ == "__main__":
    main()
