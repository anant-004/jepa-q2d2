"""Visualize JEPA encoder embeddings via t-SNE/UMAP clustering.

Encodes audio samples through the v9 encoder, extracts frame-level embeddings,
and plots them colored by speaker, phoneme content, or language.

Usage:
    python -m koe.fast.visualize_embeddings \
        --ckpt /path/to/v9_ll_stage2_400k.pt \
        --output_dir ./embedding_viz
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio


SAMPLE_RATE = 24000


def load_codec(ckpt_path: str, device: str = "cuda"):
    """Load v9 codec and return model."""
    from koe.codec_impl import WaveformJEPAFSQVAE

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    strides = cfg.get("strides", [4, 4, 4, 5, 6])
    fsq_levels = cfg.get("fsq_levels", [8, 8, 8, 8])

    model = WaveformJEPAFSQVAE(
        sample_rate=24000, code_dim=128,
        channels=[64, 128, 256, 384, 512, 512],
        strides=strides, n_res_blocks=8, n_conformer=8, conformer_heads=16,
        fsq_levels=fsq_levels, hifi_kernels=[3, 7, 11, 15, 23, 32],
    )
    model.load_state_dict(ckpt.get("state_dict", {}), strict=False)
    model.eval().to(device, dtype=torch.bfloat16)

    hop = 1
    for s in strides:
        hop *= s
    return model, hop


def extract_embeddings(model, wav_24k: np.ndarray, hop: int, device: str = "cuda"):
    """Extract frame-level encoder embeddings from audio."""
    # Pad to hop alignment
    rem = len(wav_24k) % hop
    if rem:
        wav_24k = np.pad(wav_24k, (0, hop - rem))

    wav_t = torch.from_numpy(wav_24k).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.bfloat16)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        z_e = model.encoder.encode(wav_t)  # [1, 128, T]

    return z_e[0].float().cpu().numpy()  # [128, T]


def load_samples_with_metadata(n_per_lang: int = 20):
    """Load samples from multiple sources with metadata."""
    from datasets import load_dataset
    from huggingface_hub import login
    login(token=os.environ.get("HF_TOKEN"))

    samples = []

    # English - LibriSpeech (has speaker IDs)
    print("[data] Loading English samples...")
    ds = load_dataset("librispeech_asr", split="test.clean", streaming=True,
                       token=os.environ.get("HF_TOKEN"))
    count = 0
    for item in ds:
        if count >= n_per_lang:
            break
        audio = item.get("audio", {})
        if "array" not in audio:
            continue
        wav = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        dur = len(wav) / sr
        if dur < 3.0 or dur > 10.0:
            continue
        samples.append({
            "wav": wav, "sr": sr,
            "language": "English",
            "speaker": str(item.get("speaker_id", f"en_{count}")),
            "text": item.get("text", ""),
        })
        count += 1
    print(f"  Got {count} English samples")

    # Chinese - FLEURS
    print("[data] Loading Chinese samples...")
    ds = load_dataset("google/fleurs", "cmn_hans_cn", split="test", streaming=True,
                       trust_remote_code=True, token=os.environ.get("HF_TOKEN"))
    count = 0
    for item in ds:
        if count >= n_per_lang:
            break
        audio = item.get("audio", {})
        if "array" not in audio:
            continue
        wav = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        dur = len(wav) / sr
        if dur < 3.0 or dur > 10.0:
            continue
        samples.append({
            "wav": wav, "sr": sr,
            "language": "Chinese",
            "speaker": f"zh_{count}",
            "text": item.get("transcription", ""),
        })
        count += 1
    print(f"  Got {count} Chinese samples")

    # Hindi - private dataset
    print("[data] Loading Hindi samples...")
    ds = load_dataset(os.environ.get("HINDI_DATASET", ""), split="train", streaming=True,
                       token=os.environ.get("HF_TOKEN"))
    count = 0
    for item in ds:
        if count >= n_per_lang:
            break
        audio = item.get("audio", {})
        if audio is None or "array" not in audio:
            continue
        wav = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        dur = len(wav) / sr
        if dur < 3.0 or dur > 10.0:
            continue
        samples.append({
            "wav": wav, "sr": sr,
            "language": "Hindi",
            "speaker": f"hi_{count}",
            "text": str(item.get("text", ""))[:100],
        })
        count += 1
    print(f"  Got {count} Hindi samples")

    # Japanese - FLEURS
    print("[data] Loading Japanese samples...")
    ds = load_dataset("google/fleurs", "ja_jp", split="test", streaming=True,
                       trust_remote_code=True, token=os.environ.get("HF_TOKEN"))
    count = 0
    for item in ds:
        if count >= n_per_lang:
            break
        audio = item.get("audio", {})
        if "array" not in audio:
            continue
        wav = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        dur = len(wav) / sr
        if dur < 3.0 or dur > 10.0:
            continue
        samples.append({
            "wav": wav, "sr": sr,
            "language": "Japanese",
            "speaker": f"ja_{count}",
            "text": item.get("transcription", ""),
        })
        count += 1
    print(f"  Got {count} Japanese samples")

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./embedding_viz")
    parser.add_argument("--n_per_lang", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load codec
    print("Loading codec...")
    model, hop = load_codec(args.ckpt, args.device)
    print(f"Loaded. Hop={hop}, frame_rate={24000/hop:.1f} Hz")

    # Load samples
    samples = load_samples_with_metadata(args.n_per_lang)
    print(f"\nTotal samples: {len(samples)}")

    # Extract embeddings
    print("\nExtracting embeddings...")
    all_embeddings = []  # [128, T] per sample → collect frame-level
    all_labels = []  # metadata per frame
    all_mean_embeddings = []  # mean pooled per utterance
    all_utterance_labels = []

    for i, s in enumerate(samples):
        wav = s["wav"]
        sr = s["sr"]

        # Resample to 24kHz
        if sr != SAMPLE_RATE:
            wav_t = torch.from_numpy(wav).unsqueeze(0)
            wav_t = torchaudio.functional.resample(wav_t, sr, SAMPLE_RATE)
            wav = wav_t.squeeze(0).numpy()

        if wav.ndim > 1:
            wav = wav.mean(axis=-1)

        z = extract_embeddings(model, wav, hop, args.device)  # [128, T]
        T = z.shape[1]

        # Frame-level embeddings (subsample to keep manageable)
        step = max(1, T // 10)  # ~10 frames per utterance
        for t in range(0, T, step):
            all_embeddings.append(z[:, t])
            all_labels.append({
                "language": s["language"],
                "speaker": s["speaker"],
                "frame_pos": t / T,  # relative position in utterance
            })

        # Utterance-level embedding (mean pool)
        all_mean_embeddings.append(z.mean(axis=1))
        all_utterance_labels.append({
            "language": s["language"],
            "speaker": s["speaker"],
            "text": s.get("text", "")[:50],
        })

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(samples)} processed")

    embeddings = np.stack(all_embeddings)  # [N_frames, 128]
    mean_embeddings = np.stack(all_mean_embeddings)  # [N_utterances, 128]

    print(f"\nFrame-level embeddings: {embeddings.shape}")
    print(f"Utterance-level embeddings: {mean_embeddings.shape}")

    # Dimensionality reduction
    print("\nRunning t-SNE...")
    from sklearn.manifold import TSNE

    # Frame-level t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    frame_2d = tsne.fit_transform(embeddings)

    # Utterance-level t-SNE
    tsne_utt = TSNE(n_components=2, perplexity=min(15, len(mean_embeddings) - 1),
                     random_state=42, max_iter=1000)
    utt_2d = tsne_utt.fit_transform(mean_embeddings)

    # Save raw data for custom plotting
    np.save(str(output_dir / "frame_embeddings_2d.npy"), frame_2d)
    np.save(str(output_dir / "utterance_embeddings_2d.npy"), utt_2d)
    with open(output_dir / "frame_labels.json", "w") as f:
        json.dump(all_labels, f)
    with open(output_dir / "utterance_labels.json", "w") as f:
        json.dump(all_utterance_labels, f)

    # Plot with matplotlib
    print("\nGenerating plots...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lang_colors = {"English": "#2196F3", "Chinese": "#F44336", "Hindi": "#4CAF50", "Japanese": "#FF9800"}

    # Plot 1: Frame-level embeddings colored by language
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for lang, color in lang_colors.items():
        mask = [l["language"] == lang for l in all_labels]
        pts = frame_2d[mask]
        if len(pts) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], c=color, label=lang, alpha=0.5, s=15)
    ax.legend(fontsize=12)
    ax.set_title("JEPA Encoder Frame Embeddings (t-SNE)\nColored by Language — Trained on English Only", fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(str(output_dir / "frame_embeddings_by_language.png"), dpi=200)
    print(f"  Saved frame_embeddings_by_language.png")

    # Plot 2: Utterance-level colored by language
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for lang, color in lang_colors.items():
        mask = [l["language"] == lang for l in all_utterance_labels]
        pts = utt_2d[mask]
        if len(pts) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], c=color, label=lang, alpha=0.7, s=50, edgecolors="white", linewidth=0.5)
    ax.legend(fontsize=12)
    ax.set_title("JEPA Encoder Utterance Embeddings (t-SNE, mean-pooled)\nColored by Language — Trained on English Only", fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(str(output_dir / "utterance_embeddings_by_language.png"), dpi=200)
    print(f"  Saved utterance_embeddings_by_language.png")

    # Plot 3: Frame-level colored by position in utterance (temporal structure)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    positions = np.array([l["frame_pos"] for l in all_labels])
    sc = ax.scatter(frame_2d[:, 0], frame_2d[:, 1], c=positions, cmap="viridis", alpha=0.5, s=15)
    plt.colorbar(sc, ax=ax, label="Position in Utterance (0=start, 1=end)")
    ax.set_title("JEPA Encoder Frame Embeddings (t-SNE)\nColored by Temporal Position", fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(str(output_dir / "frame_embeddings_by_position.png"), dpi=200)
    print(f"  Saved frame_embeddings_by_position.png")

    # Plot 4: English only, colored by speaker (do speakers cluster?)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    en_mask = [l["language"] == "English" for l in all_utterance_labels]
    en_pts = utt_2d[en_mask]
    en_speakers = [l["speaker"] for l, m in zip(all_utterance_labels, en_mask) if m]
    unique_speakers = list(set(en_speakers))
    speaker_cmap = plt.cm.tab20(np.linspace(0, 1, len(unique_speakers)))
    for i, spk in enumerate(unique_speakers):
        spk_mask = [s == spk for s in en_speakers]
        pts = en_pts[spk_mask]
        if len(pts) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], color=speaker_cmap[i], label=spk if len(unique_speakers) < 15 else None,
                      alpha=0.7, s=60, edgecolors="white", linewidth=0.5)
    ax.set_title("English Utterance Embeddings (t-SNE)\nColored by Speaker ID", fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    if len(unique_speakers) < 15:
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(str(output_dir / "english_by_speaker.png"), dpi=200)
    print(f"  Saved english_by_speaker.png")

    # Embedding statistics
    print("\n=== Embedding Statistics ===")
    print(f"Per-dim mean: {embeddings.mean(axis=0).mean():.4f} (should be ~0)")
    print(f"Per-dim std:  {embeddings.std(axis=0).mean():.4f}")
    print(f"Per-dim std range: [{embeddings.std(axis=0).min():.4f}, {embeddings.std(axis=0).max():.4f}]")
    cov = np.cov(embeddings.T)
    eigvals = np.linalg.eigvalsh(cov)
    effective_rank = np.exp(-np.sum(eigvals/eigvals.sum() * np.log(eigvals/eigvals.sum() + 1e-10)))
    print(f"Effective rank: {effective_rank:.1f} / 128 dims")
    print(f"Top eigenvalue ratio: {eigvals[-1]/eigvals.sum():.4f}")

    stats = {
        "n_frames": len(embeddings),
        "n_utterances": len(mean_embeddings),
        "per_dim_mean": float(embeddings.mean()),
        "per_dim_std": float(embeddings.std(axis=0).mean()),
        "effective_rank": float(effective_rank),
    }
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nAll outputs saved to {output_dir}/")
    print("Files:")
    for f in sorted(output_dir.glob("*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
