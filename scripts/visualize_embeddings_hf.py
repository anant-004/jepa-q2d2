"""Frame-level + utterance-level (mean-pooled) embedding t-SNE, weights from HF.

Loads a JEPA-Q2D2 codec from https://huggingface.co/Andy004/jepa-q2d2, runs audio
through the encoder to get pre-quantization frame features (128-dim), then:
  - frame-level t-SNE      (subsampled frames, colored by language)
  - utterance-level t-SNE  (mean-pooled per clip, colored by language)
plus a temporal-position view and an English-by-speaker view.

Audio source (pick one):
  --audio_dir DIR   : every .wav/.flac in DIR (label = parent folder name)
  (default)         : multilingual clips streamed from public HF datasets
                      (set HF_TOKEN in the environment if a source needs auth)

This is the HF-loading port of koe/fast/visualize_embeddings.py — NO tokens are
hardcoded; auth (only needed for gated datasets) comes from the HF_TOKEN env var.

Usage:
    python scripts/visualize_embeddings_hf.py --model teacher --output_dir ./embedding_viz
    python scripts/visualize_embeddings_hf.py --audio_dir ./my_clips --output_dir ./viz_local
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio

from koe.fast.hf_codec import load_codec_from_hf, MODELS

SAMPLE_RATE = 24000
LANG_COLORS = {"English": "#2196F3", "Chinese": "#F44336",
               "Hindi": "#4CAF50", "Japanese": "#FF9800"}


@torch.no_grad()
def frame_features(model, wav_24k: np.ndarray, hop: int, device: str, dtype) -> np.ndarray:
    """Pre-quantization encoder features -> [T, 128]."""
    rem = len(wav_24k) % hop
    if rem:
        wav_24k = np.pad(wav_24k, (0, hop - rem))
    x = torch.from_numpy(wav_24k).view(1, 1, -1).to(device, dtype=dtype)
    z = model.encoder.encode(x)          # [1, 128, T]
    return z[0].float().cpu().numpy().T  # [T, 128]


def _resample(wav: np.ndarray, sr: int) -> np.ndarray:
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if sr != SAMPLE_RATE:
        t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
        wav = torchaudio.functional.resample(t, sr, SAMPLE_RATE).squeeze(0).numpy()
    return np.ascontiguousarray(wav, dtype=np.float32)


def load_from_dir(audio_dir: str):
    import soundfile as sf
    samples = []
    for p in sorted(Path(audio_dir).rglob("*")):
        if p.suffix.lower() not in (".wav", ".flac", ".ogg"):
            continue
        wav, sr = sf.read(str(p), dtype="float32")
        samples.append({"wav": _resample(wav, sr), "sr": SAMPLE_RATE,
                        "language": p.parent.name, "speaker": p.parent.name,
                        "text": p.stem})
    print(f"[data] loaded {len(samples)} clips from {audio_dir}")
    return samples


def load_multilingual(n_per_lang: int):
    """Stream a few clips per language from public HF datasets."""
    from datasets import load_dataset
    token = os.environ.get("HF_TOKEN")  # only needed for gated sources
    specs = [
        ("English", lambda: load_dataset("librispeech_asr", split="test.clean",
                                         streaming=True, token=token), "text"),
        ("Chinese", lambda: load_dataset("google/fleurs", "cmn_hans_cn", split="test",
                                         streaming=True, trust_remote_code=True, token=token), "transcription"),
        ("Japanese", lambda: load_dataset("google/fleurs", "ja_jp", split="test",
                                          streaming=True, trust_remote_code=True, token=token), "transcription"),
    ]
    samples = []
    for lang, mk, text_key in specs:
        print(f"[data] {lang} ...")
        try:
            ds = mk()
        except Exception as e:
            print(f"  skipped {lang}: {e}")
            continue
        c = 0
        for item in ds:
            if c >= n_per_lang:
                break
            audio = item.get("audio") or {}
            if "array" not in audio:
                continue
            dur = len(audio["array"]) / audio["sampling_rate"]
            if not (3.0 <= dur <= 10.0):
                continue
            samples.append({
                "wav": _resample(np.asarray(audio["array"], dtype=np.float32), audio["sampling_rate"]),
                "sr": SAMPLE_RATE, "language": lang,
                "speaker": str(item.get("speaker_id", f"{lang}_{c}")),
                "text": str(item.get(text_key, ""))[:50],
            })
            c += 1
        print(f"  got {c}")
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="teacher", help=f"one of {list(MODELS)} or a subfolder")
    ap.add_argument("--repo_id", default="Andy004/jepa-q2d2")
    ap.add_argument("--audio_dir", default=None, help="use local clips instead of HF datasets")
    ap.add_argument("--n_per_lang", type=int, default=20)
    ap.add_argument("--output_dir", default="./embedding_viz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32

    model, info = load_codec_from_hf(args.model, repo_id=args.repo_id,
                                     device=args.device, dtype=dtype)
    hop = info["hop"]
    print(f"Loaded {info['subfolder']} ({info['frame_rate_hz']:.1f} Hz, "
          f"quantizer={info['quantizer']}, missing={info['missing_keys']})")

    samples = load_from_dir(args.audio_dir) if args.audio_dir else load_multilingual(args.n_per_lang)
    if not samples:
        raise SystemExit("No audio samples loaded.")

    frame_emb, frame_lab, utt_emb, utt_lab = [], [], [], []
    for i, s in enumerate(samples):
        z = frame_features(model, s["wav"], hop, args.device, dtype)  # [T, 128]
        T = z.shape[0]
        step = max(1, T // 10)
        for t in range(0, T, step):
            frame_emb.append(z[t])
            frame_lab.append({"language": s["language"], "speaker": s["speaker"],
                              "frame_pos": t / max(T, 1)})
        utt_emb.append(z.mean(axis=0))
        utt_lab.append({"language": s["language"], "speaker": s["speaker"], "text": s["text"]})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(samples)}")

    frame_emb = np.stack(frame_emb)
    utt_emb = np.stack(utt_emb)
    print(f"frame: {frame_emb.shape}  utterance: {utt_emb.shape}")

    from sklearn.manifold import TSNE
    frame_2d = TSNE(n_components=2, perplexity=30, random_state=42,
                    init="pca").fit_transform(frame_emb)
    utt_2d = TSNE(n_components=2, perplexity=min(15, len(utt_emb) - 1),
                  random_state=42, init="pca").fit_transform(utt_emb)

    np.save(out / "frame_embeddings_2d.npy", frame_2d)
    np.save(out / "utterance_embeddings_2d.npy", utt_2d)
    (out / "frame_labels.json").write_text(json.dumps(frame_lab))
    (out / "utterance_labels.json").write_text(json.dumps(utt_lab))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def scatter_by_lang(xy, labels, title, fname, s, alpha):
        fig, ax = plt.subplots(figsize=(10, 8))
        langs = sorted(set(l["language"] for l in labels))
        for lang in langs:
            m = [l["language"] == lang for l in labels]
            pts = xy[m]
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], c=LANG_COLORS.get(lang), label=lang,
                           alpha=alpha, s=s, edgecolors="white", linewidth=0.3)
        ax.legend(fontsize=12); ax.set_title(title, fontsize=13)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        plt.tight_layout(); plt.savefig(out / fname, dpi=200); plt.close(fig)
        print(f"  saved {fname}")

    scatter_by_lang(frame_2d, frame_lab,
                    f"Frame embeddings t-SNE ({info['subfolder']})",
                    "frame_embeddings_by_language.png", s=15, alpha=0.5)
    scatter_by_lang(utt_2d, utt_lab,
                    f"Utterance embeddings t-SNE, mean-pooled ({info['subfolder']})",
                    "utterance_embeddings_by_language.png", s=55, alpha=0.8)

    # temporal-position view (frame-level)
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = np.array([l["frame_pos"] for l in frame_lab])
    sc = ax.scatter(frame_2d[:, 0], frame_2d[:, 1], c=pos, cmap="viridis", alpha=0.5, s=15)
    plt.colorbar(sc, ax=ax, label="position in utterance (0=start, 1=end)")
    ax.set_title("Frame embeddings t-SNE — temporal position")
    plt.tight_layout(); plt.savefig(out / "frame_embeddings_by_position.png", dpi=200); plt.close(fig)
    print("  saved frame_embeddings_by_position.png")

    # simple effective-rank stat
    cov = np.cov(frame_emb.T)
    ev = np.linalg.eigvalsh(cov); p = ev / ev.sum()
    erank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    (out / "stats.json").write_text(json.dumps({
        "model": info["subfolder"], "n_frames": int(len(frame_emb)),
        "n_utterances": int(len(utt_emb)), "effective_rank": erank,
    }, indent=2))
    print(f"effective rank: {erank:.1f} / {frame_emb.shape[1]}")
    print(f"All outputs in {out}/")


if __name__ == "__main__":
    main()
