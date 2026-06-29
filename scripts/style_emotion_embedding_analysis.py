"""Style/emotion frame-level embedding analysis on two speaking-style datasets.

Loads 100 random samples from each of two style datasets (whisper, angry).
Dataset ids are read from env vars EMOTION_WHISPER_DS / EMOTION_ANGRY_DS.

Extracts frame-level embeddings via JEPA v9 + JEPA-SIGReg + EnCodec + DAC + Mimi,
applies fair protocol, generates t-SNE plots labeled by source dataset.

This tests whether the encoder separates speech STYLE/EMOTION
(orthogonal axis from language).

Usage:
    cd ~/koe
    CUDA_VISIBLE_DEVICES=1 HF_TOKEN=... .venv/bin/python scripts/style_emotion_embedding_analysis.py
"""

import gc
import json
import math
import os
import sys
import time
import warnings

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, os.path.expanduser("~/koe"))
warnings.filterwarnings("ignore")

SAMPLE_RATE = 24000
OUTPUT_DIR = "/local_data/analysis/style_emotion_embeddings"
N_PER_DATASET = 100
N_FRAMES_PER_UTT = 30  # use more frames since fewer utterances
RANDOM_SEED = 42

DATASETS = {
    "whisper": os.environ.get("EMOTION_WHISPER_DS", ""),
    "angry": os.environ.get("EMOTION_ANGRY_DS", ""),
}
COLORS = {"whisper": "#4472C4", "angry": "#E74C3C"}


def load_hf_datasets():
    """Load 100 random samples from each HF dataset."""
    from datasets import load_dataset
    from huggingface_hub import login

    if "HF_TOKEN" in os.environ:
        login(token=os.environ["HF_TOKEN"])

    rng = np.random.RandomState(RANDOM_SEED)
    wavs, labels = [], []

    for label, repo in DATASETS.items():
        print(f"[data] Loading {repo}...")
        ds = None
        # Try multiple loading strategies
        for attempt, kwargs in enumerate([
            {"split": "train", "verification_mode": "no_checks"},
            {"split": "train", "verification_mode": "no_checks", "download_mode": "force_redownload"},
            {"verification_mode": "no_checks"},
            {"streaming": True, "split": "train"},
        ]):
            try:
                print(f"  attempt {attempt+1}: {kwargs}")
                ds = load_dataset(repo, **kwargs)
                if hasattr(ds, "keys") and not isinstance(ds, list):
                    # DatasetDict
                    split_name = "train" if "train" in ds else list(ds.keys())[0]
                    ds = ds[split_name]
                print(f"  loaded via attempt {attempt+1}")
                break
            except Exception as e:
                print(f"  attempt {attempt+1} failed: {type(e).__name__}: {str(e)[:200]}")
                continue
        if ds is None:
            print(f"  ALL ATTEMPTS FAILED for {repo}")
            continue

        # Handle streaming vs non-streaming
        if kwargs.get("streaming"):
            print(f"  {repo}: streaming mode")
            stream_items = []
            for i, item in enumerate(ds):
                if i >= 2000:  # cap streaming exploration
                    break
                stream_items.append(item)
            n = len(stream_items)
            print(f"  collected {n} items from stream")
            indices = rng.choice(n, size=min(N_PER_DATASET, n), replace=False)
            ds = stream_items
        else:
            n = len(ds)
            print(f"  {repo}: {n} samples")
            indices = rng.choice(n, size=min(N_PER_DATASET, n), replace=False)

        loaded = 0
        for idx in indices:
            try:
                item = ds[int(idx)]
                # Try common audio field names
                audio = None
                for key in ["audio", "wav", "speech"]:
                    if key in item:
                        audio = item[key]
                        break
                if audio is None:
                    print(f"    No audio field in item, keys: {list(item.keys())}")
                    continue
                if isinstance(audio, dict):
                    wav = np.array(audio["array"], dtype=np.float32)
                    sr = audio["sampling_rate"]
                else:
                    wav = np.array(audio, dtype=np.float32)
                    sr = item.get("sampling_rate", SAMPLE_RATE)
                if sr != SAMPLE_RATE:
                    wav_t = torch.from_numpy(wav).unsqueeze(0)
                    wav_t = torchaudio.functional.resample(wav_t, sr, SAMPLE_RATE)
                    wav = wav_t.squeeze(0).numpy()
                if wav.ndim > 1:
                    wav = wav.mean(axis=-1)
                dur = len(wav) / SAMPLE_RATE
                if dur < 1.0 or dur > 30.0:
                    continue
                # Trim/pad to max 10 seconds for memory
                if dur > 10.0:
                    wav = wav[: 10 * SAMPLE_RATE]
                wavs.append(wav)
                labels.append(label)
                loaded += 1
            except Exception as e:
                print(f"    Failed item {idx}: {e}")
                continue
        print(f"  Loaded {loaded} samples")

    print(f"\n[data] Total: {len(wavs)} samples")
    label_counts = {}
    for l in labels:
        label_counts[l] = label_counts.get(l, 0) + 1
    print(f"  Counts: {label_counts}")
    return wavs, labels


def _subsample_frames(z, lang, n_per_utt):
    T = z.shape[1]
    step = max(1, T // n_per_utt)
    embs, lbls = [], []
    for t in range(0, T, step):
        embs.append(z[:, t])
        lbls.append(lang)
    return embs, lbls


def extract_jepa_v9(wavs, labels, device):
    from koe.codec_impl import WaveformJEPAFSQVAE
    ckpt = torch.load("/local_data/checkpoints/v9_wavlm_warmstart/stage2_latest.pt",
                       map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    strides = cfg.get("strides", [4, 4, 4, 5, 6])
    model = WaveformJEPAFSQVAE(
        sample_rate=24000, code_dim=128,
        channels=[64, 128, 256, 384, 512, 512],
        strides=strides, n_res_blocks=8, n_conformer=8, conformer_heads=16,
        fsq_levels=cfg.get("fsq_levels", [8, 8, 8, 8]),
        hifi_kernels=[3, 7, 11, 15, 23, 32],
    )
    model.load_state_dict(ckpt.get("state_dict", {}), strict=False)
    model = model.to(device, dtype=torch.bfloat16).eval()
    hop = math.prod(strides)
    all_e, all_l = [], []
    for i, (wav, lab) in enumerate(zip(wavs, labels)):
        rem = len(wav) % hop
        wav_pad = np.pad(wav, (0, hop - rem)) if rem else wav
        wav_t = torch.from_numpy(wav_pad).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            z_e = model.encoder.encode(wav_t)
        z = z_e[0].float().cpu().numpy()
        e, l = _subsample_frames(z, lab, N_FRAMES_PER_UTT)
        all_e.extend(e); all_l.extend(l)
    del model; torch.cuda.empty_cache(); gc.collect()
    return np.stack(all_e), all_l


def extract_jepa_sigreg(wavs, labels, device):
    from koe.codec_impl import JEPAEncoder
    ckpt = torch.load("/local_data/checkpoints/q2d2_sigreg/v2_latest.pt",
                       map_location="cpu", weights_only=False)
    strides = ckpt.get("config", {}).get("strides", [4, 4, 4, 5, 3])
    encoder = JEPAEncoder(
        sample_rate=24000, code_dim=128,
        channels=[64, 128, 256, 384, 512, 512],
        strides=strides, n_res_blocks=8, n_conformer=8, conformer_heads=16,
    )
    enc_state = {k.replace("encoder.", "", 1): v for k, v in ckpt["state_dict"].items()
                 if k.startswith("encoder.")}
    encoder.load_state_dict(enc_state, strict=False)
    encoder = encoder.to(device, dtype=torch.bfloat16).eval()
    hop = math.prod(strides)
    all_e, all_l = [], []
    for i, (wav, lab) in enumerate(zip(wavs, labels)):
        rem = len(wav) % hop
        wav_pad = np.pad(wav, (0, hop - rem)) if rem else wav
        wav_t = torch.from_numpy(wav_pad).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            z_e = encoder.encode(wav_t)
        z = z_e[0].float().cpu().numpy()
        e, l = _subsample_frames(z, lab, N_FRAMES_PER_UTT)
        all_e.extend(e); all_l.extend(l)
    del encoder; torch.cuda.empty_cache(); gc.collect()
    return np.stack(all_e), all_l


def extract_encodec(wavs, labels, device):
    from encodec import EncodecModel
    model = EncodecModel.encodec_model_24khz().to(device).eval()
    all_e, all_l = [], []
    for wav, lab in zip(wavs, labels):
        wav_t = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            z = model.encoder(wav_t)
        z_np = z[0].float().cpu().numpy()
        e, l = _subsample_frames(z_np, lab, N_FRAMES_PER_UTT)
        all_e.extend(e); all_l.extend(l)
    del model; torch.cuda.empty_cache(); gc.collect()
    return np.stack(all_e), all_l


def extract_dac(wavs, labels, device):
    import dac
    model_path = dac.utils.download(model_type="24khz")
    model = dac.DAC.load(model_path).to(device).eval()
    hop = int(np.prod(model.encoder_rates))
    all_e, all_l = [], []
    for wav, lab in zip(wavs, labels):
        wav_t = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).float().to(device)
        length = wav_t.shape[-1]
        right_pad = math.ceil(length / hop) * hop - length
        if right_pad > 0:
            wav_t = torch.nn.functional.pad(wav_t, (0, int(right_pad)))
        with torch.no_grad():
            z = model.encoder(wav_t)
        z_np = z[0].float().cpu().numpy()
        e, l = _subsample_frames(z_np, lab, N_FRAMES_PER_UTT)
        all_e.extend(e); all_l.extend(l)
    del model; torch.cuda.empty_cache(); gc.collect()
    return np.stack(all_e), all_l


def extract_mimi(wavs, labels, device):
    from moshi.models import loaders
    mimi_weight = loaders.hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device=device)
    mimi.eval()
    hop = int(SAMPLE_RATE / mimi.frame_rate)
    all_e, all_l = [], []
    for wav, lab in zip(wavs, labels):
        wav_t = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).float().to(device)
        rem = wav_t.shape[-1] % hop
        if rem:
            wav_t = torch.nn.functional.pad(wav_t, (0, hop - rem))
        with torch.no_grad():
            emb = mimi.encode_to_latent(wav_t) if hasattr(mimi, "encode_to_latent") else mimi.encoder(wav_t)
        z = emb[0].float().cpu().numpy()
        e, l = _subsample_frames(z, lab, N_FRAMES_PER_UTT)
        all_e.extend(e); all_l.extend(l)
    del mimi; torch.cuda.empty_cache(); gc.collect()
    return np.stack(all_e), all_l


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip_mimi", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_start = time.time()

    print("\n=== STEP 1: Load datasets ===")
    wavs, labels = load_hf_datasets()
    if len(wavs) == 0:
        print("ERROR: No samples loaded")
        return

    print("\n=== STEP 2: Extract embeddings (all models) ===")
    raw = {}

    print("\n[1/5] JEPA-EMA v9...")
    raw["JEPA-EMA (v9)"] = extract_jepa_v9(wavs, labels, args.device)
    print(f"  {raw['JEPA-EMA (v9)'][0].shape}")

    print("\n[2/5] JEPA-SIGReg...")
    raw["JEPA-SIGReg (25Hz)"] = extract_jepa_sigreg(wavs, labels, args.device)
    print(f"  {raw['JEPA-SIGReg (25Hz)'][0].shape}")

    print("\n[3/5] EnCodec...")
    raw["EnCodec"] = extract_encodec(wavs, labels, args.device)
    print(f"  {raw['EnCodec'][0].shape}")

    print("\n[4/5] DAC...")
    raw["DAC"] = extract_dac(wavs, labels, args.device)
    print(f"  {raw['DAC'][0].shape}")

    if not args.skip_mimi:
        print("\n[5/5] Mimi...")
        try:
            raw["Mimi"] = extract_mimi(wavs, labels, args.device)
            print(f"  {raw['Mimi'][0].shape}")
        except Exception as e:
            print(f"  Mimi failed: {e}")

    print("\n=== STEP 3: Fair protocol ===")
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    normalized = {}
    for name, (embs, lbl) in raw.items():
        scaler = StandardScaler()
        embs_norm = scaler.fit_transform(embs)
        pca = PCA(n_components=min(50, embs.shape[1]), random_state=RANDOM_SEED)
        embs_pca = pca.fit_transform(embs_norm)
        print(f"  {name}: {embs.shape[1]}d -> {embs_pca.shape[1]}d, "
              f"var={pca.explained_variance_ratio_.sum():.3f}")
        normalized[name] = (embs_pca, lbl)

    # Balance dataset labels
    rng = np.random.RandomState(RANDOM_SEED)
    equalized = {}
    for name, (embs, lbl) in normalized.items():
        arr = np.array(lbl)
        counts = {l: (arr == l).sum() for l in np.unique(arr)}
        min_count = min(counts.values())
        indices = []
        for l in np.unique(arr):
            li = np.where(arr == l)[0]
            indices.extend(rng.choice(li, size=min_count, replace=False))
        equalized[name] = (embs[indices], [lbl[i] for i in sorted(indices)])
        print(f"  {name}: balanced to {min_count}/dataset")

    print("\n=== STEP 4: Classifiers (style separation) ===")
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    clf_results = {}
    for name, (embs, lbl) in equalized.items():
        arr = np.array(lbl)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        clfs = {
            "LogReg": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
            "LinearSVM": LinearSVC(max_iter=2000, random_state=RANDOM_SEED),
            "5-NN": KNeighborsClassifier(n_neighbors=5),
        }
        model_res = {}
        for cn, clf in clfs.items():
            scores = cross_val_score(clf, embs, arr, cv=skf, scoring="accuracy")
            model_res[cn] = {"mean": float(scores.mean()), "std": float(scores.std())}
            print(f"  {name:25s} | {cn:10s}: {scores.mean():.3f} +/- {scores.std():.3f}")
        clf_results[name] = model_res
    with open(os.path.join(OUTPUT_DIR, "style_classifier_results.json"), "w") as f:
        json.dump(clf_results, f, indent=2)

    print("\n=== STEP 5: t-SNE plots ===")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    model_order = ["JEPA-EMA (v9)", "JEPA-SIGReg (25Hz)", "Mimi", "EnCodec", "DAC"]
    avail = [m for m in model_order if m in equalized]
    n = len(avail)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for mi, name in enumerate(avail):
        embs, lbl = equalized[name]
        # Use up to 10K frames for viz
        if len(embs) > 10000:
            idx = rng.choice(len(embs), 10000, replace=False)
            embs = embs[idx]
            lbl = [lbl[i] for i in idx]
        tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED, max_iter=1000)
        coords = tsne.fit_transform(embs)
        ax = axes[mi]
        for label, color in COLORS.items():
            mask = np.array([l == label for l in lbl])
            pts = coords[mask]
            if len(pts) > 0:
                ax.scatter(pts[:, 0], pts[:, 1], c=color, s=3, alpha=0.4,
                           label=label, rasterized=True)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        if mi == 0:
            ax.legend(fontsize=8, markerscale=3, loc="upper left")

    fig.suptitle("Speech Style Separation: Whisper vs Angry (frame-level)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "style_emotion_tsne.png"),
                dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, "style_emotion_tsne.pdf"),
                bbox_inches="tight")
    plt.close()
    print(f"  Saved style_emotion_tsne.png + .pdf")

    # Also utterance-level mean-pooled
    print("\n=== STEP 6: Utterance-level (mean-pooled) ===")
    utt_embs = {}
    for name, (embs, lbl) in raw.items():
        # Group by utterance index — approximate using contiguous chunks of N_FRAMES_PER_UTT
        n_utts = len(embs) // N_FRAMES_PER_UTT
        utt_emb_list = []
        utt_lbl_list = []
        for u in range(n_utts):
            start = u * N_FRAMES_PER_UTT
            end = start + N_FRAMES_PER_UTT
            utt_emb_list.append(embs[start:end].mean(axis=0))
            utt_lbl_list.append(lbl[start])
        utt_embs[name] = (np.stack(utt_emb_list), utt_lbl_list)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for mi, name in enumerate(avail):
        embs, lbl = utt_embs[name]
        if len(embs) < 5:
            continue
        embs_norm = StandardScaler().fit_transform(embs)
        perp = min(30, len(embs) - 1)
        tsne = TSNE(n_components=2, perplexity=perp, random_state=RANDOM_SEED, max_iter=1000)
        coords = tsne.fit_transform(embs_norm)
        ax = axes[mi]
        for label, color in COLORS.items():
            mask = np.array([l == label for l in lbl])
            pts = coords[mask]
            if len(pts) > 0:
                ax.scatter(pts[:, 0], pts[:, 1], c=color, s=30, alpha=0.7,
                           label=label, edgecolors="white", linewidth=0.5)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        if mi == 0:
            ax.legend(fontsize=8, markerscale=2, loc="upper left")

    fig.suptitle("Utterance-Level: Whisper vs Angry (mean-pooled)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "style_emotion_utterance_tsne.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved style_emotion_utterance_tsne.png")

    elapsed = time.time() - t_start
    print(f"\nCOMPLETE in {elapsed/60:.1f} min")
    for fn in sorted(os.listdir(OUTPUT_DIR)):
        sz = os.path.getsize(os.path.join(OUTPUT_DIR, fn))
        print(f"  {fn} ({sz/1024:.1f} KB)")


if __name__ == "__main__":
    main()
