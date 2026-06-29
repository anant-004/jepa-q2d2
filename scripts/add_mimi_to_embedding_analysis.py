"""Add Mimi (Moshi codec) to the cross-codec frame-level embedding analysis.

Mimi runs at 12.5 Hz — same frame rate as our v9 codec! Perfect fair comparison.

This script:
1. Loads the same FLEURS 6-language audio as the prior analysis
2. Extracts Mimi encoder frame embeddings
3. Runs the same fair protocol (normalize, PCA, balance frames)
4. Adds Mimi to the existing comparison plots and metrics

Usage:
    cd ~/koe
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/add_mimi_to_embedding_analysis.py
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

sys.path.insert(0, os.path.expanduser("~/koe"))
warnings.filterwarnings("ignore")

SAMPLE_RATE = 24000
FLEURS_DIR = "/local_data/data/fleurs_full"
OUTPUT_DIR = "/local_data/analysis/with_mimi_embeddings"

LANGUAGES = ["en_us", "hi_in", "ja_jp", "de_de", "fr_fr", "cmn_hans_cn"]
LANG_LABELS = {
    "en_us": "English", "hi_in": "Hindi", "ja_jp": "Japanese",
    "de_de": "German", "fr_fr": "French", "cmn_hans_cn": "Chinese",
}
LANG_COLORS = {
    "English": "#2196F3", "Hindi": "#4CAF50", "Japanese": "#FF9800",
    "German": "#9C27B0", "French": "#795548", "Chinese": "#F44336",
}
N_FRAMES_PER_UTT = 20
RANDOM_SEED = 42


def load_audio():
    wavs_by_lang = {}
    for lang_key in LANGUAGES:
        lang_dir = os.path.join(FLEURS_DIR, lang_key)
        if not os.path.isdir(lang_dir):
            continue
        files = sorted([f for f in os.listdir(lang_dir) if f.endswith(".wav")])
        lang_wavs = []
        for f in files:
            try:
                wav, sr = sf.read(os.path.join(lang_dir, f), dtype="float32")
                if sr != SAMPLE_RATE:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
                if len(wav) < SAMPLE_RATE:
                    continue
                lang_wavs.append(wav)
            except Exception:
                continue
        wavs_by_lang[lang_key] = lang_wavs

    min_count = min(len(v) for v in wavs_by_lang.values())
    rng = np.random.RandomState(RANDOM_SEED)
    wavs, labels = [], []
    for lang_key in LANGUAGES:
        all_wavs = wavs_by_lang[lang_key]
        indices = rng.choice(len(all_wavs), size=min_count, replace=False)
        for idx in sorted(indices):
            wavs.append(all_wavs[idx])
            labels.append(LANG_LABELS[lang_key])
    print(f"[data] Loaded {len(wavs)} utterances ({min_count}/lang)")
    return wavs, labels


def extract_mimi(wavs, labels, device):
    """Extract Mimi encoder frame embeddings.

    Mimi runs at 24kHz, 12.5 Hz frame rate via moshi library.
    """
    from moshi.models import loaders

    print("[mimi] Downloading Mimi checkpoint...")
    mimi_weight = loaders.hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    print(f"[mimi] Loaded from {mimi_weight}")
    mimi = loaders.get_mimi(mimi_weight, device=device)
    mimi.eval()
    print(f"[mimi] Frame rate: {mimi.frame_rate} Hz, sample rate: {mimi.sample_rate} Hz")
    hop = int(SAMPLE_RATE / mimi.frame_rate)  # 1920 at 12.5 Hz

    all_embs, all_labels = [], []
    for i, (wav, lang) in enumerate(zip(wavs, labels)):
        # Mimi expects 24kHz mono
        wav_t = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).float().to(device)
        # Pad to multiple of frame_rate for clean frames
        rem = wav_t.shape[-1] % hop
        if rem:
            wav_t = torch.nn.functional.pad(wav_t, (0, hop - rem))
        with torch.no_grad():
            # Encoder gives latent before quantization
            emb = mimi.encode_to_latent(wav_t) if hasattr(mimi, "encode_to_latent") else mimi.encoder(wav_t)
        # emb: [1, C, T]
        z = emb[0].float().cpu().numpy()
        T = z.shape[1]
        step = max(1, T // N_FRAMES_PER_UTT)
        for t in range(0, T, step):
            all_embs.append(z[:, t])
            all_labels.append(lang)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(wavs)}")

    del mimi
    torch.cuda.empty_cache()
    gc.collect()
    return np.stack(all_embs), all_labels


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_start = time.time()

    # Load HF token from env
    if "HF_TOKEN" in os.environ:
        from huggingface_hub import login
        login(token=os.environ["HF_TOKEN"])

    print("\n=== STEP 1: Load FLEURS audio ===")
    wavs, labels = load_audio()

    print("\n=== STEP 2: Extract Mimi embeddings ===")
    mimi_embs, mimi_labels = extract_mimi(wavs, labels, args.device)
    print(f"  Mimi: {mimi_embs.shape}")

    # Save raw Mimi embeddings
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "mimi_embeddings.npz"),
        embs=mimi_embs.astype(np.float32),
        labels=np.array(mimi_labels),
    )
    print(f"  Saved {OUTPUT_DIR}/mimi_embeddings.npz")

    # Compute raw stats
    cov = np.cov(mimi_embs.T)
    eigvals = np.linalg.eigvalsh(cov)
    p = eigvals / eigvals.sum()
    erank = float(np.exp(-np.sum(p * np.log(p + 1e-10))))
    stats = {
        "Mimi": {
            "n_frames": len(mimi_embs),
            "n_dims": mimi_embs.shape[1],
            "effective_rank": erank,
            "per_dim_std": float(mimi_embs.std(axis=0).mean()),
        }
    }
    with open(os.path.join(OUTPUT_DIR, "mimi_raw_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Mimi: {mimi_embs.shape[1]}d, erank={erank:.1f}")

    # Load previously saved embeddings if available, build 5-model comparison
    prev_dir = "/local_data/analysis/comprehensive_embeddings"

    print("\n=== STEP 3: Build 5-model comparison ===")
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Re-extract all 4 prior models to keep things consistent
    # Use exact same code path as before for repeatability
    # For now, just extract Mimi + load FLEURS embeddings from prior runs if persisted

    # Build comparison: load saved npz files if they exist, else compute on the fly
    raw_embeddings = {"Mimi": (mimi_embs, mimi_labels, None)}

    # We didn't persist the other 4 model embeddings to npz, so we need to re-extract.
    # Do that here to make this self-contained.

    print("[1/4] Extracting JEPA-EMA v9 embeddings...")
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
    model = model.to(args.device, dtype=torch.bfloat16).eval()
    hop = math.prod(strides)
    all_e, all_l = [], []
    for i, (wav, lang) in enumerate(zip(wavs, labels)):
        rem = len(wav) % hop
        wav_pad = np.pad(wav, (0, hop - rem)) if rem else wav
        wav_t = torch.from_numpy(wav_pad).unsqueeze(0).unsqueeze(0).to(args.device, dtype=torch.bfloat16)
        with torch.no_grad():
            z_e = model.encoder.encode(wav_t)
        z = z_e[0].float().cpu().numpy()
        T = z.shape[1]
        step = max(1, T // N_FRAMES_PER_UTT)
        for t in range(0, T, step):
            all_e.append(z[:, t])
            all_l.append(lang)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(wavs)}")
    raw_embeddings["JEPA-EMA (v9)"] = (np.stack(all_e), all_l, None)
    del model; torch.cuda.empty_cache(); gc.collect()

    print("[2/4] Extracting JEPA-SIGReg embeddings...")
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
    encoder = encoder.to(args.device, dtype=torch.bfloat16).eval()
    hop = math.prod(strides)
    all_e, all_l = [], []
    for i, (wav, lang) in enumerate(zip(wavs, labels)):
        rem = len(wav) % hop
        wav_pad = np.pad(wav, (0, hop - rem)) if rem else wav
        wav_t = torch.from_numpy(wav_pad).unsqueeze(0).unsqueeze(0).to(args.device, dtype=torch.bfloat16)
        with torch.no_grad():
            z_e = encoder.encode(wav_t)
        z = z_e[0].float().cpu().numpy()
        T = z.shape[1]
        step = max(1, T // N_FRAMES_PER_UTT)
        for t in range(0, T, step):
            all_e.append(z[:, t])
            all_l.append(lang)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(wavs)}")
    raw_embeddings["JEPA-SIGReg (25Hz)"] = (np.stack(all_e), all_l, None)
    del encoder; torch.cuda.empty_cache(); gc.collect()

    print("[3/4] Extracting EnCodec embeddings...")
    from encodec import EncodecModel
    model = EncodecModel.encodec_model_24khz().to(args.device).eval()
    all_e, all_l = [], []
    for i, (wav, lang) in enumerate(zip(wavs, labels)):
        wav_t = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).float().to(args.device)
        with torch.no_grad():
            z = model.encoder(wav_t)
        z_np = z[0].float().cpu().numpy()
        T = z_np.shape[1]
        step = max(1, T // N_FRAMES_PER_UTT)
        for t in range(0, T, step):
            all_e.append(z_np[:, t])
            all_l.append(lang)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(wavs)}")
    raw_embeddings["EnCodec"] = (np.stack(all_e), all_l, None)
    del model; torch.cuda.empty_cache(); gc.collect()

    print("[4/4] Extracting DAC embeddings...")
    import dac
    model_path = dac.utils.download(model_type="24khz")
    model = dac.DAC.load(model_path).to(args.device).eval()
    hop_dac = int(np.prod(model.encoder_rates))
    all_e, all_l = [], []
    for i, (wav, lang) in enumerate(zip(wavs, labels)):
        wav_t = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).float().to(args.device)
        length = wav_t.shape[-1]
        right_pad = math.ceil(length / hop_dac) * hop_dac - length
        if right_pad > 0:
            wav_t = torch.nn.functional.pad(wav_t, (0, int(right_pad)))
        with torch.no_grad():
            z = model.encoder(wav_t)
        z_np = z[0].float().cpu().numpy()
        T = z_np.shape[1]
        step = max(1, T // N_FRAMES_PER_UTT)
        for t in range(0, T, step):
            all_e.append(z_np[:, t])
            all_l.append(lang)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(wavs)}")
    raw_embeddings["DAC"] = (np.stack(all_e), all_l, None)
    del model; torch.cuda.empty_cache(); gc.collect()

    print("\n=== STEP 4: Fair protocol + plot ===")
    normalized = {}
    for name, (embs, lbl, _) in raw_embeddings.items():
        scaler = StandardScaler()
        embs_norm = scaler.fit_transform(embs)
        pca = PCA(n_components=min(50, embs.shape[1]), random_state=RANDOM_SEED)
        embs_pca = pca.fit_transform(embs_norm)
        normalized[name] = (embs_pca, lbl, None)
        print(f"  {name}: {embs.shape[1]}d -> {embs_pca.shape[1]}d, var={pca.explained_variance_ratio_.sum():.3f}")

    # Equalize frame counts
    min_per_lang = min((np.array(lbl) == lang).sum()
                       for _, lbl, _ in normalized.values()
                       for lang in set(list(normalized.values())[0][1]))
    min_per_lang = int(min_per_lang)
    rng = np.random.RandomState(RANDOM_SEED)
    equalized = {}
    for name, (embs, lbl, _) in normalized.items():
        arr = np.array(lbl)
        indices = []
        for lang in np.unique(arr):
            lang_idx = np.where(arr == lang)[0]
            chosen = rng.choice(lang_idx, size=min_per_lang, replace=False)
            indices.extend(chosen)
        indices = sorted(indices)
        equalized[name] = (embs[indices], [lbl[i] for i in indices])
        print(f"  {name}: {len(embs)} -> {len(indices)}")

    # 5-model t-SNE comparison plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    print("\n[plot] 5-model t-SNE comparison...")
    model_order = ["JEPA-EMA (v9)", "JEPA-SIGReg (25Hz)", "Mimi", "EnCodec", "DAC"]
    n = len(model_order)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    for mi, name in enumerate(model_order):
        if name not in equalized:
            continue
        embs, lbl = equalized[name]
        # Subsample for viz
        per_lang = min(5000, len(embs) // 6)
        arr = np.array(lbl)
        idx = []
        for lang in np.unique(arr):
            lang_idx = np.where(arr == lang)[0]
            idx.extend(rng.choice(lang_idx, min(per_lang, len(lang_idx)), replace=False))
        rng.shuffle(idx)
        embs_sub = embs[idx]
        labels_sub = [lbl[i] for i in idx]

        tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED, max_iter=1000)
        coords = tsne.fit_transform(embs_sub)
        ax = axes[mi]
        for lang in LANG_COLORS:
            mask = np.array([l == lang for l in labels_sub])
            pts = coords[mask]
            if len(pts) > 0:
                ax.scatter(pts[:, 0], pts[:, 1], c=LANG_COLORS[lang],
                           s=2, alpha=0.25, label=lang, rasterized=True)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        if mi == 0:
            ax.legend(fontsize=7, markerscale=4, loc="upper left")

    fig.suptitle("Cross-Lingual Frame Embeddings: 5-Codec Comparison",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "5_codec_tsne.png"),
                dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, "5_codec_tsne.pdf"),
                bbox_inches="tight")
    plt.close()
    print(f"  Saved 5_codec_tsne.png + .pdf")

    # Classifiers
    print("\n[clf] Language separability with Mimi added...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    clf_results = {}
    for name in model_order:
        if name not in equalized:
            continue
        embs, lbl = equalized[name]
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

    with open(os.path.join(OUTPUT_DIR, "5_codec_classifier_results.json"), "w") as f:
        json.dump(clf_results, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\nCOMPLETE in {elapsed/60:.1f} min")
    for fn in sorted(os.listdir(OUTPUT_DIR)):
        sz = os.path.getsize(os.path.join(OUTPUT_DIR, fn))
        print(f"  {fn} ({sz/1024:.1f} KB)")


if __name__ == "__main__":
    main()
