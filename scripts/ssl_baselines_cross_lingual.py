"""SSL baseline comparisons for cross-lingual emergence claim.

Binds out the multilingual-training confound by comparing our JEPA encoder
against PUBLISHED SSL encoders trained with different data regimes:

  English-only SSL:
    - wav2vec2-base (LibriSpeech 960h English) — facebook/wav2vec2-base
    - HuBERT-base-ls960 (LibriSpeech English) — facebook/hubert-base-ls960

  Multilingual SSL:
    - wav2vec2-XLS-R-300m (128 langs, 436K hours) — facebook/wav2vec2-xls-r-300m
    - mHuBERT-147 (147 langs) — utter-project/mHuBERT-147

The question: does our English-only JEPA produce MORE language-separable frame
embeddings than these baselines? If yes -> JEPA's predictive objective is the
reason, not just data. If no -> the claim weakens to "any SSL beats codec encoders".

Usage:
    cd ~/koe
    CUDA_VISIBLE_DEVICES=0 HF_TOKEN=... .venv/bin/python scripts/ssl_baselines_cross_lingual.py
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

SAMPLE_RATE_24K = 24000
SAMPLE_RATE_16K = 16000  # all SSL models work at 16 kHz
FLEURS_DIR = "/local_data/data/fleurs_full"
OUTPUT_DIR = "/local_data/analysis/ssl_baselines"

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

SSL_MODELS = {
    # English-only SSL
    "wav2vec2-base (en)": {
        "hf_id": "facebook/wav2vec2-base",
        "type": "wav2vec2",
        "multilingual": False,
        "training_data": "LibriSpeech 960h (English)",
    },
    "HuBERT-base (en)": {
        "hf_id": "facebook/hubert-base-ls960",
        "type": "hubert",
        "multilingual": False,
        "training_data": "LibriSpeech 960h (English)",
    },
    # Multilingual SSL
    "XLS-R-300m": {
        "hf_id": "facebook/wav2vec2-xls-r-300m",
        "type": "wav2vec2",
        "multilingual": True,
        "training_data": "VoxPopuli + MLS + CommonVoice + VoxLingua + BABEL (128 langs, 436K hrs)",
    },
    "mHuBERT-147": {
        "hf_id": "utter-project/mHuBERT-147",
        "type": "hubert",
        "multilingual": True,
        "training_data": "147 languages",
    },
}


def load_audio_16k():
    """Load same FLEURS subset as prior analyses but resample to 16 kHz for SSL models."""
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
                if wav.ndim > 1:
                    wav = wav.mean(axis=-1)
                # Resample to 16 kHz for SSL models
                if sr != SAMPLE_RATE_16K:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE_16K)
                if len(wav) < SAMPLE_RATE_16K:
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
        # Use SAME seed and selection as prior analyses for consistent utterances
        indices = rng.choice(len(all_wavs), size=min_count, replace=False)
        for idx in sorted(indices):
            wavs.append(all_wavs[idx])
            labels.append(LANG_LABELS[lang_key])
    print(f"[data] Loaded {len(wavs)} utterances @ 16 kHz ({min_count}/lang)")
    return wavs, labels


def extract_ssl_embeddings(model_name, hf_id, model_type, wavs, labels, device):
    """Extract frame-level embeddings from a Hugging Face SSL model."""
    from transformers import AutoModel, AutoFeatureExtractor

    print(f"[{model_name}] Loading {hf_id}...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id, torch_dtype=torch.bfloat16).to(device).eval()

    # Get model hidden state dim and frame rate
    dummy = torch.zeros(1, SAMPLE_RATE_16K, dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        out = model(dummy, output_hidden_states=False)
    if hasattr(out, "last_hidden_state"):
        hidden = out.last_hidden_state
    else:
        hidden = out
    n_frames_per_sec = hidden.shape[1] / 1.0
    hidden_dim = hidden.shape[-1]
    print(f"  dim={hidden_dim}, frame rate≈{n_frames_per_sec:.1f} Hz")

    all_embs, all_labels = [], []
    for i, (wav, lang) in enumerate(zip(wavs, labels)):
        inputs = feature_extractor(wav, sampling_rate=SAMPLE_RATE_16K,
                                   return_tensors="pt")
        input_values = inputs.input_values.to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            outputs = model(input_values, output_hidden_states=False)
        # Pick last hidden state — final layer's frame embeddings
        z = outputs.last_hidden_state[0].float().cpu().numpy()  # [T, D]
        z = z.T  # [D, T] to match codec format
        T = z.shape[1]
        step = max(1, T // N_FRAMES_PER_UTT)
        for t in range(0, T, step):
            all_embs.append(z[:, t])
            all_labels.append(lang)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(wavs)}")

    del model, feature_extractor
    torch.cuda.empty_cache()
    gc.collect()
    return np.stack(all_embs), all_labels, hidden_dim


def fair_normalize(embeddings_dict, pca_dim=50):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    normalized = {}
    for name, (embs, labels) in embeddings_dict.items():
        scaler = StandardScaler()
        embs_norm = scaler.fit_transform(embs)
        actual_dim = min(pca_dim, embs.shape[1], embs.shape[0])
        pca = PCA(n_components=actual_dim, random_state=RANDOM_SEED)
        embs_pca = pca.fit_transform(embs_norm)
        var = pca.explained_variance_ratio_.sum()
        print(f"[norm] {name}: {embs.shape[1]}d -> {embs_pca.shape[1]}d, var={var:.3f}")
        normalized[name] = (embs_pca, labels)
    return normalized


def equalize_frame_counts(embeddings_dict):
    rng = np.random.RandomState(RANDOM_SEED)
    min_per_lang = float("inf")
    for name, (embs, labels) in embeddings_dict.items():
        arr = np.array(labels)
        for lang in np.unique(arr):
            min_per_lang = min(min_per_lang, (arr == lang).sum())
    min_per_lang = int(min_per_lang)
    equalized = {}
    for name, (embs, labels) in embeddings_dict.items():
        arr = np.array(labels)
        indices = []
        for lang in np.unique(arr):
            li = np.where(arr == lang)[0]
            chosen = rng.choice(li, size=min_per_lang, replace=False)
            indices.extend(chosen)
        indices = sorted(indices)
        equalized[name] = (embs[indices], [labels[i] for i in indices])
        print(f"[eq] {name}: {len(embs)} -> {len(indices)}")
    return equalized


def language_classifiers(embeddings_dict):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    results = {}
    for name, (embs, labels) in embeddings_dict.items():
        arr = np.array(labels)
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
            print(f"  {name:30s} | {cn:10s}: {scores.mean():.3f} +/- {scores.std():.3f}")
        results[name] = model_res
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--include_our_jepa", action="store_true",
                        help="Also re-extract our JEPA + EnCodec/DAC at 16 kHz for combined plot")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_start = time.time()

    # HF auth
    if "HF_TOKEN" in os.environ:
        from huggingface_hub import login
        login(token=os.environ["HF_TOKEN"])

    print("=" * 60)
    print("SSL Baselines Cross-Lingual Comparison")
    print("=" * 60)

    wavs, labels = load_audio_16k()

    raw_embeddings = {}
    model_metadata = {}

    for i, (model_name, cfg) in enumerate(SSL_MODELS.items()):
        print(f"\n[{i+1}/{len(SSL_MODELS)}] {model_name} ({cfg['hf_id']})")
        print(f"  Multilingual: {cfg['multilingual']}")
        print(f"  Training: {cfg['training_data']}")
        try:
            embs, lbls, dim = extract_ssl_embeddings(
                model_name, cfg["hf_id"], cfg["type"], wavs, labels, args.device
            )
            raw_embeddings[model_name] = (embs, lbls)
            model_metadata[model_name] = {**cfg, "n_frames": len(embs), "dim": dim}
            print(f"  Got {embs.shape}")
        except Exception as e:
            print(f"  FAILED: {e}")
            model_metadata[model_name] = {**cfg, "error": str(e)}

    # Save raw stats
    raw_stats = {}
    for name, (embs, lbls) in raw_embeddings.items():
        cov = np.cov(embs.T)
        eigvals = np.linalg.eigvalsh(cov)
        p = eigvals / eigvals.sum()
        erank = float(np.exp(-np.sum(p * np.log(p + 1e-10))))
        raw_stats[name] = {
            "n_frames": len(embs), "n_dims": embs.shape[1],
            "effective_rank": erank,
            "per_dim_std": float(embs.std(axis=0).mean()),
        }
        print(f"  {name}: erank={erank:.1f}")
    with open(os.path.join(OUTPUT_DIR, "ssl_raw_stats.json"), "w") as f:
        json.dump(raw_stats, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "ssl_model_metadata.json"), "w") as f:
        json.dump(model_metadata, f, indent=2)

    print("\n=== Fair protocol ===")
    normalized = fair_normalize(raw_embeddings, pca_dim=50)
    equalized = equalize_frame_counts(normalized)

    print("\n=== Language classifiers ===")
    clf_results = language_classifiers(equalized)
    with open(os.path.join(OUTPUT_DIR, "ssl_classifier_results.json"), "w") as f:
        json.dump(clf_results, f, indent=2)

    # Cluster purity (NMI/ARI)
    print("\n=== Cluster purity (k=64) ===")
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
    cluster_results = {}
    for name, (embs, lbls) in equalized.items():
        arr = np.array(lbls)
        km = MiniBatchKMeans(n_clusters=64, random_state=RANDOM_SEED, n_init=10, batch_size=2048)
        cids = km.fit_predict(embs)
        nmi = normalized_mutual_info_score(arr, cids)
        ari = adjusted_rand_score(arr, cids)
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(len(embs), min(10000, len(embs)), replace=False)
        sil = silhouette_score(embs[idx], cids[idx])
        cluster_results[name] = {"NMI": float(nmi), "ARI": float(ari), "silhouette": float(sil)}
        print(f"  {name:30s}: NMI={nmi:.3f}, ARI={ari:.3f}, sil={sil:.3f}")
    with open(os.path.join(OUTPUT_DIR, "ssl_cluster_results.json"), "w") as f:
        json.dump(cluster_results, f, indent=2)

    # t-SNE plot
    print("\n=== t-SNE plot ===")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    model_order = list(raw_embeddings.keys())
    n = len(model_order)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]
    rng = np.random.RandomState(RANDOM_SEED)
    for mi, name in enumerate(model_order):
        embs, lbls = equalized[name]
        arr = np.array(lbls)
        unique = np.unique(arr)
        per_lang = min(5000, len(embs) // len(unique))
        idx = []
        for lang in unique:
            li = np.where(arr == lang)[0]
            idx.extend(rng.choice(li, min(per_lang, len(li)), replace=False))
        rng.shuffle(idx)
        embs_sub = embs[idx]
        labels_sub = [lbls[i] for i in idx]
        tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED, max_iter=1000)
        coords = tsne.fit_transform(embs_sub)
        ax = axes[mi]
        for lang in LANG_COLORS:
            mask = np.array([l == lang for l in labels_sub])
            pts = coords[mask]
            if len(pts) > 0:
                ax.scatter(pts[:, 0], pts[:, 1], c=LANG_COLORS[lang],
                           s=2, alpha=0.25, label=lang, rasterized=True)
        is_multi = SSL_MODELS[name]["multilingual"]
        marker = "[multilingual]" if is_multi else "[English-only]"
        ax.set_title(f"{name}\n{marker}", fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        if mi == 0:
            ax.legend(fontsize=7, markerscale=4, loc="upper left")

    fig.suptitle("SSL Encoder Cross-Lingual Frame Embeddings: English-only vs Multilingual Training",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ssl_baselines_tsne.png"),
                dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, "ssl_baselines_tsne.pdf"),
                bbox_inches="tight")
    plt.close()
    print(f"  Saved ssl_baselines_tsne.png + .pdf")

    # Summary comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    names = list(clf_results.keys())
    x = np.arange(len(names))
    width = 0.25
    colors = ["#4472C4", "#ED7D31", "#70AD47"]
    for i, clf_name in enumerate(["LogReg", "LinearSVM", "5-NN"]):
        means = [clf_results[m][clf_name]["mean"] for m in names]
        stds = [clf_results[m][clf_name]["std"] for m in names]
        ax.bar(x + i * width - width, means, width, yerr=stds,
               label=clf_name, capsize=2, color=colors[i])
    # Color x labels by multilingual status
    short_names = []
    for n in names:
        is_multi = SSL_MODELS[n]["multilingual"]
        tag = " [M]" if is_multi else " [E]"
        short_names.append(n + tag)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=15, ha="right", fontsize=9)
    ax.axhline(y=1/6, color="gray", ls="--", alpha=0.5, label="Chance (6 langs)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Language Separability: SSL Baselines (E=English-only, M=Multilingual)", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ssl_classifier_bars.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved ssl_classifier_bars.png")

    elapsed = time.time() - t_start
    print(f"\nCOMPLETE in {elapsed/60:.1f} min")
    for fn in sorted(os.listdir(OUTPUT_DIR)):
        sz = os.path.getsize(os.path.join(OUTPUT_DIR, fn))
        print(f"  {fn} ({sz/1024:.1f} KB)")


if __name__ == "__main__":
    main()
