"""Speech-only embedding analysis — filters out silence frames.

Addresses reviewer concerns:
1. Remove silence/low-energy frames, rerun all metrics
2. Fix F0 reliability: mask when RMS < threshold or confidence low
3. Use "coarse acoustic categories" language
4. Generate compact 4-panel paper figure

Loads raw embeddings + audio chunks saved from the comprehensive analysis,
filters to speech-only frames, reruns classifiers/clustering/acoustic analysis.

Usage:
    cd ~/koe
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/speech_only_analysis.py
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
OUTPUT_DIR = "/local_data/analysis/speech_only_embeddings"
PREV_OUTPUT = "/local_data/analysis/comprehensive_embeddings"

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
ACOUSTIC_CONTEXT_MS = 500
RMS_SILENCE_THRESHOLD = 0.005  # frames below this are silence
RANDOM_SEED = 42


# ── Re-extract with audio chunks for all models ─────────────────────

def load_audio(n_per_lang=None):
    wavs_by_lang = {}
    for lang_key in LANGUAGES:
        lang_dir = os.path.join(FLEURS_DIR, lang_key)
        if not os.path.isdir(lang_dir):
            continue
        files = sorted([f for f in os.listdir(lang_dir) if f.endswith(".wav")])
        if n_per_lang:
            files = files[:n_per_lang]
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
        if lang_key not in wavs_by_lang:
            continue
        all_wavs = wavs_by_lang[lang_key]
        indices = rng.choice(len(all_wavs), size=min_count, replace=False)
        for idx in sorted(indices):
            wavs.append(all_wavs[idx])
            labels.append(LANG_LABELS[lang_key])
    print(f"[data] Loaded {len(wavs)} utterances ({min_count}/lang)")
    return wavs, labels


def _extract_frames(z, wav, hop, lang, n_per_utt):
    T = z.shape[1]
    step = max(1, T // n_per_utt)
    context_samples = int(ACOUSTIC_CONTEXT_MS / 1000.0 * SAMPLE_RATE)
    embs, frame_labels, audio_contexts = [], [], []
    for t in range(0, T, step):
        embs.append(z[:, t])
        frame_labels.append(lang)
        center = t * hop + hop // 2
        start = max(0, center - context_samples // 2)
        end = min(len(wav), center + context_samples // 2)
        audio_contexts.append(wav[start:end])
    return embs, frame_labels, audio_contexts


def extract_all_models(wavs, labels, device):
    """Extract embeddings with audio chunks for all 4 models."""
    results = {}

    # 1. JEPA-EMA v9
    print("\n[1/4] JEPA-EMA v9...")
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
    all_e, all_l, all_a = [], [], []
    for i, (wav, lang) in enumerate(zip(wavs, labels)):
        rem = len(wav) % hop
        wav_pad = np.pad(wav, (0, hop - rem)) if rem else wav
        wav_t = torch.from_numpy(wav_pad).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            z_e = model.encoder.encode(wav_t)
        z = z_e[0].float().cpu().numpy()
        e, fl, ac = _extract_frames(z, wav, hop, lang, N_FRAMES_PER_UTT)
        all_e.extend(e); all_l.extend(fl); all_a.extend(ac)
        if (i + 1) % 200 == 0: print(f"  {i+1}/{len(wavs)}")
    results["JEPA-EMA (v9)"] = (np.stack(all_e), all_l, all_a)
    print(f"  {len(all_e)} frames")
    del model; torch.cuda.empty_cache(); gc.collect()

    # 2. JEPA-SIGReg
    print("\n[2/4] JEPA-SIGReg 25Hz...")
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
    all_e, all_l, all_a = [], [], []
    for i, (wav, lang) in enumerate(zip(wavs, labels)):
        rem = len(wav) % hop
        wav_pad = np.pad(wav, (0, hop - rem)) if rem else wav
        wav_t = torch.from_numpy(wav_pad).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            z_e = encoder.encode(wav_t)
        z = z_e[0].float().cpu().numpy()
        e, fl, ac = _extract_frames(z, wav, hop, lang, N_FRAMES_PER_UTT)
        all_e.extend(e); all_l.extend(fl); all_a.extend(ac)
        if (i + 1) % 200 == 0: print(f"  {i+1}/{len(wavs)}")
    results["JEPA-SIGReg (25Hz)"] = (np.stack(all_e), all_l, all_a)
    print(f"  {len(all_e)} frames")
    del encoder; torch.cuda.empty_cache(); gc.collect()

    # 3. EnCodec
    print("\n[3/4] EnCodec...")
    from encodec import EncodecModel
    model = EncodecModel.encodec_model_24khz().to(device).eval()
    hop_ec = 320
    all_e, all_l, all_a = [], [], []
    for i, (wav, lang) in enumerate(zip(wavs, labels)):
        wav_t = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            z = model.encoder(wav_t)
        z_np = z[0].float().cpu().numpy()
        e, fl, ac = _extract_frames(z_np, wav, hop_ec, lang, N_FRAMES_PER_UTT)
        all_e.extend(e); all_l.extend(fl); all_a.extend(ac)
        if (i + 1) % 200 == 0: print(f"  {i+1}/{len(wavs)}")
    results["EnCodec"] = (np.stack(all_e), all_l, all_a)
    print(f"  {len(all_e)} frames")
    del model; torch.cuda.empty_cache(); gc.collect()

    # 4. DAC
    print("\n[4/4] DAC...")
    import dac
    model_path = dac.utils.download(model_type="24khz")
    model = dac.DAC.load(model_path).to(device).eval()
    hop_dac = int(np.prod(model.encoder_rates))
    all_e, all_l, all_a = [], [], []
    for i, (wav, lang) in enumerate(zip(wavs, labels)):
        wav_t = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).float().to(device)
        length = wav_t.shape[-1]
        right_pad = math.ceil(length / hop_dac) * hop_dac - length
        if right_pad > 0:
            wav_t = torch.nn.functional.pad(wav_t, (0, int(right_pad)))
        with torch.no_grad():
            z = model.encoder(wav_t)
        z_np = z[0].float().cpu().numpy()
        e, fl, ac = _extract_frames(z_np, wav, hop_dac, lang, N_FRAMES_PER_UTT)
        all_e.extend(e); all_l.extend(fl); all_a.extend(ac)
        if (i + 1) % 200 == 0: print(f"  {i+1}/{len(wavs)}")
    results["DAC"] = (np.stack(all_e), all_l, all_a)
    print(f"  {len(all_e)} frames")
    del model; torch.cuda.empty_cache(); gc.collect()

    return results


# ── Silence filtering ────────────────────────────────────────────────

def compute_rms(audio_chunks):
    """Compute RMS for each audio chunk."""
    return np.array([np.sqrt(np.mean(c ** 2)) if len(c) > 0 else 0.0
                     for c in audio_chunks])


def filter_speech_only(embeddings_dict, threshold=RMS_SILENCE_THRESHOLD):
    """Remove frames with RMS below threshold."""
    filtered = {}
    for name, (embs, frame_labels, audio_chunks) in embeddings_dict.items():
        rms = compute_rms(audio_chunks)
        speech_mask = rms >= threshold
        n_total = len(embs)
        n_speech = speech_mask.sum()
        filtered[name] = (
            embs[speech_mask],
            [frame_labels[i] for i in range(n_total) if speech_mask[i]],
            [audio_chunks[i] for i in range(n_total) if speech_mask[i]],
        )
        print(f"[filter] {name}: {n_total} -> {n_speech} speech frames "
              f"({n_speech/n_total:.1%}), removed {n_total - n_speech} silence")
    return filtered


# ── Fair protocol ────────────────────────────────────────────────────

def fair_normalize(embeddings_dict, pca_dim=50):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    normalized = {}
    for name, (embs, labels, audio) in embeddings_dict.items():
        scaler = StandardScaler()
        embs_norm = scaler.fit_transform(embs)
        actual_dim = min(pca_dim, embs.shape[1], embs.shape[0])
        pca = PCA(n_components=actual_dim, random_state=RANDOM_SEED)
        embs_pca = pca.fit_transform(embs_norm)
        var = pca.explained_variance_ratio_.sum()
        print(f"[norm] {name}: {embs.shape[1]}d -> {embs_pca.shape[1]}d, var={var:.3f}")
        normalized[name] = (embs_pca, labels, audio)
    return normalized


def equalize_frame_counts(embeddings_dict):
    rng = np.random.RandomState(RANDOM_SEED)
    min_per_lang = float("inf")
    for name, (embs, labels, _) in embeddings_dict.items():
        arr = np.array(labels)
        for lang in np.unique(arr):
            min_per_lang = min(min_per_lang, (arr == lang).sum())
    min_per_lang = int(min_per_lang)

    equalized = {}
    for name, (embs, labels, audio) in embeddings_dict.items():
        arr = np.array(labels)
        indices = []
        for lang in np.unique(arr):
            lang_idx = np.where(arr == lang)[0]
            chosen = rng.choice(lang_idx, size=min_per_lang, replace=False)
            indices.extend(chosen)
        indices = sorted(indices)
        equalized[name] = (
            embs[indices],
            [labels[i] for i in indices],
            [audio[i] for i in indices],
        )
        print(f"[eq] {name}: {len(embs)} -> {len(indices)} "
              f"({min_per_lang}/lang x {len(np.unique(arr))})")
    return equalized


# ── Analyses ─────────────────────────────────────────────────────────

def language_classifiers(embeddings_dict):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    n_langs = len(set(list(embeddings_dict.values())[0][1]))
    print(f"  Chance: {1/n_langs:.3f} ({n_langs} langs)")
    results = {}
    for name, (embs, labels, _) in embeddings_dict.items():
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
            print(f"  {name:25s} | {cn:10s}: {scores.mean():.3f} +/- {scores.std():.3f}")
        results[name] = model_res
    return results


def cluster_purity(embeddings_dict, k_values=(32, 64, 128)):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score

    results = {}
    for name, (embs, labels, _) in embeddings_dict.items():
        arr = np.array(labels)
        model_res = {}
        for k in k_values:
            km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10, batch_size=2048)
            cids = km.fit_predict(embs)
            nmi = normalized_mutual_info_score(arr, cids)
            ari = adjusted_rand_score(arr, cids)
            rng = np.random.RandomState(RANDOM_SEED)
            idx = rng.choice(len(embs), min(10000, len(embs)), replace=False)
            sil = silhouette_score(embs[idx], cids[idx])
            model_res[k] = {"NMI": float(nmi), "ARI": float(ari), "silhouette": float(sil)}
            print(f"  {name} | k={k}: NMI={nmi:.3f}, ARI={ari:.3f}, sil={sil:.3f}")
        results[name] = model_res
    return results


def cluster_consistency(embeddings_dict, k=64):
    from sklearn.cluster import MiniBatchKMeans
    from scipy.stats import entropy
    results = {}
    for name, (embs, labels, _) in embeddings_dict.items():
        arr = np.array(labels)
        langs = sorted(set(arr))
        n_langs = len(langs)
        km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10, batch_size=2048)
        cids = km.fit_predict(embs)
        max_ent = np.log(n_langs)
        dists, ents, balanced = [], [], 0
        for c in range(k):
            mask = cids == c
            if mask.sum() == 0: continue
            cl = arr[mask]
            d = {l: float((cl == l).sum()) / mask.sum() for l in langs}
            dists.append({"cluster": c, "size": int(mask.sum()), **d})
            e = entropy([d[l] for l in langs])
            ents.append(e)
            if e > 0.5 * max_ent: balanced += 1
        results[name] = {
            "k": k, "balanced_clusters": balanced, "balanced_ratio": balanced / k,
            "mean_entropy": float(np.mean(ents)), "normalized_entropy": float(np.mean(ents) / max_ent),
            "cluster_distributions": dists,
        }
        print(f"  {name}: {balanced}/{k} balanced, H/Hmax={np.mean(ents)/max_ent:.3f}")
    return results


def acoustic_features_speech_only(raw_embs, raw_labels, audio_chunks, k=64):
    """Acoustic features with reliable F0 (masked for low-energy frames)."""
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10, batch_size=2048)
    cids = km.fit_predict(raw_embs)

    cluster_features = []
    for c in range(k):
        mask = cids == c
        if mask.sum() < 5: continue
        chunks = [audio_chunks[i] for i in range(len(audio_chunks)) if mask[i]]

        rms_vals, zcr_vals, sc_vals, f0_vals, voiced_ratios = [], [], [], [], []
        for chunk in chunks[:300]:
            if len(chunk) < 512: continue
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            rms_vals.append(rms)
            zcr = float(np.mean(np.abs(np.diff(np.sign(chunk))) > 0))
            zcr_vals.append(zcr)
            fft = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
            freqs = np.fft.rfftfreq(len(chunk), 1.0 / SAMPLE_RATE)
            sc = float(np.sum(freqs * fft) / np.sum(fft)) if fft.sum() > 1e-10 else 0.0
            sc_vals.append(sc)
            # F0 — already speech-only so RMS is above silence threshold
            if len(chunk) >= 960:
                try:
                    f0, voiced, voiced_prob = librosa.pyin(
                        chunk, fmin=60, fmax=500, sr=SAMPLE_RATE, frame_length=1024)
                    # Only use frames with high confidence
                    confident = voiced_prob > 0.5
                    valid_f0 = f0[confident & ~np.isnan(f0)]
                    if len(valid_f0) > 2:
                        f0_vals.append(float(np.median(valid_f0)))
                    voiced_frames = voiced_prob > 0.5
                    voiced_ratios.append(float(np.mean(voiced_frames)))
                except Exception:
                    pass

        if not rms_vals: continue
        feat = {
            "cluster": c, "n_frames": int(mask.sum()),
            "rms_mean": float(np.mean(rms_vals)), "rms_std": float(np.std(rms_vals)),
            "zcr_mean": float(np.mean(zcr_vals)), "zcr_std": float(np.std(zcr_vals)),
            "spectral_centroid_mean": float(np.mean(sc_vals)),
            "spectral_centroid_std": float(np.std(sc_vals)),
        }
        if f0_vals:
            feat["f0_median"] = float(np.median(f0_vals))
            feat["f0_std"] = float(np.std(f0_vals))
        if voiced_ratios:
            feat["voiced_ratio"] = float(np.mean(voiced_ratios))
        cluster_features.append(feat)

    if not cluster_features: return None

    # Categorize using speech-relevant thresholds
    rms_values = [cf["rms_mean"] for cf in cluster_features]
    zcr_values = [cf["zcr_mean"] for cf in cluster_features]
    sc_values = [cf["spectral_centroid_mean"] for cf in cluster_features]

    zcr_p75 = np.percentile(zcr_values, 75)
    zcr_p25 = np.percentile(zcr_values, 25)
    sc_p60 = np.percentile(sc_values, 60)

    sc_p40 = np.percentile(sc_values, 40)
    rms_p30 = np.percentile(rms_values, 30)

    for cf in cluster_features:
        voiced = cf.get("voiced_ratio", None)
        sc = cf["spectral_centroid_mean"]
        zcr = cf["zcr_mean"]
        rms = cf["rms_mean"]

        if zcr > zcr_p75 and sc > sc_p60:
            cf["category"] = "frication"
        elif rms < rms_p30 and zcr > np.median(zcr_values):
            cf["category"] = "low-energy transition"
        elif sc < sc_p40 and zcr < zcr_p25:
            cf["category"] = "sonorant/vowel"
        elif voiced is not None and voiced > 0.55 and sc < sc_p60:
            cf["category"] = "voiced consonant"
        elif voiced is not None and voiced > 0.45 and zcr < np.median(zcr_values):
            cf["category"] = "nasal/approximant"
        elif zcr > np.median(zcr_values):
            cf["category"] = "unvoiced/plosive"
        else:
            cf["category"] = "mixed"

    cats = [cf["category"] for cf in cluster_features]
    for cat in sorted(set(cats)):
        print(f"  {cat}: {cats.count(cat)} clusters")
    return cluster_features


# ── Paper figure (4-panel) ───────────────────────────────────────────

def plot_paper_figure(eq_embeddings, clf_results, cluster_results,
                      consistency_results, acoustic_features, output_dir):
    """Compact 4-panel figure for the paper."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # ── Panel A: t-SNE comparison (2x2 grid within panel) ──
    gs_a = gs[0, 0].subgridspec(2, 2, hspace=0.15, wspace=0.1)
    model_names = list(eq_embeddings.keys())
    rng = np.random.RandomState(RANDOM_SEED)

    for mi, name in enumerate(model_names):
        ax = fig.add_subplot(gs_a[mi // 2, mi % 2])
        embs, labels, _ = eq_embeddings[name]
        arr = np.array(labels)
        unique = np.unique(arr)
        per_lang = min(3000, len(embs) // len(unique))
        idx = []
        for lang in unique:
            li = np.where(arr == lang)[0]
            idx.extend(rng.choice(li, min(per_lang, len(li)), replace=False))
        rng.shuffle(idx)
        embs_sub = embs[idx]
        labels_sub = [labels[i] for i in idx]

        tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED, max_iter=1000)
        coords = tsne.fit_transform(embs_sub)
        for lang in LANG_COLORS:
            mask = np.array([l == lang for l in labels_sub])
            pts = coords[mask]
            if len(pts) > 0:
                ax.scatter(pts[:, 0], pts[:, 1], c=LANG_COLORS[lang],
                           s=1, alpha=0.2, rasterized=True,
                           label=lang if mi == 0 else None)
        short_name = name.replace("JEPA-", "").replace(" (", "\n(")
        ax.set_title(short_name, fontsize=8, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        if mi == 0:
            ax.legend(fontsize=5, markerscale=4, loc="upper left",
                      handletextpad=0.1, borderpad=0.2)

    fig.text(0.25, 0.96, "(A) t-SNE: Speech-Only Frame Embeddings",
             ha="center", fontsize=10, fontweight="bold")

    # ── Panel B: Classifier accuracy ──
    ax_b = fig.add_subplot(gs[0, 1])
    clf_names = list(clf_results[model_names[0]].keys())
    x = np.arange(len(model_names))
    width = 0.22
    colors = ["#4472C4", "#ED7D31", "#70AD47"]
    short_names = [n.replace("JEPA-", "J-").replace(" (25Hz)", "\n25Hz").replace(" (v9)", "\nv9")
                   for n in model_names]
    for i, cn in enumerate(clf_names):
        means = [clf_results[m][cn]["mean"] for m in model_names]
        stds = [clf_results[m][cn]["std"] for m in model_names]
        ax_b.bar(x + i * width - width, means, width, yerr=stds,
                 label=cn, capsize=2, color=colors[i])
    ax_b.axhline(y=1/6, color="gray", ls="--", alpha=0.5, lw=0.8)
    ax_b.set_ylabel("Accuracy", fontsize=9)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(short_names, fontsize=7)
    ax_b.legend(fontsize=7)
    ax_b.set_ylim(0, 1.05)
    ax_b.set_title("(B) Language Separability\n(lower = more shared)", fontsize=10, fontweight="bold")

    # ── Panel C: NMI + silhouette ──
    ax_c = fig.add_subplot(gs[1, 0])
    k_values = sorted(cluster_results[model_names[0]].keys())
    x = np.arange(len(model_names))
    width = 0.35
    k64 = 64
    nmis = [cluster_results[m][k64]["NMI"] for m in model_names]
    sils = [cluster_results[m][k64]["silhouette"] for m in model_names]
    bars1 = ax_c.bar(x - width/2, nmis, width, label="NMI (cluster, lang)", color="#4472C4")
    bars2 = ax_c.bar(x + width/2, sils, width, label="Silhouette", color="#ED7D31")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(short_names, fontsize=7)
    ax_c.legend(fontsize=7)
    ax_c.set_ylabel("Score", fontsize=9)
    ax_c.set_title("(C) Cluster Quality (k=64)\nNMI: lang correlation | Sil: cluster tightness",
                    fontsize=10, fontweight="bold")

    # Add cluster consistency text
    for mi, name in enumerate(model_names):
        cr = consistency_results[name]
        ax_c.text(mi, max(nmis[mi], sils[mi]) + 0.01,
                  f"H/Hmax={cr['normalized_entropy']:.2f}",
                  ha="center", fontsize=6, color="gray")

    # ── Panel D: Acoustic feature table (speech-only, top clusters) ──
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis("off")

    if acoustic_features:
        top = sorted(acoustic_features, key=lambda x: -x["n_frames"])[:12]
        headers = ["#", "N", "RMS", "ZCR", "SC(Hz)", "F0", "V%", "Category"]
        rows = []
        for cf in top:
            f0_str = f"{cf['f0_median']:.0f}" if "f0_median" in cf else "-"
            v_str = f"{cf['voiced_ratio']:.0%}" if "voiced_ratio" in cf else "-"
            rows.append([
                str(cf["cluster"]), str(cf["n_frames"]),
                f"{cf['rms_mean']:.3f}", f"{cf['zcr_mean']:.3f}",
                f"{cf['spectral_centroid_mean']:.0f}",
                f0_str, v_str, cf.get("category", "?"),
            ])
        table = ax_d.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.3)
        for j in range(len(headers)):
            table[0, j].set_facecolor("#4472C4")
            table[0, j].set_text_props(color="white", fontweight="bold", fontsize=7)
        cat_colors = {
            "sonorant/vowel": "#90EE90", "frication": "#FFD700",
            "voiced consonant": "#87CEEB", "unvoiced/mixed": "#FFB6C1",
            "low-energy transition": "#E8E8E8",
        }
        for r in range(len(rows)):
            cat = rows[r][-1]
            if cat in cat_colors:
                table[r + 1, len(headers) - 1].set_facecolor(cat_colors[cat])
    ax_d.set_title("(D) Coarse Acoustic Categories\n(speech-only, k=64, top 12)",
                    fontsize=10, fontweight="bold")

    plt.savefig(os.path.join(output_dir, "paper_figure_4panel.png"),
                dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "paper_figure_4panel.pdf"),
                bbox_inches="tight")
    plt.close()
    print(f"  Saved paper_figure_4panel.png + .pdf")


# ── Standalone plots ─────────────────────────────────────────────────

def plot_tsne_comparison(eq_embeddings, output_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    model_names = list(eq_embeddings.keys())
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1: axes = [axes]
    rng = np.random.RandomState(RANDOM_SEED)

    for mi, name in enumerate(model_names):
        embs, labels, _ = eq_embeddings[name]
        arr = np.array(labels)
        unique = np.unique(arr)
        per_lang = min(5000, len(embs) // len(unique))
        idx = []
        for lang in unique:
            li = np.where(arr == lang)[0]
            idx.extend(rng.choice(li, min(per_lang, len(li)), replace=False))
        rng.shuffle(idx)

        tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED, max_iter=1000)
        coords = tsne.fit_transform(embs[idx])
        labels_sub = [labels[i] for i in idx]
        ax = axes[mi]
        for lang in LANG_COLORS:
            mask = np.array([l == lang for l in labels_sub])
            pts = coords[mask]
            if len(pts) > 0:
                ax.scatter(pts[:, 0], pts[:, 1], c=LANG_COLORS[lang],
                           s=2, alpha=0.25, label=lang, rasterized=True)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        if mi == 0:
            ax.legend(fontsize=7, markerscale=4, loc="upper left")

    fig.suptitle("Speech-Only Frame Embeddings (silence removed, RMS > 0.005)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "speech_only_tsne.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved speech_only_tsne.png")


def compute_utterance_embeddings(wavs, labels, device):
    """Extract mean-pooled utterance embeddings for all models."""
    import matplotlib; matplotlib.use("Agg")

    results = {}

    # JEPA v9
    print("  [utt] JEPA-EMA v9...")
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
    utt_embs = []
    for wav in wavs:
        rem = len(wav) % hop
        wav_pad = np.pad(wav, (0, hop - rem)) if rem else wav
        wav_t = torch.from_numpy(wav_pad).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            z_e = model.encoder.encode(wav_t)
        utt_embs.append(z_e[0].float().cpu().numpy().mean(axis=1))
    results["JEPA-EMA (v9)"] = np.stack(utt_embs)
    del model; torch.cuda.empty_cache(); gc.collect()

    # JEPA SIGReg
    print("  [utt] JEPA-SIGReg...")
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
    utt_embs = []
    for wav in wavs:
        rem = len(wav) % hop
        wav_pad = np.pad(wav, (0, hop - rem)) if rem else wav
        wav_t = torch.from_numpy(wav_pad).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            z_e = encoder.encode(wav_t)
        utt_embs.append(z_e[0].float().cpu().numpy().mean(axis=1))
    results["JEPA-SIGReg (25Hz)"] = np.stack(utt_embs)
    del encoder; torch.cuda.empty_cache(); gc.collect()

    # EnCodec
    print("  [utt] EnCodec...")
    from encodec import EncodecModel
    model = EncodecModel.encodec_model_24khz().to(device).eval()
    utt_embs = []
    for wav in wavs:
        wav_t = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            z = model.encoder(wav_t)
        utt_embs.append(z[0].float().cpu().numpy().mean(axis=1))
    results["EnCodec"] = np.stack(utt_embs)
    del model; torch.cuda.empty_cache(); gc.collect()

    # DAC
    print("  [utt] DAC...")
    import dac
    model_path = dac.utils.download(model_type="24khz")
    model = dac.DAC.load(model_path).to(device).eval()
    hop_dac = int(np.prod(model.encoder_rates))
    utt_embs = []
    for wav in wavs:
        wav_t = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).float().to(device)
        length = wav_t.shape[-1]
        right_pad = math.ceil(length / hop_dac) * hop_dac - length
        if right_pad > 0:
            wav_t = torch.nn.functional.pad(wav_t, (0, int(right_pad)))
        with torch.no_grad():
            z = model.encoder(wav_t)
        utt_embs.append(z[0].float().cpu().numpy().mean(axis=1))
    results["DAC"] = np.stack(utt_embs)
    del model; torch.cuda.empty_cache(); gc.collect()

    return results


def plot_utterance_level(utt_embeddings, labels, output_dir):
    """t-SNE of utterance-level (mean-pooled) embeddings."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    model_names = list(utt_embeddings.keys())
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1: axes = [axes]

    for mi, name in enumerate(model_names):
        embs = StandardScaler().fit_transform(utt_embeddings[name])
        perp = min(30, len(embs) - 1)
        tsne = TSNE(n_components=2, perplexity=perp, random_state=RANDOM_SEED, max_iter=1000)
        coords = tsne.fit_transform(embs)
        ax = axes[mi]
        for lang in LANG_COLORS:
            mask = np.array([l == lang for l in labels])
            pts = coords[mask]
            if len(pts) > 0:
                ax.scatter(pts[:, 0], pts[:, 1], c=LANG_COLORS[lang],
                           s=15, alpha=0.5, label=lang, edgecolors="white", linewidth=0.3)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        if mi == 0:
            ax.legend(fontsize=7, markerscale=2, loc="upper left")

    fig.suptitle("Utterance-Level Embeddings (mean-pooled)\nLanguage separation emerges at utterance level",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "utterance_level_tsne.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved utterance_level_tsne.png")


def plot_all_vs_speech_comparison(all_results, speech_results, output_dir):
    """Side-by-side comparison: all frames vs speech-only."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_names = list(all_results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (title, results) in zip(axes, [("All Frames", all_results),
                                            ("Speech-Only", speech_results)]):
        clf_names = list(results[model_names[0]].keys())
        x = np.arange(len(model_names))
        width = 0.22
        colors = ["#4472C4", "#ED7D31", "#70AD47"]
        for i, cn in enumerate(clf_names):
            means = [results[m][cn]["mean"] for m in model_names]
            stds = [results[m][cn]["std"] for m in model_names]
            ax.bar(x + i * width - width, means, width, yerr=stds,
                   label=cn, capsize=2, color=colors[i])
        ax.axhline(y=1/6, color="gray", ls="--", alpha=0.5)
        ax.set_ylabel("Accuracy")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

    fig.suptitle("Language Separability: All Frames vs Speech-Only", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_vs_speech_classifiers.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved all_vs_speech_classifiers.png")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_per_lang", type=int, default=None)
    args = parser.parse_args()

    device = args.device
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_start = time.time()

    # Load previous all-frames classifier results for comparison
    prev_clf = None
    prev_clf_path = os.path.join(PREV_OUTPUT, "classifier_results.json")
    if os.path.exists(prev_clf_path):
        with open(prev_clf_path) as f:
            prev_clf = json.load(f)
        print(f"[prev] Loaded all-frames classifier results from {prev_clf_path}")

    # Extract embeddings with audio chunks for ALL models
    print("\n" + "=" * 60)
    print("STEP 1: Extract embeddings with audio chunks")
    print("=" * 60)
    wavs, labels = load_audio(n_per_lang=args.n_per_lang)
    raw_embeddings = extract_all_models(wavs, labels, device)

    # Filter to speech-only
    print("\n" + "=" * 60)
    print("STEP 2: Filter silence (RMS < 0.005)")
    print("=" * 60)
    speech_embeddings = filter_speech_only(raw_embeddings)

    # Normalize + equalize
    print("\n" + "=" * 60)
    print("STEP 3: Normalize + equalize (speech-only)")
    print("=" * 60)
    norm_speech = fair_normalize(speech_embeddings, pca_dim=50)
    eq_speech = equalize_frame_counts(norm_speech)

    # Classifiers
    print("\n" + "=" * 60)
    print("STEP 4: Language classifiers (speech-only)")
    print("=" * 60)
    clf_results = language_classifiers(eq_speech)
    with open(os.path.join(OUTPUT_DIR, "speech_only_classifier_results.json"), "w") as f:
        json.dump(clf_results, f, indent=2)

    # Cluster purity
    print("\n" + "=" * 60)
    print("STEP 5: Cluster purity (speech-only)")
    print("=" * 60)
    cluster_results = cluster_purity(eq_speech)
    with open(os.path.join(OUTPUT_DIR, "speech_only_cluster_purity.json"), "w") as f:
        json.dump(cluster_results, f, indent=2)

    # Cluster consistency
    print("\n" + "=" * 60)
    print("STEP 6: Cluster consistency (speech-only)")
    print("=" * 60)
    consistency = cluster_consistency(eq_speech, k=64)
    cons_save = {k: {kk: vv for kk, vv in v.items() if kk != "cluster_distributions"}
                 for k, v in consistency.items()}
    with open(os.path.join(OUTPUT_DIR, "speech_only_consistency.json"), "w") as f:
        json.dump(cons_save, f, indent=2)

    # Acoustic features (on raw speech-only v9)
    print("\n" + "=" * 60)
    print("STEP 7: Acoustic features (speech-only, reliable F0)")
    print("=" * 60)
    v9_speech = speech_embeddings["JEPA-EMA (v9)"]
    acoustic = acoustic_features_speech_only(v9_speech[0], v9_speech[1], v9_speech[2], k=64)
    if acoustic:
        with open(os.path.join(OUTPUT_DIR, "speech_only_acoustic.json"), "w") as f:
            json.dump(acoustic, f, indent=2)

    # Plots
    print("\n" + "=" * 60)
    print("STEP 8: Plots")
    print("=" * 60)

    print("\n[8a] t-SNE comparison (speech-only)...")
    plot_tsne_comparison(eq_speech, OUTPUT_DIR)

    print("\n[8b] Paper 4-panel figure...")
    plot_paper_figure(eq_speech, clf_results, cluster_results,
                      consistency, acoustic, OUTPUT_DIR)

    if prev_clf:
        print("\n[8c] All vs speech comparison...")
        plot_all_vs_speech_comparison(prev_clf, clf_results, OUTPUT_DIR)

    # Utterance-level analysis
    print("\n" + "=" * 60)
    print("STEP 9: Utterance-level embeddings (mean-pooled)")
    print("=" * 60)
    utt_embeddings = compute_utterance_embeddings(wavs, labels, device)
    print("\n[9a] Utterance-level t-SNE...")
    plot_utterance_level(utt_embeddings, labels, OUTPUT_DIR)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"COMPLETE in {elapsed/60:.1f} minutes")
    print(f"{'=' * 60}")
    for fn in sorted(os.listdir(OUTPUT_DIR)):
        sz = os.path.getsize(os.path.join(OUTPUT_DIR, fn))
        print(f"  {fn} ({sz/1024:.1f} KB)")


if __name__ == "__main__":
    main()
