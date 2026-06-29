"""Comprehensive cross-lingual embedding analysis for paper.

Implements all advisor-recommended analyses:
1. Language separability classifiers (logistic regression, linear SVM, 5-NN)
2. Cluster purity metrics (NMI, ARI, silhouette at k=32,64,128)
3. Cluster consistency across languages (per-cluster language distribution)
4. Phonetic/acoustic interpretation (RMS, ZCR, F0, spectral centroid per cluster)
5. Visualization robustness (t-SNE multiple seeds/perps, UMAP, PCA, 3D)
6. Fair comparison protocol (normalize, PCA to 50 dims, same settings)

Models: JEPA-EMA v9, JEPA-SIGReg 25Hz, EnCodec, DAC

Usage:
    cd ~/koe
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/comprehensive_embedding_analysis.py
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
FLEURS_DIR = "/local_data/data/fleurs_full"
OUTPUT_DIR = "/local_data/analysis/comprehensive_embeddings"

LANGUAGES = ["en_us", "hi_in", "ja_jp", "de_de", "fr_fr", "cmn_hans_cn"]
FLEURS_CODES = {
    "en_us": "en_us", "hi_in": "hi_in", "ja_jp": "ja_jp",
    "de_de": "de_de", "fr_fr": "fr_fr", "cmn_hans_cn": "cmn_hans_cn",
}
LANG_LABELS = {
    "en_us": "English", "hi_in": "Hindi", "ja_jp": "Japanese",
    "de_de": "German", "fr_fr": "French", "cmn_hans_cn": "Chinese",
}
LANG_COLORS = {
    "English": "#2196F3", "Hindi": "#4CAF50", "Japanese": "#FF9800",
    "German": "#9C27B0", "French": "#795548", "Chinese": "#F44336",
}

N_FRAMES_VIZ = 50000
N_FRAMES_PER_UTT = 20
ACOUSTIC_CONTEXT_MS = 500  # 500ms windows for acoustic features
RANDOM_SEED = 42


# ── Step 0: Download FLEURS ──────────────────────────────────────────

def download_fleurs():
    from datasets import load_dataset

    os.makedirs(FLEURS_DIR, exist_ok=True)

    for lang_key, fleurs_code in FLEURS_CODES.items():
        lang_dir = os.path.join(FLEURS_DIR, lang_key)
        existing = len([f for f in os.listdir(lang_dir) if f.endswith(".wav")]) if os.path.isdir(lang_dir) else 0
        if existing >= 100:
            print(f"[data] {lang_key}: {existing} files already exist, skipping")
            continue

        os.makedirs(lang_dir, exist_ok=True)
        count = 0
        for split in ["test", "validation"]:
            print(f"[data] Downloading FLEURS {fleurs_code} {split}...")
            try:
                ds = load_dataset("google/fleurs", fleurs_code, split=split,
                                  trust_remote_code=True)
                for i, item in enumerate(ds):
                    audio = item.get("audio", {})
                    if "array" not in audio:
                        continue
                    wav = np.array(audio["array"], dtype=np.float32)
                    sr = audio["sampling_rate"]
                    if sr != SAMPLE_RATE:
                        wav_t = torch.from_numpy(wav).unsqueeze(0)
                        wav_t = torchaudio.functional.resample(wav_t, sr, SAMPLE_RATE)
                        wav = wav_t.squeeze(0).numpy()
                    dur = len(wav) / SAMPLE_RATE
                    if dur < 2.0 or dur > 15.0:
                        continue
                    sf.write(os.path.join(lang_dir, f"{lang_key}_{split}_{i:04d}.wav"),
                             wav, SAMPLE_RATE)
                    count += 1
            except Exception as e:
                print(f"  ERROR: {e}")
        print(f"  {lang_key}: saved {count} files total")

    print("\n[data] Final counts:")
    for lang_key in LANGUAGES:
        lang_dir = os.path.join(FLEURS_DIR, lang_key)
        n = len([f for f in os.listdir(lang_dir) if f.endswith(".wav")]) if os.path.isdir(lang_dir) else 0
        print(f"  {lang_key}: {n} files")


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

    # Balance: same number of utterances per language
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

    lang_counts = {}
    for l in labels:
        lang_counts[l] = lang_counts.get(l, 0) + 1
    print(f"[data] Loaded {len(wavs)} utterances ({min_count}/lang), langs: {lang_counts}")
    return wavs, labels


# ── Step 1: Extract embeddings ────────────────────────────────────────

def _subsample_frames(z, wav, hop, lang, n_per_utt):
    """Subsample frames from encoder output, returning embeddings + audio context."""
    T = z.shape[1]
    step = max(1, T // n_per_utt)
    context_samples = int(ACOUSTIC_CONTEXT_MS / 1000.0 * SAMPLE_RATE)

    embs, frame_labels, audio_contexts = [], [], []
    for t in range(0, T, step):
        embs.append(z[:, t])
        frame_labels.append(lang)
        # Grab a centered audio window (wider than hop for spectral analysis)
        center = t * hop + hop // 2
        start = max(0, center - context_samples // 2)
        end = min(len(wav), center + context_samples // 2)
        audio_contexts.append(wav[start:end])
    return embs, frame_labels, audio_contexts


def extract_jepa_v9(wavs, labels, device):
    from koe.codec_impl import WaveformJEPAFSQVAE

    ckpt_path = "/local_data/checkpoints/v9_wavlm_warmstart/stage2_latest.pt"
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
    model = model.to(device, dtype=torch.bfloat16).eval()
    hop = math.prod(strides)
    print(f"[v9] Loaded (step {ckpt.get('step', '?')}, hop={hop}, {SAMPLE_RATE/hop:.1f} Hz)")

    all_embs, all_labels, all_audio = [], [], []
    for i, (wav, lang) in enumerate(zip(wavs, labels)):
        rem = len(wav) % hop
        wav_pad = np.pad(wav, (0, hop - rem)) if rem else wav
        wav_t = torch.from_numpy(wav_pad).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            z_e = model.encoder.encode(wav_t)
        z = z_e[0].float().cpu().numpy()
        e, fl, ac = _subsample_frames(z, wav, hop, lang, N_FRAMES_PER_UTT)
        all_embs.extend(e)
        all_labels.extend(fl)
        all_audio.extend(ac)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(wavs)}")

    del model; torch.cuda.empty_cache(); gc.collect()
    return np.stack(all_embs), all_labels, all_audio


def extract_jepa_sigreg(wavs, labels, device):
    from koe.codec_impl import JEPAEncoder

    ckpt_path = "/local_data/checkpoints/q2d2_sigreg/v2_latest.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    strides = cfg.get("strides", [4, 4, 4, 5, 3])

    encoder = JEPAEncoder(
        sample_rate=24000, code_dim=128,
        channels=[64, 128, 256, 384, 512, 512],
        strides=strides, n_res_blocks=8, n_conformer=8, conformer_heads=16,
    )
    state = ckpt.get("state_dict", {})
    enc_state = {k.replace("encoder.", "", 1): v for k, v in state.items()
                 if k.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys: {missing[:5]}")
    encoder = encoder.to(device, dtype=torch.bfloat16).eval()
    hop = math.prod(strides)
    print(f"[sigreg] Loaded (step {ckpt.get('step', '?')}, hop={hop}, {SAMPLE_RATE/hop:.1f} Hz)")

    all_embs, all_labels, all_audio = [], [], []
    for i, (wav, lang) in enumerate(zip(wavs, labels)):
        rem = len(wav) % hop
        wav_pad = np.pad(wav, (0, hop - rem)) if rem else wav
        wav_t = torch.from_numpy(wav_pad).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            z_e = encoder.encode(wav_t)
        z = z_e[0].float().cpu().numpy()
        e, fl, ac = _subsample_frames(z, wav, hop, lang, N_FRAMES_PER_UTT)
        all_embs.extend(e)
        all_labels.extend(fl)
        all_audio.extend(ac)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(wavs)}")

    del encoder; torch.cuda.empty_cache(); gc.collect()
    return np.stack(all_embs), all_labels, all_audio


def extract_encodec(wavs, labels, device):
    from encodec import EncodecModel

    model = EncodecModel.encodec_model_24khz().to(device).eval()
    hop = 320  # EnCodec 24kHz uses 320-sample hop = 75 Hz

    all_embs, all_labels = [], []
    for i, (wav, lang) in enumerate(zip(wavs, labels)):
        wav_t = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            z = model.encoder(wav_t)  # [1, 128, T]
        z_np = z[0].float().cpu().numpy()
        T = z_np.shape[1]
        step = max(1, T // N_FRAMES_PER_UTT)
        for t in range(0, T, step):
            all_embs.append(z_np[:, t])
            all_labels.append(lang)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(wavs)}")

    del model; torch.cuda.empty_cache(); gc.collect()
    return np.stack(all_embs), all_labels, None


def extract_dac(wavs, labels, device):
    import dac

    model_path = dac.utils.download(model_type="24khz")
    model = dac.DAC.load(model_path).to(device).eval()
    # DAC hop_length computed from encoder strides
    hop = np.prod(model.encoder_rates)

    all_embs, all_labels = [], []
    for i, (wav, lang) in enumerate(zip(wavs, labels)):
        wav_t = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).float().to(device)
        # Pad to multiple of hop_length
        length = wav_t.shape[-1]
        right_pad = math.ceil(length / hop) * hop - length
        if right_pad > 0:
            wav_t = torch.nn.functional.pad(wav_t, (0, int(right_pad)))
        with torch.no_grad():
            z = model.encoder(wav_t)  # [1, C, T]
        z_np = z[0].float().cpu().numpy()
        T = z_np.shape[1]
        step = max(1, T // N_FRAMES_PER_UTT)
        for t in range(0, T, step):
            all_embs.append(z_np[:, t])
            all_labels.append(lang)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(wavs)}")

    del model; torch.cuda.empty_cache(); gc.collect()
    return np.stack(all_embs), all_labels, None


# ── Step 2: Fair comparison protocol ──────────────────────────────────

def fair_normalize(embeddings_dict, pca_dim=50):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    normalized = {}
    for name, (embs, frame_labels, audio_chunks) in embeddings_dict.items():
        scaler = StandardScaler()
        embs_norm = scaler.fit_transform(embs)
        actual_dim = min(pca_dim, embs.shape[1], embs.shape[0])
        pca = PCA(n_components=actual_dim, random_state=RANDOM_SEED)
        embs_pca = pca.fit_transform(embs_norm)
        var_explained = pca.explained_variance_ratio_.sum()
        print(f"[norm] {name}: {embs.shape[1]}d -> {embs_pca.shape[1]}d, "
              f"var explained: {var_explained:.3f}")
        normalized[name] = (embs_pca, frame_labels, audio_chunks)
    return normalized


def equalize_frame_counts(embeddings_dict):
    """Subsample all models to same total frame count, balanced across languages."""
    rng = np.random.RandomState(RANDOM_SEED)

    # Find minimum frames per language across ALL models
    min_per_lang = float("inf")
    for name, (embs, frame_labels, _) in embeddings_dict.items():
        labels_arr = np.array(frame_labels)
        for lang in np.unique(labels_arr):
            count = (labels_arr == lang).sum()
            min_per_lang = min(min_per_lang, count)
    min_per_lang = int(min_per_lang)

    equalized = {}
    for name, (embs, frame_labels, audio_chunks) in embeddings_dict.items():
        labels_arr = np.array(frame_labels)
        unique_langs = np.unique(labels_arr)
        indices = []
        for lang in unique_langs:
            lang_idx = np.where(labels_arr == lang)[0]
            chosen = rng.choice(lang_idx, size=min_per_lang, replace=False)
            indices.extend(chosen)
        indices = sorted(indices)
        new_embs = embs[indices]
        new_labels = [frame_labels[i] for i in indices]
        new_audio = [audio_chunks[i] for i in indices] if audio_chunks else None
        print(f"[eq] {name}: {len(embs)} -> {len(new_embs)} "
              f"({min_per_lang}/lang x {len(unique_langs)} langs)")
        equalized[name] = (new_embs, new_labels, new_audio)
    return equalized


# ── Step 3: Language separability classifiers ─────────────────────────

def language_classifiers(embeddings_dict):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    results = {}
    n_langs = len(set(list(embeddings_dict.values())[0][1]))
    print(f"  Chance accuracy: {1/n_langs:.3f} ({n_langs} languages)")

    for name, (embs, frame_labels, _) in embeddings_dict.items():
        labels_arr = np.array(frame_labels)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

        classifiers = {
            "LogReg": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
            "LinearSVM": LinearSVC(max_iter=2000, random_state=RANDOM_SEED),
            "5-NN": KNeighborsClassifier(n_neighbors=5),
        }

        model_results = {}
        for clf_name, clf in classifiers.items():
            scores = cross_val_score(clf, embs, labels_arr, cv=skf, scoring="accuracy")
            model_results[clf_name] = {
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "folds": [float(s) for s in scores],
            }
            print(f"  {name:25s} | {clf_name:10s}: "
                  f"{scores.mean():.3f} +/- {scores.std():.3f}")
        results[name] = model_results
    return results


# ── Step 4: Cluster purity metrics ───────────────────────────────────

def cluster_purity(embeddings_dict, k_values=(32, 64, 128)):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import (normalized_mutual_info_score,
                                 adjusted_rand_score, silhouette_score)

    results = {}
    for name, (embs, frame_labels, _) in embeddings_dict.items():
        labels_arr = np.array(frame_labels)
        model_results = {}
        for k in k_values:
            print(f"  {name} | k={k}...", end=" ", flush=True)
            km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_SEED,
                                 n_init=10, batch_size=2048)
            cluster_ids = km.fit_predict(embs)
            nmi = normalized_mutual_info_score(labels_arr, cluster_ids)
            ari = adjusted_rand_score(labels_arr, cluster_ids)
            # Silhouette on subsample for speed
            sil_n = min(10000, len(embs))
            rng = np.random.RandomState(RANDOM_SEED)
            idx = rng.choice(len(embs), sil_n, replace=False)
            sil = silhouette_score(embs[idx], cluster_ids[idx])
            model_results[k] = {
                "NMI": float(nmi), "ARI": float(ari), "silhouette": float(sil),
            }
            print(f"NMI={nmi:.3f}, ARI={ari:.3f}, sil={sil:.3f}")
        results[name] = model_results
    return results


# ── Step 5: Cluster consistency across languages ─────────────────────

def cluster_consistency(embeddings_dict, k=64):
    from sklearn.cluster import MiniBatchKMeans
    from scipy.stats import entropy

    results = {}
    for name, (embs, frame_labels, _) in embeddings_dict.items():
        labels_arr = np.array(frame_labels)
        unique_langs = sorted(set(labels_arr))
        n_langs = len(unique_langs)

        km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_SEED,
                             n_init=10, batch_size=2048)
        cluster_ids = km.fit_predict(embs)
        max_ent = np.log(n_langs)

        cluster_distributions = []
        entropies = []
        balanced_clusters = 0
        for c in range(k):
            mask = cluster_ids == c
            if mask.sum() == 0:
                continue
            cluster_langs = labels_arr[mask]
            dist = {}
            for lang in unique_langs:
                dist[lang] = float((cluster_langs == lang).sum()) / mask.sum()
            cluster_distributions.append({"cluster": c, "size": int(mask.sum()), **dist})

            ent = entropy([dist[l] for l in unique_langs])
            entropies.append(ent)
            if ent > 0.5 * max_ent:
                balanced_clusters += 1

        mean_entropy = float(np.mean(entropies))
        results[name] = {
            "k": k, "n_langs": n_langs,
            "balanced_clusters": balanced_clusters,
            "balanced_ratio": balanced_clusters / k,
            "mean_entropy": mean_entropy,
            "max_entropy": float(max_ent),
            "normalized_entropy": mean_entropy / max_ent,
            "cluster_distributions": cluster_distributions,
        }
        print(f"  {name}: {balanced_clusters}/{k} clusters balanced, "
              f"mean H/Hmax = {mean_entropy/max_ent:.3f}")
    return results


# ── Step 6: Acoustic interpretation ──────────────────────────────────

def acoustic_features_per_cluster(raw_embs, raw_labels, audio_chunks, k=64):
    if audio_chunks is None:
        print("[acoustic] No audio chunks, skipping")
        return None

    from sklearn.cluster import MiniBatchKMeans

    embs = raw_embs
    km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10, batch_size=2048)
    cluster_ids = km.fit_predict(embs)

    cluster_features = []
    for c in range(k):
        mask = cluster_ids == c
        if mask.sum() < 5:
            continue
        chunks = [audio_chunks[i] for i in range(len(audio_chunks)) if mask[i]]

        rms_vals, zcr_vals, sc_vals, f0_vals, voiced_ratio = [], [], [], [], []
        for chunk in chunks[:300]:
            if len(chunk) < 512:
                continue
            # RMS
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            rms_vals.append(rms)
            # ZCR
            zcr = float(np.mean(np.abs(np.diff(np.sign(chunk))) > 0))
            zcr_vals.append(zcr)
            # Spectral centroid
            fft = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
            freqs = np.fft.rfftfreq(len(chunk), 1.0 / SAMPLE_RATE)
            if fft.sum() > 1e-10:
                sc = float(np.sum(freqs * fft) / np.sum(fft))
            else:
                sc = 0.0
            sc_vals.append(sc)
            # F0 via pyin (needs >= ~40ms at 24kHz = 960 samples)
            if len(chunk) >= 960:
                try:
                    f0, voiced, _ = librosa.pyin(
                        chunk, fmin=60, fmax=500, sr=SAMPLE_RATE, frame_length=1024)
                    valid_f0 = f0[~np.isnan(f0)]
                    if len(valid_f0) > 0:
                        f0_vals.append(float(np.median(valid_f0)))
                    voiced_ratio.append(float(np.mean(voiced)))
                except Exception:
                    pass

        if not rms_vals:
            continue

        feat = {
            "cluster": c,
            "n_frames": int(mask.sum()),
            "rms_mean": float(np.mean(rms_vals)),
            "rms_std": float(np.std(rms_vals)),
            "zcr_mean": float(np.mean(zcr_vals)),
            "zcr_std": float(np.std(zcr_vals)),
            "spectral_centroid_mean": float(np.mean(sc_vals)),
            "spectral_centroid_std": float(np.std(sc_vals)),
        }
        if f0_vals:
            feat["f0_median"] = float(np.median(f0_vals))
            feat["f0_std"] = float(np.std(f0_vals))
        if voiced_ratio:
            feat["voiced_ratio"] = float(np.mean(voiced_ratio))
        cluster_features.append(feat)

    # Categorize clusters by acoustic percept
    if not cluster_features:
        return None

    rms_values = [cf["rms_mean"] for cf in cluster_features]
    zcr_values = [cf["zcr_mean"] for cf in cluster_features]
    sc_values = [cf["spectral_centroid_mean"] for cf in cluster_features]

    rms_p25 = np.percentile(rms_values, 25)
    zcr_p75 = np.percentile(zcr_values, 75)
    zcr_p25 = np.percentile(zcr_values, 25)
    sc_p50 = np.percentile(sc_values, 50)

    for cf in cluster_features:
        voiced = cf.get("voiced_ratio", 0.5)
        if cf["rms_mean"] < rms_p25:
            cf["percept"] = "silence/pause"
        elif cf["zcr_mean"] > zcr_p75 and cf["spectral_centroid_mean"] > sc_p50:
            cf["percept"] = "fricative/sibilant"
        elif voiced > 0.6 and cf["zcr_mean"] < zcr_p25:
            cf["percept"] = "vowel/sonorant"
        elif voiced > 0.5:
            cf["percept"] = "nasal/approximant"
        else:
            cf["percept"] = "plosive/mixed"

    print(f"[acoustic] Categorized {len(cluster_features)} clusters:")
    percepts = [cf["percept"] for cf in cluster_features]
    for p in sorted(set(percepts)):
        print(f"  {p}: {percepts.count(p)} clusters")
    return cluster_features


# ── Step 7: Visualizations ───────────────────────────────────────────

def _subsample_for_viz(embs, labels, max_n=N_FRAMES_VIZ):
    rng = np.random.RandomState(RANDOM_SEED)
    labels_arr = np.array(labels)
    unique_langs = np.unique(labels_arr)
    per_lang = max_n // len(unique_langs)
    indices = []
    for lang in unique_langs:
        lang_idx = np.where(labels_arr == lang)[0]
        n = min(per_lang, len(lang_idx))
        chosen = rng.choice(lang_idx, size=n, replace=False)
        indices.extend(chosen)
    rng.shuffle(indices)
    return embs[indices], [labels[i] for i in indices]


def plot_main_comparison(embeddings_dict, output_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    model_names = list(embeddings_dict.keys())
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1: axes = [axes]

    for mi, name in enumerate(model_names):
        embs, labels, _ = embeddings_dict[name]
        embs_sub, labels_sub = _subsample_for_viz(embs, labels)
        tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED, max_iter=1000)
        coords = tsne.fit_transform(embs_sub)
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

    fig.suptitle("English-Trained JEPA Encoders Form Shared Multilingual Acoustic Clusters",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "main_comparison_tsne.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_umap_comparison(embeddings_dict, output_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    model_names = list(embeddings_dict.keys())
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1: axes = [axes]

    for mi, name in enumerate(model_names):
        embs, labels, _ = embeddings_dict[name]
        embs_sub, labels_sub = _subsample_for_viz(embs, labels)
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=RANDOM_SEED)
        coords = reducer.fit_transform(embs_sub)
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

    fig.suptitle("UMAP: Cross-Lingual Organization of Frame Embeddings", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "main_comparison_umap.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_pca_comparison(embeddings_dict, output_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    model_names = list(embeddings_dict.keys())
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1: axes = [axes]

    for mi, name in enumerate(model_names):
        embs, labels, _ = embeddings_dict[name]
        embs_sub, labels_sub = _subsample_for_viz(embs, labels)
        pca = PCA(n_components=2, random_state=RANDOM_SEED)
        coords = pca.fit_transform(embs_sub)
        ax = axes[mi]
        for lang in LANG_COLORS:
            mask = np.array([l == lang for l in labels_sub])
            pts = coords[mask]
            if len(pts) > 0:
                ax.scatter(pts[:, 0], pts[:, 1], c=LANG_COLORS[lang],
                           s=2, alpha=0.25, label=lang, rasterized=True)
        var = pca.explained_variance_ratio_
        ax.set_title(f"{name}\n(PC1: {var[0]:.1%}, PC2: {var[1]:.1%})",
                     fontsize=10, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        if mi == 0:
            ax.legend(fontsize=7, markerscale=4, loc="upper left")

    fig.suptitle("PCA: Cross-Lingual Organization of Frame Embeddings", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "main_comparison_pca.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_tsne_robustness(embeddings_dict, output_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    perplexities = [15, 30, 50]
    seeds = [42, 123, 456]

    for name, (embs, labels, _) in embeddings_dict.items():
        embs_sub, labels_sub = _subsample_for_viz(embs, labels, max_n=20000)
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")

        fig, axes = plt.subplots(len(perplexities), len(seeds),
                                 figsize=(5*len(seeds), 5*len(perplexities)))
        fig.suptitle(f"t-SNE Robustness: {name}", fontsize=14, y=0.98)

        for pi, perp in enumerate(perplexities):
            for si, seed in enumerate(seeds):
                ax = axes[pi, si]
                tsne = TSNE(n_components=2, perplexity=perp,
                            random_state=seed, max_iter=1000)
                coords = tsne.fit_transform(embs_sub)
                for lang in LANG_COLORS:
                    mask = np.array([l == lang for l in labels_sub])
                    pts = coords[mask]
                    if len(pts) > 0:
                        ax.scatter(pts[:, 0], pts[:, 1], c=LANG_COLORS[lang],
                                   s=2, alpha=0.3, rasterized=True,
                                   label=lang if pi == 0 and si == 0 else None)
                ax.set_title(f"perp={perp}, seed={seed}", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])

        axes[0, 0].legend(fontsize=7, markerscale=3)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(output_dir, f"tsne_robustness_{safe}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved {path}")


def plot_3d_tsne(embeddings_dict, output_dir, model_name="JEPA-EMA (v9)"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    embs, labels, _ = embeddings_dict[model_name]
    embs_sub, labels_sub = _subsample_for_viz(embs, labels, max_n=15000)

    tsne = TSNE(n_components=3, perplexity=30, random_state=RANDOM_SEED, max_iter=1000)
    coords = tsne.fit_transform(embs_sub)

    for elev, azim, suffix in [(20, 45, "a"), (20, 135, "b"), (60, 45, "c")]:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        for lang in LANG_COLORS:
            mask = np.array([l == lang for l in labels_sub])
            pts = coords[mask]
            if len(pts) > 0:
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                           c=LANG_COLORS[lang], s=2, alpha=0.25, label=lang)
        ax.view_init(elev=elev, azim=azim)
        ax.legend(fontsize=9, markerscale=3)
        safe = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        ax.set_title(f"3D t-SNE: {model_name}", fontsize=12)
        path = os.path.join(output_dir, f"3d_tsne_{safe}_{suffix}.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved {path}")


def plot_cluster_consistency_heatmap(consistency_results, output_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import entropy as sp_entropy

    for name, res in consistency_results.items():
        dists = res["cluster_distributions"]
        if not dists:
            continue
        langs = sorted([k for k in dists[0].keys() if k not in ("cluster", "size")])
        matrix = np.array([[d[l] for l in langs] for d in dists])

        entropies = [sp_entropy([d[l] for l in langs]) for d in dists]
        order = np.argsort(entropies)[::-1]
        matrix = matrix[order]

        fig, ax = plt.subplots(figsize=(8, max(6, len(dists) * 0.15)))
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.5)
        ax.set_xticks(range(len(langs)))
        ax.set_xticklabels(langs, rotation=45, ha="right")
        ax.set_ylabel("Cluster (sorted by entropy, most balanced at top)")
        ax.set_title(f"Per-Cluster Language Distribution: {name}", fontsize=12)
        plt.colorbar(im, ax=ax, label="Proportion")
        plt.tight_layout()
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        path = os.path.join(output_dir, f"cluster_consistency_{safe}.png")
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"  Saved {path}")


def plot_classifier_bars(classifier_results, output_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_names = list(classifier_results.keys())
    clf_names = list(classifier_results[model_names[0]].keys())
    n_clf = len(clf_names)
    n_models = len(model_names)
    n_langs = 6

    x = np.arange(n_models)
    width = 0.22
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#4472C4", "#ED7D31", "#70AD47"]
    for i, clf in enumerate(clf_names):
        means = [classifier_results[m][clf]["mean"] for m in model_names]
        stds = [classifier_results[m][clf]["std"] for m in model_names]
        ax.bar(x + i * width - width, means, width, yerr=stds,
               label=clf, capsize=3, color=colors[i])

    ax.axhline(y=1.0/n_langs, color="gray", linestyle="--", alpha=0.5,
               label=f"Chance ({n_langs} langs)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Language Separability from Frame Embeddings\n"
                 "(lower = more language-invariant)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = os.path.join(output_dir, "language_classifier_accuracy.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved {path}")


def plot_nmi_bars(cluster_results, output_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_names = list(cluster_results.keys())
    k_values = sorted(cluster_results[model_names[0]].keys())

    x = np.arange(len(model_names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#4472C4", "#ED7D31", "#70AD47"]
    for i, k in enumerate(k_values):
        nmis = [cluster_results[m][k]["NMI"] for m in model_names]
        ax.bar(x + i * width - width, nmis, width, label=f"k={k}", color=colors[i])

    ax.set_ylabel("NMI (cluster, language)")
    ax.set_title("Cluster-Language NMI\n(lower = clusters are NOT language-specific)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "cluster_nmi_comparison.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved {path}")


def plot_acoustic_table(acoustic_features, output_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not acoustic_features:
        return

    top = sorted(acoustic_features, key=lambda x: -x["n_frames"])[:20]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis("off")
    headers = ["Cluster", "Frames", "RMS", "ZCR", "Spec.Cent.", "F0 (Hz)", "Voiced%", "Percept"]
    rows = []
    for cf in top:
        rows.append([
            str(cf["cluster"]),
            str(cf["n_frames"]),
            f"{cf['rms_mean']:.4f}",
            f"{cf['zcr_mean']:.3f}",
            f"{cf['spectral_centroid_mean']:.0f}",
            f"{cf.get('f0_median', 'N/A'):.0f}" if isinstance(cf.get("f0_median"), float) else "N/A",
            f"{cf.get('voiced_ratio', 0):.0%}" if "voiced_ratio" in cf else "N/A",
            cf.get("percept", "?"),
        ])

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    for j in range(len(headers)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Color-code percept column
    percept_colors = {
        "silence/pause": "#E8E8E8", "fricative/sibilant": "#FFD700",
        "vowel/sonorant": "#90EE90", "nasal/approximant": "#87CEEB",
        "plosive/mixed": "#FFB6C1",
    }
    for r in range(len(rows)):
        p = rows[r][-1]
        if p in percept_colors:
            table[r + 1, len(headers) - 1].set_facecolor(percept_colors[p])

    ax.set_title("Acoustic Feature Summary per Cluster (JEPA-EMA v9, k=64, top 20)",
                 fontsize=13, pad=20)
    plt.tight_layout()
    path = os.path.join(output_dir, "acoustic_features_table.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--n_per_lang", type=int, default=None)
    args = parser.parse_args()

    device = args.device
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_start = time.time()
    print(f"Output: {OUTPUT_DIR}")
    print(f"Device: {device}")

    # ── Step 0 ──
    if not args.skip_download:
        print("\n" + "=" * 60)
        print("STEP 0: Download FLEURS data (test + validation)")
        print("=" * 60)
        download_fleurs()

    # ── Step 1 ──
    print("\n" + "=" * 60)
    print("STEP 1: Load audio & extract embeddings")
    print("=" * 60)
    wavs, labels = load_audio(n_per_lang=args.n_per_lang)

    raw_embeddings = {}

    print("\n[1/4] JEPA-EMA v9...")
    t0 = time.time()
    v9_embs, v9_labels, v9_audio = extract_jepa_v9(wavs, labels, device)
    print(f"  {v9_embs.shape} in {time.time()-t0:.1f}s")
    raw_embeddings["JEPA-EMA (v9)"] = (v9_embs, v9_labels, v9_audio)

    print("\n[2/4] JEPA-SIGReg 25Hz...")
    t0 = time.time()
    sr_embs, sr_labels, sr_audio = extract_jepa_sigreg(wavs, labels, device)
    print(f"  {sr_embs.shape} in {time.time()-t0:.1f}s")
    raw_embeddings["JEPA-SIGReg (25Hz)"] = (sr_embs, sr_labels, sr_audio)

    print("\n[3/4] EnCodec...")
    t0 = time.time()
    ec_embs, ec_labels, _ = extract_encodec(wavs, labels, device)
    print(f"  {ec_embs.shape} in {time.time()-t0:.1f}s")
    raw_embeddings["EnCodec"] = (ec_embs, ec_labels, None)

    print("\n[4/4] DAC...")
    t0 = time.time()
    dac_embs, dac_labels, _ = extract_dac(wavs, labels, device)
    print(f"  {dac_embs.shape} in {time.time()-t0:.1f}s")
    raw_embeddings["DAC"] = (dac_embs, dac_labels, None)

    # Raw embedding stats
    raw_stats = {}
    for name, (embs, fl, _) in raw_embeddings.items():
        cov = np.cov(embs.T)
        eigvals = np.linalg.eigvalsh(cov)
        p = eigvals / eigvals.sum()
        erank = float(np.exp(-np.sum(p * np.log(p + 1e-10))))
        raw_stats[name] = {
            "n_frames": len(embs), "n_dims": embs.shape[1],
            "effective_rank": erank,
            "per_dim_std": float(embs.std(axis=0).mean()),
        }
        print(f"  {name}: {embs.shape}, erank={erank:.1f}")
    with open(os.path.join(OUTPUT_DIR, "raw_embedding_stats.json"), "w") as f:
        json.dump(raw_stats, f, indent=2)

    # ── Step 2 ──
    print("\n" + "=" * 60)
    print("STEP 2: Fair comparison protocol (normalize + PCA + equalize)")
    print("=" * 60)
    norm_embeddings = fair_normalize(raw_embeddings, pca_dim=50)
    eq_embeddings = equalize_frame_counts(norm_embeddings)

    # ── Step 3 ──
    print("\n" + "=" * 60)
    print("STEP 3: Language separability classifiers")
    print("=" * 60)
    clf_results = language_classifiers(eq_embeddings)
    with open(os.path.join(OUTPUT_DIR, "classifier_results.json"), "w") as f:
        json.dump(clf_results, f, indent=2)

    # ── Step 4 ──
    print("\n" + "=" * 60)
    print("STEP 4: Cluster purity (NMI, ARI, silhouette)")
    print("=" * 60)
    cluster_results = cluster_purity(eq_embeddings)
    with open(os.path.join(OUTPUT_DIR, "cluster_purity_results.json"), "w") as f:
        json.dump(cluster_results, f, indent=2)

    # ── Step 5 ──
    print("\n" + "=" * 60)
    print("STEP 5: Cluster consistency across languages")
    print("=" * 60)
    consistency_results = cluster_consistency(eq_embeddings, k=64)
    consistency_save = {}
    for k, v in consistency_results.items():
        consistency_save[k] = {kk: vv for kk, vv in v.items()
                               if kk != "cluster_distributions"}
    with open(os.path.join(OUTPUT_DIR, "cluster_consistency_results.json"), "w") as f:
        json.dump(consistency_save, f, indent=2)

    # ── Step 6 ──
    print("\n" + "=" * 60)
    print("STEP 6: Acoustic interpretation (v9)")
    print("=" * 60)
    v9_raw = raw_embeddings["JEPA-EMA (v9)"]
    acoustic = acoustic_features_per_cluster(v9_raw[0], v9_raw[1], v9_raw[2], k=64)
    if acoustic:
        with open(os.path.join(OUTPUT_DIR, "acoustic_features.json"), "w") as f:
            json.dump(acoustic, f, indent=2)

    # ── Step 7 ──
    print("\n" + "=" * 60)
    print("STEP 7: Visualizations")
    print("=" * 60)

    print("\n[7a] Main t-SNE comparison (all models)...")
    plot_main_comparison(eq_embeddings, OUTPUT_DIR)

    print("\n[7b] UMAP comparison...")
    plot_umap_comparison(eq_embeddings, OUTPUT_DIR)

    print("\n[7c] PCA comparison...")
    plot_pca_comparison(eq_embeddings, OUTPUT_DIR)

    print("\n[7d] 3D t-SNE (v9)...")
    plot_3d_tsne(eq_embeddings, OUTPUT_DIR, model_name="JEPA-EMA (v9)")

    print("\n[7e] t-SNE robustness (v9 only)...")
    plot_tsne_robustness(
        {"JEPA-EMA (v9)": eq_embeddings["JEPA-EMA (v9)"]}, OUTPUT_DIR)

    print("\n[7f] Cluster consistency heatmaps...")
    plot_cluster_consistency_heatmap(consistency_results, OUTPUT_DIR)

    print("\n[7g] Classifier accuracy bar chart...")
    plot_classifier_bars(clf_results, OUTPUT_DIR)

    print("\n[7h] NMI bar chart...")
    plot_nmi_bars(cluster_results, OUTPUT_DIR)

    print("\n[7i] Acoustic features table...")
    plot_acoustic_table(acoustic, OUTPUT_DIR)

    # ── Summary ──
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"COMPLETE in {elapsed/60:.1f} minutes")
    print("=" * 60)
    print(f"\nAll outputs in {OUTPUT_DIR}/")
    for fn in sorted(os.listdir(OUTPUT_DIR)):
        sz = os.path.getsize(os.path.join(OUTPUT_DIR, fn))
        print(f"  {fn} ({sz/1024:.1f} KB)")


if __name__ == "__main__":
    main()
