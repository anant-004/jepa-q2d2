"""Multi-class Hindi emotion separation on held-out emotional speech.

Emotion categories of Hindi/Hinglish synthetic speech from a single synthetic
voice family (the codec was trained on English LibriLight only, so all classes
are cross-lingual held out). Dataset ids are read from env vars EMOTION_*_DS;
set them to your own repos.

Pipeline (mirrors style_emotion_embedding_analysis.py):
  1. Load N samples per class, resample to 24 kHz
  2. Extract frame-level embeddings: JEPA-EMA v9, JEPA-SIGReg, EnCodec, DAC, Mimi
  3. Fair protocol: standardize -> PCA50 -> equalize counts
  4. Classifiers: LogReg / LinSVM / 5-NN, 5-fold CV (multi-class)
  5. Frame-level + utterance-level t-SNE

Run on VM:
  HF_TOKEN=$TOKEN CUDA_VISIBLE_DEVICES=0 \\
    .venv/bin/python -u scripts/emotion_multiclass_analysis.py
"""
import gc
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, os.path.expanduser("~/koe"))
warnings.filterwarnings("ignore")

SAMPLE_RATE = 24000
OUTPUT_DIR = "/local_data/analysis/emotion_multiclass"
N_PER_CLASS = 100
N_FRAMES_PER_UTT = 30
RANDOM_SEED = 42

# (label, repo, format). Dataset ids come from env vars; set them to your repos.
DATASETS = [
    (label, os.environ.get(env, ""), "parquet")
    for label, env in [
        ("whisper", "EMOTION_WHISPER_DS"),
        ("angry",   "EMOTION_ANGRY_DS"),
        ("excited", "EMOTION_EXCITED_DS"),
        ("neutral", "EMOTION_NEUTRAL_DS"),
        ("sad",     "EMOTION_SAD_DS"),
    ]
]
DATASETS = [d for d in DATASETS if d[1]]
COLORS = {
    "whisper":  "#4472C4",   # blue
    "angry":    "#E74C3C",   # red
    "excited":  "#F39C12",   # orange
    "neutral":  "#7F8C8D",   # grey
    "sad":      "#8E44AD",   # purple
}


def _resample_to_target(wav, sr):
    if sr != SAMPLE_RATE:
        wav_t = torch.from_numpy(wav).unsqueeze(0)
        wav_t = torchaudio.functional.resample(wav_t, sr, SAMPLE_RATE)
        wav = wav_t.squeeze(0).numpy()
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    return wav


def _trim(wav, max_seconds=10.0):
    dur = len(wav) / SAMPLE_RATE
    if dur < 1.0:
        return None
    if dur > max_seconds:
        wav = wav[: int(max_seconds * SAMPLE_RATE)]
    return wav


def load_parquet_repo(repo, label, n, rng):
    """Load via datasets.load_dataset, robust multi-attempt (as in original)."""
    from datasets import load_dataset
    ds = None
    for attempt, kwargs in enumerate([
        {"split": "train", "verification_mode": "no_checks"},
        {"split": "train", "verification_mode": "no_checks", "download_mode": "force_redownload"},
        {"streaming": True, "split": "train"},
    ]):
        try:
            print(f"  [{label}] attempt {attempt+1}: {kwargs}", flush=True)
            ds = load_dataset(repo, **kwargs)
            if hasattr(ds, "keys") and not isinstance(ds, list):
                split_name = "train" if "train" in ds else list(ds.keys())[0]
                ds = ds[split_name]
            print(f"  [{label}] loaded via attempt {attempt+1}", flush=True)
            streaming = kwargs.get("streaming", False)
            break
        except Exception as e:
            print(f"  [{label}] attempt {attempt+1} failed: {type(e).__name__}: {str(e)[:160]}",
                  flush=True)
            ds = None
            continue
    if ds is None:
        return []
    wavs = []
    if streaming:
        for i, item in enumerate(ds):
            if len(wavs) >= n:
                break
            try:
                audio = item.get("audio")
                if audio is None:
                    continue
                if isinstance(audio, dict):
                    wav = np.asarray(audio["array"], dtype=np.float32)
                    sr = audio["sampling_rate"]
                else:
                    wav = np.asarray(audio, dtype=np.float32)
                    sr = item.get("sampling_rate", SAMPLE_RATE)
                wav = _resample_to_target(wav, sr)
                wav = _trim(wav)
                if wav is not None:
                    wavs.append(wav)
            except Exception:
                continue
    else:
        total = len(ds)
        idxs = rng.choice(total, size=min(n * 3, total), replace=False)
        for idx in idxs:
            if len(wavs) >= n:
                break
            try:
                item = ds[int(idx)]
                audio = item.get("audio")
                if audio is None:
                    continue
                if isinstance(audio, dict):
                    wav = np.asarray(audio["array"], dtype=np.float32)
                    sr = audio["sampling_rate"]
                else:
                    wav = np.asarray(audio, dtype=np.float32)
                    sr = item.get("sampling_rate", SAMPLE_RATE)
                wav = _resample_to_target(wav, sr)
                wav = _trim(wav)
                if wav is not None:
                    wavs.append(wav)
            except Exception:
                continue
    print(f"  [{label}] loaded {len(wavs)} samples", flush=True)
    return wavs


def load_loose_wav_repo(repo, label, n, rng):
    """Snapshot-download the audio/ folder and read every .wav."""
    from huggingface_hub import snapshot_download
    cache_dir = f"/local_data/hf_cache/{label}"
    os.makedirs(cache_dir, exist_ok=True)
    try:
        path = snapshot_download(repo_id=repo, repo_type="dataset",
                                  allow_patterns=["audio/**/*.wav"],
                                  local_dir=cache_dir)
    except Exception as e:
        print(f"  [{label}] snapshot_download failed: {type(e).__name__}: {e}",
              flush=True)
        return []
    wavs_files = sorted(Path(path).rglob("*.wav"))
    print(f"  [{label}] found {len(wavs_files)} wav files", flush=True)
    if not wavs_files:
        return []
    if len(wavs_files) > n:
        idxs = rng.choice(len(wavs_files), size=n, replace=False)
        wavs_files = [wavs_files[i] for i in sorted(idxs)]
    wavs = []
    for f in wavs_files:
        try:
            data, sr = sf.read(str(f), dtype="float32")
            wav = _resample_to_target(np.asarray(data), sr)
            wav = _trim(wav)
            if wav is not None:
                wavs.append(wav)
        except Exception:
            continue
    print(f"  [{label}] kept {len(wavs)} samples", flush=True)
    return wavs


def load_all():
    if "HF_TOKEN" in os.environ:
        from huggingface_hub import login
        login(token=os.environ["HF_TOKEN"])
    rng = np.random.RandomState(RANDOM_SEED)
    wavs, labels = [], []
    for label, repo, fmt in DATASETS:
        print(f"\n[data] {label}: {repo} ({fmt})", flush=True)
        if fmt == "parquet":
            ws = load_parquet_repo(repo, label, N_PER_CLASS, rng)
        else:
            ws = load_loose_wav_repo(repo, label, N_PER_CLASS, rng)
        for w in ws:
            wavs.append(w)
            labels.append(label)
    counts = {l: labels.count(l) for l in set(labels)}
    print(f"\n[data] TOTAL: {len(wavs)} samples; per-class: {counts}", flush=True)
    return wavs, labels


def _subsample_frames(z, lab, n_per_utt):
    T = z.shape[1]
    step = max(1, T // n_per_utt)
    embs, lbls = [], []
    for t in range(0, T, step):
        embs.append(z[:, t])
        lbls.append(lab)
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
    e_all, l_all = [], []
    for wav, lab in zip(wavs, labels):
        rem = len(wav) % hop
        wav_p = np.pad(wav, (0, hop - rem)) if rem else wav
        wt = torch.from_numpy(wav_p).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            z = model.encoder.encode(wt)[0].float().cpu().numpy()
        e, l = _subsample_frames(z, lab, N_FRAMES_PER_UTT)
        e_all.extend(e); l_all.extend(l)
    del model; torch.cuda.empty_cache(); gc.collect()
    return np.stack(e_all), l_all


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
    e_all, l_all = [], []
    for wav, lab in zip(wavs, labels):
        rem = len(wav) % hop
        wav_p = np.pad(wav, (0, hop - rem)) if rem else wav
        wt = torch.from_numpy(wav_p).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            z = encoder.encode(wt)[0].float().cpu().numpy()
        e, l = _subsample_frames(z, lab, N_FRAMES_PER_UTT)
        e_all.extend(e); l_all.extend(l)
    del encoder; torch.cuda.empty_cache(); gc.collect()
    return np.stack(e_all), l_all


def extract_encodec(wavs, labels, device):
    from encodec import EncodecModel
    m = EncodecModel.encodec_model_24khz().to(device).eval()
    e_all, l_all = [], []
    for wav, lab in zip(wavs, labels):
        wt = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            z = m.encoder(wt)[0].float().cpu().numpy()
        e, l = _subsample_frames(z, lab, N_FRAMES_PER_UTT)
        e_all.extend(e); l_all.extend(l)
    del m; torch.cuda.empty_cache(); gc.collect()
    return np.stack(e_all), l_all


def extract_dac(wavs, labels, device):
    import dac
    mp = dac.utils.download(model_type="24khz")
    m = dac.DAC.load(mp).to(device).eval()
    hop = int(np.prod(m.encoder_rates))
    e_all, l_all = [], []
    for wav, lab in zip(wavs, labels):
        wt = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).float().to(device)
        rp = math.ceil(wt.shape[-1] / hop) * hop - wt.shape[-1]
        if rp > 0:
            wt = torch.nn.functional.pad(wt, (0, int(rp)))
        with torch.no_grad():
            z = m.encoder(wt)[0].float().cpu().numpy()
        e, l = _subsample_frames(z, lab, N_FRAMES_PER_UTT)
        e_all.extend(e); l_all.extend(l)
    del m; torch.cuda.empty_cache(); gc.collect()
    return np.stack(e_all), l_all


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip_mimi", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()

    print("=== STEP 1: load all 6 emotion sets ===", flush=True)
    wavs, labels = load_all()
    if len(wavs) == 0:
        print("ERROR: no samples loaded"); return
    seen = sorted(set(labels))
    print(f"loaded classes: {seen}", flush=True)

    print("\n=== STEP 2: extract embeddings ===", flush=True)
    raw = {}
    for name, fn in [
        ("JEPA-EMA (v9)", extract_jepa_v9),
        ("JEPA-SIGReg (25Hz)", extract_jepa_sigreg),
        ("EnCodec", extract_encodec),
        ("DAC", extract_dac),
    ]:
        print(f"  {name} ...", flush=True)
        try:
            raw[name] = fn(wavs, labels, args.device)
            print(f"    shape={raw[name][0].shape}", flush=True)
        except Exception as e:
            print(f"    {name} FAILED: {type(e).__name__}: {e}", flush=True)

    print("\n=== STEP 3: fair protocol ===", flush=True)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    norm = {}
    for name, (embs, lbl) in raw.items():
        embs_s = StandardScaler().fit_transform(embs)
        pca = PCA(n_components=min(50, embs.shape[1]), random_state=RANDOM_SEED)
        norm[name] = (pca.fit_transform(embs_s), lbl)
    rng = np.random.RandomState(RANDOM_SEED)
    eq = {}
    for name, (embs, lbl) in norm.items():
        arr = np.array(lbl)
        mc = min((arr == l).sum() for l in seen if (arr == l).sum() > 0)
        idxs = []
        for l in seen:
            li = np.where(arr == l)[0]
            if len(li) >= mc:
                idxs.extend(rng.choice(li, size=mc, replace=False))
        eq[name] = (embs[idxs], [lbl[i] for i in sorted(idxs)])
        print(f"  {name}: balanced to {mc} frames/class", flush=True)

    print("\n=== STEP 4: classifiers ===", flush=True)
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    clf_results = {}
    for name, (embs, lbl) in eq.items():
        arr = np.array(lbl)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        clfs = {
            "LogReg": LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
            "LinearSVM": LinearSVC(max_iter=3000, random_state=RANDOM_SEED),
            "5-NN": KNeighborsClassifier(n_neighbors=5),
        }
        mr = {}
        for cn, clf in clfs.items():
            sc = cross_val_score(clf, embs, arr, cv=skf, scoring="accuracy")
            mr[cn] = {"mean": float(sc.mean()), "std": float(sc.std())}
            print(f"  {name:24s} {cn:10s} {sc.mean():.3f} +/- {sc.std():.3f}", flush=True)
        clf_results[name] = mr
    with open(os.path.join(OUTPUT_DIR, "emotion_classifier_results.json"), "w") as f:
        json.dump({"chance": 1.0 / len(seen), "classes": seen,
                   "results": clf_results}, f, indent=2)

    print("\n=== STEP 5: t-SNE ===", flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    order = ["JEPA-EMA (v9)", "JEPA-SIGReg (25Hz)", "EnCodec", "DAC"]
    avail = [m for m in order if m in eq]
    n = len(avail)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.6))
    if n == 1: axes = [axes]
    for mi, name in enumerate(avail):
        embs, lbl = eq[name]
        if len(embs) > 10000:
            i2 = rng.choice(len(embs), 10000, replace=False)
            embs = embs[i2]; lbl = [lbl[i] for i in i2]
        ts = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED, max_iter=1000)
        co = ts.fit_transform(embs)
        ax = axes[mi]
        for cl in seen:
            mask = np.array([l == cl for l in lbl])
            if mask.sum() > 0:
                ax.scatter(co[mask, 0], co[mask, 1], c=COLORS[cl], s=3, alpha=0.45,
                            label=cl, rasterized=True)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        if mi == 0:
            ax.legend(fontsize=8, markerscale=3, loc="best")
    fig.suptitle(f"Hindi emotion separation, {len(seen)}-class, frame-level "
                  f"(codec trained on English only)", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "emotion_multiclass_tsne.png"),
                dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, "emotion_multiclass_tsne.pdf"),
                bbox_inches="tight")
    plt.close()

    # utterance-level
    print("\n=== STEP 6: utterance t-SNE ===", flush=True)
    utt = {}
    for name, (embs, lbl) in raw.items():
        nu = len(embs) // N_FRAMES_PER_UTT
        u_e, u_l = [], []
        for u in range(nu):
            s = u * N_FRAMES_PER_UTT; e = s + N_FRAMES_PER_UTT
            u_e.append(embs[s:e].mean(axis=0)); u_l.append(lbl[s])
        utt[name] = (np.stack(u_e), u_l)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.6))
    if n == 1: axes = [axes]
    for mi, name in enumerate(avail):
        embs, lbl = utt[name]
        if len(embs) < 5: continue
        es = StandardScaler().fit_transform(embs)
        ts = TSNE(n_components=2, perplexity=min(30, len(embs) - 1),
                  random_state=RANDOM_SEED, max_iter=1000)
        co = ts.fit_transform(es)
        ax = axes[mi]
        for cl in seen:
            mask = np.array([l == cl for l in lbl])
            if mask.sum() > 0:
                ax.scatter(co[mask, 0], co[mask, 1], c=COLORS[cl], s=42, alpha=0.85,
                            edgecolor="white", linewidth=0.5, label=cl)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        if mi == 0:
            ax.legend(fontsize=8, markerscale=1.3, loc="best")
    fig.suptitle(f"Hindi emotion separation, utterance-level mean-pooled, "
                  f"{len(seen)} classes", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "emotion_multiclass_utterance_tsne.png"),
                dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, "emotion_multiclass_utterance_tsne.pdf"),
                bbox_inches="tight")
    plt.close()

    print(f"\nDONE in {(time.time()-t0)/60:.1f} min. Output: {OUTPUT_DIR}", flush=True)
    for fn in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {fn}", flush=True)


if __name__ == "__main__":
    main()
