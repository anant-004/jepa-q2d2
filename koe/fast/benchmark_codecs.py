"""Benchmark codecs: v9 12.5Hz JEPA vs DAC vs Mimi.

Evaluates PESQ, STOI, mel distance across English, Chinese, Hindi, Japanese.
Saves per-sample CSV + reconstructed audio for A/B listening (5 per language).

Usage:
    python -m koe.fast.benchmark_codecs \
        --v9_ckpt /path/to/v9_ll_stage2_352k.pt \
        --output_dir ./benchmark_results \
        --languages en zh hi ja \
        --samples_per_lang 50 \
        --audio_samples 5
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

SAMPLE_RATE = 24000
MAX_SECONDS = 15.0
MIN_SECONDS = 2.0

# ═══════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════

def _decode_hf_audio(item) -> Optional[Tuple[np.ndarray, int]]:
    """Decode audio from HF dataset item, handling different formats."""
    audio = item.get("audio")
    if audio is None:
        for k, v in item.items():
            if isinstance(v, dict) and ("array" in v or "bytes" in v):
                audio = v
                break
    if audio is None:
        return None
    if "array" in audio and audio["array"] is not None:
        return np.array(audio["array"], dtype=np.float32), audio["sampling_rate"]
    if "bytes" in audio and audio["bytes"] is not None:
        import io
        data, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        return data, sr
    if "path" in audio and audio["path"]:
        data, sr = sf.read(audio["path"], dtype="float32")
        return data, sr
    return None


def load_librispeech_samples(n: int, split: str = "test.clean") -> List[Tuple[np.ndarray, int, dict]]:
    """Load n samples from LibriSpeech test-clean."""
    from datasets import load_dataset
    ds = load_dataset("librispeech_asr", split=split, streaming=True,
                       token=os.environ.get("HF_TOKEN"))
    samples = []
    for item in ds:
        result = _decode_hf_audio(item)
        if result is None:
            continue
        wav, sr = result
        dur = len(wav) / sr
        if dur < MIN_SECONDS or dur > MAX_SECONDS:
            continue
        meta = {"id": item.get("id", len(samples)), "text": item.get("text", ""),
                "speaker": item.get("speaker_id", ""), "language": "en"}
        samples.append((wav, sr, meta))
        if len(samples) >= n:
            break
    print(f"[data] Loaded {len(samples)} English samples from LibriSpeech {split}")
    return samples


def load_commonvoice_samples(lang: str, n: int) -> List[Tuple[np.ndarray, int, dict]]:
    """Load n samples from Common Voice for a given language."""
    from datasets import load_dataset
    lang_map = {"zh": "zh-CN", "ja": "ja", "hi": "hi"}
    cv_lang = lang_map.get(lang, lang)
    # Try Common Voice with auth token, fallback to alternative datasets
    datasets_to_try = [
        ("mozilla-foundation/common_voice_17_0", cv_lang, "test", False),
        ("google/fleurs", {"zh": "cmn_hans_cn", "ja": "ja_jp", "hi": "hi_in", "en": "en_us"}.get(lang, lang + "_" + lang), "test", True),
    ]
    for ds_name, config, split, trust_code in datasets_to_try:
        try:
            ds = load_dataset(ds_name, config, split=split, streaming=True,
                              token=os.environ.get("HF_TOKEN"),
                              trust_remote_code=trust_code)
            samples = []
            for item in ds:
                result = _decode_hf_audio(item)
                if result is None:
                    continue
                wav, sr = result
                dur = len(wav) / sr
                if dur < MIN_SECONDS or dur > MAX_SECONDS:
                    continue
                meta = {"id": str(len(samples)),
                        "text": item.get("sentence", item.get("transcription", "")),
                        "language": lang}
                samples.append((wav, sr, meta))
                if len(samples) >= n:
                    break
            if samples:
                print(f"[data] Loaded {len(samples)} {lang} samples from {ds_name}/{config}")
                return samples
        except Exception as e:
            print(f"[data] Failed {ds_name}/{config}: {e}")
    print(f"[data] WARNING: No {lang} samples loaded!")
    return []


def load_hindi_samples(n: int) -> List[Tuple[np.ndarray, int, dict]]:
    """Load n Hindi samples from a private HF dataset.

    Dataset ids come from env vars HINDI_DATASET (primary) and
    HINDI_DATASET_FALLBACK (optional). Set them to your own repos.
    """
    from datasets import load_dataset
    datasets_to_try = [
        (os.environ.get("HINDI_DATASET", ""), "train"),
        (os.environ.get("HINDI_DATASET_FALLBACK", ""), "train"),
    ]
    datasets_to_try = [(d, s) for d, s in datasets_to_try if d]
    samples = []
    for ds_name, split in datasets_to_try:
        if len(samples) >= n:
            break
        try:
            ds = load_dataset(ds_name, split=split, streaming=True,
                               token=os.environ.get("HF_TOKEN"))
            for item in ds:
                if len(samples) >= n:
                    break
                result = _decode_hf_audio(item)
                if result is None:
                    continue
                wav, sr = result
                dur = len(wav) / sr
                if dur < MIN_SECONDS or dur > MAX_SECONDS:
                    continue
                text = item.get("text", item.get("sentence", item.get("transcription", "")))
                meta = {"id": str(len(samples)), "text": str(text)[:200] if text else "",
                        "language": "hi", "source": ds_name.split("/")[-1]}
                samples.append((wav, sr, meta))
            print(f"[data] Got {len(samples)} Hindi samples from {ds_name}")
        except Exception as e:
            print(f"[data] Failed to load {ds_name}: {e}")
    if not samples:
        print("[data] Falling back to Common Voice Hindi")
        return load_commonvoice_samples("hi", n)
    return samples


def resample_to_24k(wav: np.ndarray, sr: int) -> np.ndarray:
    """Resample audio to 24kHz."""
    if sr == SAMPLE_RATE:
        return wav
    wav_t = torch.from_numpy(wav).unsqueeze(0)
    wav_t = torchaudio.functional.resample(wav_t, sr, SAMPLE_RATE)
    return wav_t.squeeze(0).numpy()


# ═══════════════════════════════════════════════════════════
# Codec Wrappers
# ═══════════════════════════════════════════════════════════

class V9Codec:
    """Wrapper for our v9 12.5Hz JEPA codec."""

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        from koe.codec_impl import WaveformJEPAFSQVAE
        self.device = torch.device(device)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        strides = cfg.get("strides", [4, 4, 4, 5, 6])
        fsq_levels = cfg.get("fsq_levels", [8, 8, 8, 8])
        n_res_blocks = cfg.get("n_res_blocks", 8)

        self.model = WaveformJEPAFSQVAE(
            sample_rate=24000, code_dim=128,
            channels=[64, 128, 256, 384, 512, 512],
            strides=strides, n_res_blocks=n_res_blocks,
            n_conformer=8, conformer_heads=16,
            fsq_levels=fsq_levels, hifi_kernels=[3, 7, 11, 15, 23, 32],
        )
        sd = ckpt.get("state_dict", {})
        self.model.load_state_dict(sd, strict=False)
        self.model.eval().to(self.device, dtype=torch.bfloat16)

        hop = 1
        for s in strides:
            hop *= s
        self.hop = hop
        self.name = f"v9_jepa_{int(24000/hop)}hz"
        print(f"[codec] Loaded {self.name} from {ckpt_path} (hop={hop})")

    @torch.no_grad()
    def reconstruct(self, wav_24k: np.ndarray) -> np.ndarray:
        # Pad to hop alignment
        rem = len(wav_24k) % self.hop
        if rem:
            wav_24k = np.pad(wav_24k, (0, self.hop - rem))
        wav_t = torch.from_numpy(wav_24k).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.bfloat16)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            rec, _, _, _ = self.model(wav_t)
        return rec[0, 0].float().cpu().numpy()[:len(wav_24k)]


class DACCodec:
    """Wrapper for Descript Audio Codec."""

    def __init__(self, model_type: str = "24khz", device: str = "cuda"):
        import dac
        model_path = dac.utils.download(model_type=model_type)
        self.model = dac.DAC.load(model_path)
        self.model.eval().to(device)
        self.device = device
        self.name = f"dac_{model_type}"
        print(f"[codec] Loaded DAC {model_type}")

    @torch.no_grad()
    def reconstruct(self, wav_24k: np.ndarray) -> np.ndarray:
        from audiotools import AudioSignal
        sig = AudioSignal(wav_24k, sample_rate=SAMPLE_RATE)
        sig = sig.to(self.device)
        x = self.model.preprocess(sig.audio_data, sig.sample_rate)
        z, codes, latents, _, _ = self.model.encode(x)
        rec = self.model.decode(z)
        return rec[0, 0].detach().cpu().numpy()[:len(wav_24k)]


class MimiCodec:
    """Wrapper for Kyutai Mimi codec."""

    def __init__(self, device: str = "cuda"):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained("kyutai/mimi", trust_remote_code=True)
        self.model.eval().to(device)
        self.device = device
        self.name = "mimi"
        print(f"[codec] Loaded Mimi")

    @torch.no_grad()
    def reconstruct(self, wav_24k: np.ndarray) -> np.ndarray:
        wav_t = torch.from_numpy(wav_24k).unsqueeze(0).unsqueeze(0).to(self.device)
        enc = self.model.encode(wav_t)
        codes = enc.audio_codes
        rec = self.model.decode(codes)
        return rec.audio_values[0, 0].detach().cpu().numpy()[:len(wav_24k)]


class EnCodecWrapper:
    """Wrapper for Facebook EnCodec at configurable bandwidth."""

    def __init__(self, bandwidth: float = 6.0, device: str = "cuda"):
        from transformers import EncodecModel, AutoProcessor
        self.processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        self.model = EncodecModel.from_pretrained("facebook/encodec_24khz")
        self.model.eval().to(device)
        self.model.config.target_bandwidths = [bandwidth]
        self.device = device
        self.bandwidth = bandwidth
        self.name = f"encodec_{bandwidth}kbps"
        print(f"[codec] Loaded EnCodec at {bandwidth} kbps")

    @torch.no_grad()
    def reconstruct(self, wav_24k: np.ndarray) -> np.ndarray:
        inputs = self.processor(raw_audio=wav_24k, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        enc = self.model.encode(**inputs)
        rec = self.model.decode(enc.audio_codes, enc.audio_scales)
        return rec.audio_values[0, 0].detach().cpu().numpy()[:len(wav_24k)]


# ═══════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════

def compute_metrics(original: np.ndarray, reconstructed: np.ndarray, sr: int = 24000,
                     compute_heavy: bool = True) -> Dict[str, float]:
    """Compute PESQ, STOI, mel distance, and optionally UTMOS, WER, speaker sim."""
    n = min(len(original), len(reconstructed))
    orig, rec = original[:n], reconstructed[:n]

    metrics = {}

    # PESQ (needs 16kHz)
    try:
        import librosa
        from pesq import pesq
        o16 = librosa.resample(orig, orig_sr=sr, target_sr=16000)
        r16 = librosa.resample(rec, orig_sr=sr, target_sr=16000)
        n16 = min(len(o16), len(r16))
        metrics["pesq"] = pesq(16000, o16[:n16], r16[:n16], "wb")
    except Exception:
        metrics["pesq"] = float("nan")

    # STOI
    try:
        from pystoi import stoi
        metrics["stoi"] = stoi(orig, rec, sr, extended=True)
    except Exception:
        metrics["stoi"] = float("nan")

    # UTMOS (learned MOS predictor, no-reference on reconstructed only)
    if compute_heavy:
        try:
            from koe.fast.eval_metrics import compute_utmos
            metrics["utmos"] = compute_utmos(rec, sr)
        except Exception:
            metrics["utmos"] = float("nan")

        # WER via Whisper (content preservation)
        try:
            from koe.fast.eval_metrics import compute_wer
            metrics["wer"] = compute_wer(orig, rec, sr)
        except Exception:
            metrics["wer"] = float("nan")

        # Speaker similarity via WavLM
        try:
            from koe.fast.eval_metrics import compute_speaker_similarity
            metrics["speaker_sim"] = compute_speaker_similarity(orig, rec, sr)
        except Exception:
            metrics["speaker_sim"] = float("nan")

    # Mel spectral distance
    try:
        import librosa
        total = 0.0
        for n_mels in (80, 128):
            for n_fft in (1024, 2048):
                hop = n_fft // 4
                m_o = librosa.feature.melspectrogram(y=orig, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
                m_r = librosa.feature.melspectrogram(y=rec, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
                lo, lr = np.log1p(m_o), np.log1p(m_r)
                t = min(lo.shape[1], lr.shape[1])
                total += np.mean(np.abs(lo[:, :t] - lr[:, :t]))
        metrics["mel_distance"] = total / 4.0
    except Exception:
        metrics["mel_distance"] = float("nan")

    return metrics


# ═══════════════════════════════════════════════════════════
# HTML A/B Player
# ═══════════════════════════════════════════════════════════

def generate_html_player(output_dir: Path, results: List[dict], codecs: List[str], max_per_lang: int = 5):
    """Generate an HTML page for side-by-side audio comparison (first max_per_lang per language)."""
    html = """<!DOCTYPE html>
<html><head><title>Codec Benchmark A/B Comparison</title>
<style>
body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
table { border-collapse: collapse; width: 100%; margin: 20px 0; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
th { background: #f4f4f4; }
.lang-header { background: #2196F3; color: white; font-size: 18px; }
audio { width: 200px; }
.metric { font-size: 12px; color: #666; }
.best { background: #e8f5e9; font-weight: bold; }
</style></head><body>
<h1>Codec Benchmark: v9 JEPA vs DAC vs Mimi</h1>
<p>Click play to compare original vs reconstructed audio for each codec.</p>
"""

    current_lang = None
    lang_count = 0
    for r in results:
        if r["language"] != current_lang:
            lang_count = 0
        if r["sample_idx"] >= max_per_lang:
            continue
        if r["language"] != current_lang:
            if current_lang is not None:
                html += "</table>\n"
            current_lang = r["language"]
            lang_names = {"en": "English", "zh": "Chinese", "hi": "Hindi", "ja": "Japanese"}
            html += f'<h2>{lang_names.get(current_lang, current_lang)}</h2>\n'
            html += "<table><tr><th>#</th><th>Original</th>"
            for c in codecs:
                html += f"<th>{c}<br><span class='metric'>PESQ / STOI</span></th>"
            html += "</tr>\n"

        html += f"<tr><td>{r['sample_idx']}</td>"
        html += f"<td><audio controls src='audio/{r['language']}/original_{r['sample_idx']}.wav'></audio></td>"
        for c in codecs:
            pesq_val = r.get(f"{c}_pesq", "?")
            stoi_val = r.get(f"{c}_stoi", "?")
            pesq_str = f"{pesq_val:.3f}" if isinstance(pesq_val, float) else str(pesq_val)
            stoi_str = f"{stoi_val:.3f}" if isinstance(stoi_val, float) else str(stoi_val)
            html += f"<td><audio controls src='audio/{r['language']}/{c}_{r['sample_idx']}.wav'></audio>"
            html += f"<br><span class='metric'>{pesq_str} / {stoi_str}</span></td>"
        html += "</tr>\n"

    html += "</table>\n</body></html>"

    (output_dir / "comparison.html").write_text(html)
    print(f"[html] Saved comparison player to {output_dir / 'comparison.html'}")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Benchmark codecs across languages")
    parser.add_argument("--v9_ckpt", type=str, required=True,
                        help="Path to v9 12.5Hz stage2 checkpoint")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results")
    parser.add_argument("--languages", nargs="+", default=["en", "zh", "hi", "ja"])
    parser.add_argument("--samples_per_lang", type=int, default=50)
    parser.add_argument("--audio_samples", type=int, default=5,
                        help="Number of samples per language to save audio for HTML player")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_dac", action="store_true", help="Skip DAC benchmark")
    parser.add_argument("--skip_mimi", action="store_true", help="Skip Mimi benchmark")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── HF Login for private datasets ──
    from huggingface_hub import login
    login(token=os.environ.get("HF_TOKEN"))

    # ── Load codecs ──
    codecs = {}
    print("Loading codecs...")
    codecs["v9_jepa"] = V9Codec(args.v9_ckpt, args.device)

    if not args.skip_dac:
        try:
            codecs["dac"] = DACCodec("24khz", args.device)
        except Exception as e:
            print(f"[warn] DAC not available: {e}. Install with: pip install descript-audio-codec")

    if not args.skip_mimi:
        try:
            codecs["mimi"] = MimiCodec(args.device)
        except Exception as e:
            print(f"[warn] Mimi not available: {e}")

    # EnCodec at multiple bitrates for fair comparison
    for bw in [3.0, 6.0]:
        try:
            codecs[f"encodec_{bw}kbps"] = EnCodecWrapper(bandwidth=bw, device=args.device)
        except Exception as e:
            print(f"[warn] EnCodec {bw}kbps not available: {e}")

    codec_names = list(codecs.keys())
    print(f"Benchmarking codecs: {codec_names}")

    # ── Load data per language ──
    all_results = []
    n = args.samples_per_lang

    for lang in args.languages:
        print(f"\n{'='*60}")
        print(f"Language: {lang}")
        print(f"{'='*60}")

        if lang == "en":
            samples = load_librispeech_samples(n)
        elif lang == "hi":
            samples = load_hindi_samples(n)
        else:
            samples = load_commonvoice_samples(lang, n)

        # Create audio output dirs
        audio_dir = output_dir / "audio" / lang
        audio_dir.mkdir(parents=True, exist_ok=True)

        for idx, (wav, sr, meta) in enumerate(samples):
            wav_24k = resample_to_24k(wav, sr)
            save_audio = idx < args.audio_samples  # Only save first N for HTML player

            # Save original (only for audio comparison samples)
            if save_audio:
                sf.write(str(audio_dir / f"original_{idx}.wav"), wav_24k, SAMPLE_RATE)

            row = {
                "language": lang,
                "sample_idx": idx,
                "duration_s": len(wav_24k) / SAMPLE_RATE,
                "text": meta.get("text", "")[:100],
            }

            for codec_name, codec in codecs.items():
                try:
                    t0 = time.time()
                    rec = codec.reconstruct(wav_24k)
                    elapsed = time.time() - t0

                    # Save reconstructed audio (only for comparison samples)
                    if save_audio:
                        sf.write(str(audio_dir / f"{codec_name}_{idx}.wav"), rec, SAMPLE_RATE)

                    # Compute metrics (heavy metrics every 5th sample to save time)
                    heavy = (idx % 5 == 0)
                    metrics = compute_metrics(wav_24k, rec, SAMPLE_RATE, compute_heavy=heavy)
                    row[f"{codec_name}_pesq"] = metrics["pesq"]
                    row[f"{codec_name}_stoi"] = metrics["stoi"]
                    row[f"{codec_name}_mel"] = metrics["mel_distance"]
                    row[f"{codec_name}_time_ms"] = elapsed * 1000
                    if "utmos" in metrics:
                        row[f"{codec_name}_utmos"] = metrics["utmos"]
                    if "wer" in metrics:
                        row[f"{codec_name}_wer"] = metrics["wer"]
                    if "speaker_sim" in metrics:
                        row[f"{codec_name}_speaker_sim"] = metrics["speaker_sim"]

                except Exception as e:
                    print(f"  [error] {codec_name} sample {idx}: {e}")
                    row[f"{codec_name}_pesq"] = float("nan")
                    row[f"{codec_name}_stoi"] = float("nan")
                    row[f"{codec_name}_mel"] = float("nan")
                    row[f"{codec_name}_time_ms"] = float("nan")

            all_results.append(row)

            if (idx + 1) % 10 == 0:
                print(f"  [{lang}] {idx+1}/{len(samples)} samples processed")

    # ── Save CSV ──
    csv_path = output_dir / "benchmark_results.csv"
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n[csv] Saved {len(all_results)} results to {csv_path}")

    # ── Generate summary stats ──
    summary_path = output_dir / "summary.csv"
    summary_rows = []
    for lang in args.languages:
        lang_results = [r for r in all_results if r["language"] == lang]
        for codec_name in codec_names:
            def _gather(key):
                return [r[f"{codec_name}_{key}"] for r in lang_results
                        if not np.isnan(r.get(f"{codec_name}_{key}", float("nan")))]
            pesqs = _gather("pesq")
            stois = _gather("stoi")
            mels = _gather("mel")
            utmos_vals = _gather("utmos")
            wer_vals = _gather("wer")
            spksim_vals = _gather("speaker_sim")
            row = {
                "language": lang,
                "codec": codec_name,
                "n_samples": len(pesqs),
                "pesq_mean": np.mean(pesqs) if pesqs else float("nan"),
                "pesq_std": np.std(pesqs) if pesqs else float("nan"),
                "stoi_mean": np.mean(stois) if stois else float("nan"),
                "stoi_std": np.std(stois) if stois else float("nan"),
                "mel_mean": np.mean(mels) if mels else float("nan"),
                "mel_std": np.std(mels) if mels else float("nan"),
            }
            if utmos_vals:
                row["utmos_mean"] = np.mean(utmos_vals)
                row["utmos_std"] = np.std(utmos_vals)
            if wer_vals:
                row["wer_mean"] = np.mean(wer_vals)
                row["wer_std"] = np.std(wer_vals)
            if spksim_vals:
                row["speaker_sim_mean"] = np.mean(spksim_vals)
                row["speaker_sim_std"] = np.std(spksim_vals)
            summary_rows.append(row)

    if summary_rows:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"[csv] Saved summary to {summary_path}")

        # Print summary table
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"{'Language':<10} {'Codec':<15} {'PESQ':>8} {'STOI':>8} {'Mel':>8}")
        print("-" * 60)
        for r in summary_rows:
            print(f"{r['language']:<10} {r['codec']:<15} {r['pesq_mean']:>8.3f} {r['stoi_mean']:>8.3f} {r['mel_mean']:>8.3f}")

    # ── Generate HTML player ──
    generate_html_player(output_dir, all_results, codec_names, max_per_lang=args.audio_samples)

    # ── Save config ──
    config = {
        "codecs": codec_names,
        "languages": args.languages,
        "samples_per_lang": args.samples_per_lang,
        "v9_ckpt": args.v9_ckpt,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"\nBenchmark complete! Results in {output_dir}/")
    print(f"  - benchmark_results.csv (per-sample)")
    print(f"  - summary.csv (per-language aggregates)")
    print(f"  - comparison.html (A/B audio player)")
    print(f"  - audio/ (original + reconstructed WAVs)")


if __name__ == "__main__":
    main()
