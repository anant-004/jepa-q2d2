"""Evaluation metrics for the JEPA audio codec.

Three tiers of evaluation, designed for async CPU execution while GPU trains:

Every 1K steps (fast, CPU):
    - Mel Spectral Distance (multi-scale L1 on mel spectrograms)
    - Multi-Resolution STFT Distance
    - L1 waveform error

Every 5K steps (medium, CPU):
    - PESQ (perceptual quality, -0.5 to 4.5)
    - STOI (intelligibility, 0 to 1)
    - F0 correlation (pitch preservation)

Every 10K steps (heavier):
    - UTMOS (neural MOS prediction)
    - WER via Whisper (content preservation)
    - Speaker cosine similarity via WavLM

Usage:
    from koe.eval_codec import CodecEvaluator

    evaluator = CodecEvaluator(sample_rate=24000)

    # Fast metrics (every 1K steps)
    fast = evaluator.fast_metrics(original, reconstructed)
    # {'mel_distance': 0.42, 'stft_distance': 0.31, 'l1_error': 0.008}

    # Medium metrics (every 5K steps)
    medium = evaluator.medium_metrics(original, reconstructed)
    # {'pesq': 2.8, 'stoi': 0.91, 'f0_corr': 0.95}

    # Run eval on a set of samples and log to W&B
    evaluator.evaluate_and_log(codec, eval_samples, step=5000)
"""

import math
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# Mel spectrogram computation (no librosa dependency at import)
# ──────────────────────────────────────────────────────────────

def _mel_spectrogram(
    wav: np.ndarray,
    sr: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
) -> np.ndarray:
    """Compute log-mel spectrogram using librosa.

    Args:
        wav: 1D numpy array of audio samples
        sr: sample rate
        n_fft: FFT window size
        hop_length: hop size
        n_mels: number of mel bands

    Returns:
        Log-mel spectrogram [n_mels, T]
    """
    import librosa

    mel = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
    )
    return librosa.power_to_db(mel, ref=np.max)


# ──────────────────────────────────────────────────────────────
# Fast metrics (CPU, <100ms per sample)
# ──────────────────────────────────────────────────────────────

def mel_spectral_distance(
    original: np.ndarray,
    reconstructed: np.ndarray,
    sr: int = 24000,
    scales: List[Tuple[int, int]] = [(1024, 256), (2048, 512), (512, 128)],
    n_mels: int = 80,
) -> float:
    """Multi-scale mel spectral distance (L1 on log-mel spectrograms).

    Averages across multiple STFT window/hop sizes to capture
    different temporal resolutions.
    """
    # Match lengths
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]

    distances = []
    for n_fft, hop in scales:
        mel_orig = _mel_spectrogram(original, sr, n_fft, hop, n_mels)
        mel_rec = _mel_spectrogram(reconstructed, sr, n_fft, hop, n_mels)
        # Match time dimension
        min_t = min(mel_orig.shape[1], mel_rec.shape[1])
        dist = np.mean(np.abs(mel_orig[:, :min_t] - mel_rec[:, :min_t]))
        distances.append(dist)

    return float(np.mean(distances))


def stft_distance(
    original: np.ndarray,
    reconstructed: np.ndarray,
    scales: List[Tuple[int, int]] = [(1024, 256), (2048, 512), (512, 128)],
) -> float:
    """Multi-resolution STFT distance (spectral convergence + log magnitude).

    Standard codec evaluation metric from Encodec / SoundStream papers.
    """
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]

    total = 0.0
    for n_fft, hop in scales:
        orig_stft = np.abs(np.fft.rfft(
            np.lib.stride_tricks.sliding_window_view(original, n_fft)[::hop]
        ))
        rec_stft = np.abs(np.fft.rfft(
            np.lib.stride_tricks.sliding_window_view(reconstructed, n_fft)[::hop]
        ))
        min_t = min(orig_stft.shape[0], rec_stft.shape[0])
        orig_stft = orig_stft[:min_t]
        rec_stft = rec_stft[:min_t]

        # Spectral convergence
        sc = np.linalg.norm(orig_stft - rec_stft) / (np.linalg.norm(orig_stft) + 1e-8)
        # Log magnitude distance
        log_orig = np.log(np.maximum(orig_stft, 1e-8))
        log_rec = np.log(np.maximum(rec_stft, 1e-8))
        lm = np.mean(np.abs(log_orig - log_rec))
        total += sc + lm

    return float(total / len(scales))


def l1_waveform_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """L1 waveform error between original and reconstructed."""
    min_len = min(len(original), len(reconstructed))
    return float(np.mean(np.abs(original[:min_len] - reconstructed[:min_len])))


# ──────────────────────────────────────────────────────────────
# Medium metrics (CPU, ~1s per sample)
# ──────────────────────────────────────────────────────────────

def compute_pesq(
    original: np.ndarray,
    reconstructed: np.ndarray,
    sr: int = 24000,
) -> float:
    """Compute PESQ (Perceptual Evaluation of Speech Quality).

    Returns value in [-0.5, 4.5]. Higher is better.
    Requires 16kHz input, so we resample.
    """
    from pesq import pesq
    import librosa

    # PESQ requires 16kHz
    if sr != 16000:
        original = librosa.resample(original, orig_sr=sr, target_sr=16000)
        reconstructed = librosa.resample(reconstructed, orig_sr=sr, target_sr=16000)

    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]

    try:
        return float(pesq(16000, original, reconstructed, "wb"))
    except Exception:
        return float("nan")


def compute_stoi(
    original: np.ndarray,
    reconstructed: np.ndarray,
    sr: int = 24000,
) -> float:
    """Compute STOI (Short-Time Objective Intelligibility).

    Returns value in [0, 1]. Higher is better.
    """
    from pystoi import stoi

    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]

    try:
        return float(stoi(original, reconstructed, sr, extended=True))
    except Exception:
        return float("nan")


def compute_f0_correlation(
    original: np.ndarray,
    reconstructed: np.ndarray,
    sr: int = 24000,
) -> float:
    """Compute F0 (pitch) correlation using parselmouth/Praat.

    Returns Pearson correlation of F0 contours. Higher is better.
    Falls back to simple autocorrelation if parselmouth unavailable.
    """
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]

    try:
        import parselmouth

        snd_orig = parselmouth.Sound(original, sampling_frequency=sr)
        snd_rec = parselmouth.Sound(reconstructed, sampling_frequency=sr)

        pitch_orig = snd_orig.to_pitch(time_step=0.01)
        pitch_rec = snd_rec.to_pitch(time_step=0.01)

        f0_orig = pitch_orig.selected_array["frequency"]
        f0_rec = pitch_rec.selected_array["frequency"]

        min_f = min(len(f0_orig), len(f0_rec))
        f0_orig = f0_orig[:min_f]
        f0_rec = f0_rec[:min_f]

        # Only correlate voiced frames (F0 > 0)
        voiced = (f0_orig > 0) & (f0_rec > 0)
        if voiced.sum() < 5:
            return float("nan")

        corr = np.corrcoef(f0_orig[voiced], f0_rec[voiced])[0, 1]
        return float(corr)

    except ImportError:
        return float("nan")


# ──────────────────────────────────────────────────────────────
# W&B logging helpers
# ──────────────────────────────────────────────────────────────

def log_audio_to_wandb(
    original: np.ndarray,
    reconstructed: np.ndarray,
    sr: int,
    step: int,
    prefix: str = "eval",
):
    """Log audio samples and spectrograms to W&B."""
    try:
        import wandb
        if not wandb.run:
            return
    except ImportError:
        return

    wandb.log({
        f"{prefix}/audio_original": wandb.Audio(original, sample_rate=sr),
        f"{prefix}/audio_reconstructed": wandb.Audio(reconstructed, sample_rate=sr),
    }, step=step)

    # Side-by-side mel spectrograms
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        mel_orig = _mel_spectrogram(original, sr)
        mel_rec = _mel_spectrogram(reconstructed, sr)

        ax1.imshow(mel_orig, aspect="auto", origin="lower")
        ax1.set_title("Original")
        ax1.set_ylabel("Mel bin")

        ax2.imshow(mel_rec, aspect="auto", origin="lower")
        ax2.set_title("Reconstructed")

        plt.tight_layout()
        wandb.log({f"{prefix}/spectrograms": wandb.Image(fig)}, step=step)
        plt.close(fig)
    except Exception:
        pass


def log_fsq_analytics_to_wandb(
    tokens: np.ndarray,
    step: int,
    num_groups: int = 19,
    num_tokens: int = 16384,
    prefix: str = "eval",
):
    """Log FSQ token analytics to W&B.

    Args:
        tokens: [B, T_z, num_groups] packed token values in [0, num_tokens-1]
        step: training step
        num_groups: number of FSQ groups (19)
        num_tokens: max token value + 1 (16384)
    """
    try:
        import wandb
        if not wandb.run:
            return
    except ImportError:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Token usage histogram per group
    fig, axes = plt.subplots(4, 5, figsize=(20, 12))
    axes = axes.flatten()
    utilization_rates = []
    entropies = []

    for g in range(num_groups):
        group_tokens = tokens[:, :, g].flatten()
        unique_count = len(np.unique(group_tokens))
        utilization = unique_count / num_tokens
        utilization_rates.append(utilization)

        # Shannon entropy
        counts = np.bincount(group_tokens.astype(np.int64), minlength=num_tokens)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        entropies.append(entropy)

        if g < len(axes):
            axes[g].hist(group_tokens, bins=min(100, num_tokens), density=True)
            axes[g].set_title(f"G{g} util={utilization:.1%}")
            axes[g].set_xlabel("")

    # Hide unused subplot
    if num_groups < len(axes):
        for i in range(num_groups, len(axes)):
            axes[i].set_visible(False)

    plt.suptitle(f"Token Usage per Group (step {step})")
    plt.tight_layout()
    wandb.log({f"{prefix}/token_usage_hist": wandb.Image(fig)}, step=step)
    plt.close(fig)

    # Utilization rate
    wandb.log({
        f"{prefix}/token_utilization_mean": float(np.mean(utilization_rates)),
        f"{prefix}/token_utilization_min": float(np.min(utilization_rates)),
    }, step=step)

    # Entropy per group
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(num_groups), entropies)
    ax.set_xlabel("Group")
    ax.set_ylabel("Shannon Entropy (bits)")
    ax.set_title(f"Token Entropy per Group (step {step})")
    plt.tight_layout()
    wandb.log({f"{prefix}/token_entropy": wandb.Image(fig)}, step=step)
    plt.close(fig)

    # Token sequence heatmap (first sample)
    if tokens.shape[0] > 0:
        fig, ax = plt.subplots(figsize=(12, 4))
        sample = tokens[0]  # [T_z, num_groups]
        ax.imshow(sample.T, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_xlabel("Time frame")
        ax.set_ylabel("Group")
        ax.set_title(f"Token Sequence Heatmap (step {step})")
        plt.tight_layout()
        wandb.log({f"{prefix}/token_heatmap": wandb.Image(fig)}, step=step)
        plt.close(fig)


def log_fsq_codebook_to_wandb(
    indices: np.ndarray,
    z_e: Optional[np.ndarray],
    step: int,
    num_levels: int = 4,
    group_size: int = 7,
    prefix: str = "eval",
):
    """Log per-dimension FSQ codebook analytics to W&B.

    Complements log_fsq_analytics_to_wandb() which operates on packed tokens.
    This function operates on raw unpacked FSQ indices to show the health
    of individual quantization dimensions.

    Args:
        indices: [B, T, D=128] raw FSQ indices (0 to num_levels-1 per dim)
        z_e: [B, D=128, T] continuous encoder output before FSQ (optional)
        step: training step
        num_levels: FSQ levels per dimension (4)
        group_size: dims per group for grouping analysis (7)
        prefix: W&B logging prefix
    """
    try:
        import wandb
        if not wandb.run:
            return
    except ImportError:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    D = indices.shape[-1]  # 128
    flat = indices.reshape(-1, D)  # [B*T, D]
    num_samples = flat.shape[0]

    # ── Per-dim level distribution heatmap [D × num_levels] ──
    dist = np.zeros((D, num_levels), dtype=np.float64)
    for d in range(D):
        counts = np.bincount(flat[:, d].astype(np.int64), minlength=num_levels)[:num_levels]
        dist[d] = counts / num_samples

    fig, ax = plt.subplots(figsize=(6, 16))
    im = ax.imshow(dist, aspect="auto", cmap="YlOrRd", interpolation="nearest",
                   vmin=0, vmax=dist.max())
    ax.set_xlabel("FSQ Level")
    ax.set_ylabel("Dimension")
    ax.set_xticks(range(num_levels))
    ax.set_title(f"Per-Dim Level Distribution (step {step})")
    plt.colorbar(im, ax=ax, label="Frequency")
    plt.tight_layout()
    wandb.log({f"{prefix}/codebook_heatmap": wandb.Image(fig)}, step=step)
    plt.close(fig)

    # ── Dead code rate ──
    counts_raw = np.zeros((D, num_levels), dtype=np.int64)
    for d in range(D):
        counts_raw[d] = np.bincount(flat[:, d].astype(np.int64), minlength=num_levels)[:num_levels]
    dead_pairs = (counts_raw == 0).sum()
    total_pairs = D * num_levels
    dead_rate = dead_pairs / total_pairs
    wandb.log({f"{prefix}/dead_code_rate": float(dead_rate)}, step=step)

    # ── Codebook balance score (mean Gini coefficient across dims) ──
    gini_values = []
    for d in range(D):
        freq = counts_raw[d].astype(np.float64)
        freq_sorted = np.sort(freq)
        n = len(freq_sorted)
        cum = np.cumsum(freq_sorted)
        total = freq_sorted.sum()
        if total > 0:
            gini = (2.0 * np.sum((np.arange(1, n + 1) * freq_sorted)) - (n + 1) * total) / (n * total)
            gini_values.append(max(0.0, gini))
        else:
            gini_values.append(1.0)
    mean_gini = float(np.mean(gini_values))
    wandb.log({f"{prefix}/codebook_balance_gini": mean_gini}, step=step)

    # ── Per-group variance of z_e ──
    if z_e is not None:
        # z_e: [B, D=128, T] → reshape to [B*T, D]
        z_flat = z_e.transpose(0, 2, 1).reshape(-1, D)  # [B*T, D]
        num_groups = (D + group_size - 1) // group_size
        group_vars = []
        for g in range(num_groups):
            start = g * group_size
            end = min(start + group_size, D)
            group_var = np.var(z_flat[:, start:end])
            group_vars.append(float(group_var))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(num_groups), group_vars)
        ax.set_xlabel("Group")
        ax.set_ylabel("Variance of z_e")
        ax.set_title(f"Per-Group Encoder Variance (step {step})")
        plt.tight_layout()
        wandb.log({f"{prefix}/group_variance": wandb.Image(fig)}, step=step)
        plt.close(fig)

        wandb.log({
            f"{prefix}/z_e_variance_mean": float(np.mean(group_vars)),
            f"{prefix}/z_e_variance_std": float(np.std(group_vars)),
        }, step=step)


# ──────────────────────────────────────────────────────────────
# CodecEvaluator — main evaluation class
# ──────────────────────────────────────────────────────────────

def compute_utmos(
    waveform: np.ndarray,
    sr: int = 24000,
    _cache: dict = {},
) -> float:
    """Compute UTMOS (neural MOS prediction) via torch.hub.

    Returns value in [1.0, 5.0]. Higher is better.
    Model is lazily loaded and cached across calls.
    """
    try:
        if "predictor" not in _cache:
            predictor = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
            )
            predictor.eval()
            _cache["predictor"] = predictor

        predictor = _cache["predictor"]
        wav_tensor = torch.from_numpy(waveform).unsqueeze(0).float()  # [1, T]
        with torch.no_grad():
            score = predictor(wav_tensor, sr)
        return float(score.item())
    except Exception as e:
        return float("nan")


class CodecEvaluator:
    """Evaluate codec reconstruction quality with tiered metrics.

    Designed to run asynchronously on CPU threads while GPU trains.
    """

    def __init__(self, sample_rate: int = 24000):
        self.sr = sample_rate
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._pending_future: Optional[Future] = None

    def fast_metrics(
        self, original: np.ndarray, reconstructed: np.ndarray,
    ) -> Dict[str, float]:
        """Fast metrics: mel distance, STFT distance, L1 error. ~100ms/sample."""
        return {
            "mel_distance": mel_spectral_distance(original, reconstructed, self.sr),
            "stft_distance": stft_distance(original, reconstructed),
            "l1_error": l1_waveform_error(original, reconstructed),
        }

    def medium_metrics(
        self, original: np.ndarray, reconstructed: np.ndarray,
    ) -> Dict[str, float]:
        """Medium metrics: PESQ, STOI, F0 correlation. ~1s/sample."""
        return {
            "pesq": compute_pesq(original, reconstructed, self.sr),
            "stoi": compute_stoi(original, reconstructed, self.sr),
            "f0_corr": compute_f0_correlation(original, reconstructed, self.sr),
        }

    def heavy_metrics(
        self, reconstructed: np.ndarray,
    ) -> Dict[str, float]:
        """Heavy metrics: UTMOS (neural MOS). ~2-5s/sample on CPU."""
        return {
            "utmos": compute_utmos(reconstructed, self.sr),
        }

    def evaluate_batch(
        self,
        originals: List[np.ndarray],
        reconstructeds: List[np.ndarray],
        include_medium: bool = False,
        include_heavy: bool = False,
    ) -> Dict[str, float]:
        """Evaluate a batch of samples and return averaged metrics.

        Args:
            originals: list of original audio numpy arrays
            reconstructeds: list of reconstructed audio numpy arrays
            include_medium: if True, also compute PESQ/STOI/F0
            include_heavy: if True, also compute UTMOS

        Returns:
            Dict of averaged metric values
        """
        all_metrics: Dict[str, List[float]] = {}

        for orig, rec in zip(originals, reconstructeds):
            fast = self.fast_metrics(orig, rec)
            for k, v in fast.items():
                all_metrics.setdefault(k, []).append(v)

            if include_medium:
                med = self.medium_metrics(orig, rec)
                for k, v in med.items():
                    if not math.isnan(v):
                        all_metrics.setdefault(k, []).append(v)

            if include_heavy:
                heavy = self.heavy_metrics(rec)
                for k, v in heavy.items():
                    if not math.isnan(v):
                        all_metrics.setdefault(k, []).append(v)

        return {k: float(np.mean(v)) for k, v in all_metrics.items()}

    def evaluate_and_log(
        self,
        codec,
        eval_wavs: List[torch.Tensor],
        step: int,
        include_medium: bool = False,
        include_heavy: bool = False,
        wandb_prefix: str = "eval",
    ) -> Dict[str, float]:
        """Run codec on eval samples, compute metrics, log to W&B.

        Args:
            codec: KoeCodec instance (on CPU or GPU)
            eval_wavs: list of [1, 1, T] waveform tensors
            step: training step
            include_medium: compute PESQ/STOI/F0
            include_heavy: compute UTMOS
            wandb_prefix: W&B log prefix

        Returns:
            Dict of averaged metrics
        """
        originals = []
        reconstructeds = []

        device = next(codec.parameters()).device

        for wav in eval_wavs:
            wav = wav.to(device)
            with torch.no_grad():
                rec = codec.roundtrip(wav)

            orig_np = wav[0, 0].cpu().numpy()
            rec_np = rec[0, 0].cpu().numpy()
            originals.append(orig_np)
            reconstructeds.append(rec_np)

        metrics = self.evaluate_batch(
            originals, reconstructeds, include_medium, include_heavy
        )

        # Log metrics to W&B
        try:
            import wandb
            if wandb.run:
                wandb.log(
                    {f"{wandb_prefix}/{k}": v for k, v in metrics.items()},
                    step=step,
                )
        except (ImportError, Exception):
            pass

        # Log first sample's audio + spectrograms
        if originals:
            log_audio_to_wandb(
                originals[0], reconstructeds[0], self.sr, step, wandb_prefix,
            )

        return metrics

    def evaluate_async(
        self,
        codec_state_dict: Dict,
        codec_config,
        eval_wavs: List[torch.Tensor],
        step: int,
        include_medium: bool = False,
    ) -> Future:
        """Run evaluation asynchronously on a CPU thread.

        Returns a Future that resolves to the metrics dict.
        This allows the GPU to continue training while eval runs.

        Args:
            codec_state_dict: state dict (already on CPU)
            codec_config: CodecConfig instance
            eval_wavs: list of eval waveforms (already on CPU)
            step: training step
            include_medium: compute PESQ/STOI/F0
        """
        def _run():
            from koe.codec import KoeCodec

            # Reconstruct codec on CPU
            codec = KoeCodec(codec_config)
            codec.model.load_state_dict(codec_state_dict)
            codec.eval()

            return self.evaluate_and_log(
                codec, eval_wavs, step,
                include_medium=include_medium,
            )

        future = self._executor.submit(_run)
        self._pending_future = future
        return future

    def wait_for_pending(self, timeout: float = 300.0) -> Optional[Dict[str, float]]:
        """Wait for any pending async evaluation to complete.

        Returns metrics dict or None if no pending eval.
        """
        if self._pending_future is None:
            return None
        try:
            result = self._pending_future.result(timeout=timeout)
            self._pending_future = None
            return result
        except Exception as e:
            print(f"[eval] Async eval failed: {e}")
            self._pending_future = None
            return None
