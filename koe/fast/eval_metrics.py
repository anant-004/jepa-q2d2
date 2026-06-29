"""Enhanced evaluation metrics for Stage 2 decoder training.

Wraps and extends the existing CodecEvaluator from koe.eval_codec with:
- WER via Whisper-small (content preservation)
- Speaker cosine similarity via WavLM (speaker identity preservation)
- Tiered eval schedule logic (3 tiers at different frequencies)
- W&B audio sample logging (reconstructed audio as wandb.Audio)

Usage:
    from koe.fast.eval_metrics import EnhancedCodecEvaluator

    evaluator = EnhancedCodecEvaluator(eval_every=1000, num_samples=5)

    if evaluator.should_eval(step):
        metrics = evaluator.evaluate(codec, eval_wavs, step, device)
"""

import math
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Optional

import numpy as np
import torch

from koe.eval_codec import (
    mel_spectral_distance,
    stft_distance,
    l1_waveform_error,
    compute_pesq,
    compute_stoi,
    compute_f0_correlation,
    compute_utmos,
    log_audio_to_wandb,
)


# ──────────────────────────────────────────────────────────
# Lazy-loaded model caches (module-level, persist across calls)
# ──────────────────────────────────────────────────────────

_whisper_cache: Dict = {}
_speaker_cache: Dict = {}
_warned: Dict[str, bool] = {}


def _warn_once(key: str, msg: str):
    """Print a warning message only on first occurrence."""
    if key not in _warned:
        _warned[key] = True
        print(f"[eval_metrics] WARNING: {msg}")


# ──────────────────────────────────────────────────────────
# WER via Whisper-small
# ──────────────────────────────────────────────────────────

def compute_wer(
    original: np.ndarray,
    reconstructed: np.ndarray,
    sr: int = 24000,
) -> float:
    """Compute WER using Whisper-small transcription.

    Transcribes both original and reconstructed audio, then computes
    word error rate between the two transcriptions. This measures how
    well the codec preserves speech content.

    Uses lazy-loaded Whisper model (cached across calls). The model
    runs on CPU to avoid competing with the training GPU.

    Args:
        original: 1D numpy array of original audio samples.
        reconstructed: 1D numpy array of reconstructed audio samples.
        sr: Sample rate of the audio (default 24000).

    Returns:
        WER as a float (0.0 = perfect, 1.0+ = poor).
        Returns NaN if whisper or jiwer are not installed.
    """
    try:
        import whisper
    except ImportError:
        _warn_once("whisper", "whisper not installed. WER will be NaN. pip install openai-whisper")
        return float("nan")

    try:
        import jiwer
    except ImportError:
        _warn_once("jiwer", "jiwer not installed. WER will be NaN. pip install jiwer")
        return float("nan")

    try:
        # Lazy-load whisper model (CPU only)
        if "model" not in _whisper_cache:
            print("[eval_metrics] Loading Whisper-small (first use)...")
            model = whisper.load_model("small", device="cpu")
            model.eval()
            _whisper_cache["model"] = model

        model = _whisper_cache["model"]

        # Whisper expects 16kHz float32 audio
        if sr != 16000:
            import librosa
            original_16k = librosa.resample(original.astype(np.float32), orig_sr=sr, target_sr=16000)
            reconstructed_16k = librosa.resample(reconstructed.astype(np.float32), orig_sr=sr, target_sr=16000)
        else:
            original_16k = original.astype(np.float32)
            reconstructed_16k = reconstructed.astype(np.float32)

        # Transcribe both
        with torch.no_grad():
            result_orig = model.transcribe(original_16k, language="en", fp16=False)
            result_rec = model.transcribe(reconstructed_16k, language="en", fp16=False)

        text_orig = result_orig["text"].strip()
        text_rec = result_rec["text"].strip()

        # Handle empty transcriptions
        if not text_orig and not text_rec:
            return 0.0  # Both empty = perfect match
        if not text_orig:
            # Original is empty but reconstruction produced text
            return 1.0

        wer = jiwer.wer(text_orig, text_rec)
        return float(wer)

    except Exception as e:
        _warn_once("wer_error", f"WER computation failed: {e}")
        return float("nan")


# ──────────────────────────────────────────────────────────
# Speaker cosine similarity via WavLM
# ──────────────────────────────────────────────────────────

def compute_speaker_similarity(
    original: np.ndarray,
    reconstructed: np.ndarray,
    sr: int = 24000,
) -> float:
    """Compute speaker cosine similarity using WavLM features.

    Extracts speaker embeddings from both original and reconstructed
    audio using a WavLM-based speaker verification model, then returns
    the cosine similarity between embeddings. This measures how well
    the codec preserves speaker identity.

    Uses lazy-loaded model (cached across calls). Runs on CPU.

    Args:
        original: 1D numpy array of original audio samples.
        reconstructed: 1D numpy array of reconstructed audio samples.
        sr: Sample rate of the audio (default 24000).

    Returns:
        Cosine similarity as a float in [-1, 1]. Higher is better.
        Returns NaN if required libraries are not installed.
    """
    try:
        import torchaudio
        from torchaudio.pipelines import WAVLM_BASE
    except ImportError:
        _warn_once("wavlm", "torchaudio not available. Speaker similarity will be NaN.")
        return float("nan")

    try:
        # Lazy-load WavLM model (CPU only)
        if "model" not in _speaker_cache:
            print("[eval_metrics] Loading WavLM-Base for speaker similarity (first use)...")
            bundle = WAVLM_BASE
            model = bundle.get_model()
            model.eval()
            _speaker_cache["model"] = model
            _speaker_cache["sample_rate"] = bundle.sample_rate

        model = _speaker_cache["model"]
        target_sr = _speaker_cache["sample_rate"]  # 16000

        # Resample to model's expected sample rate
        orig_tensor = torch.from_numpy(original.astype(np.float32)).unsqueeze(0)  # [1, T]
        rec_tensor = torch.from_numpy(reconstructed.astype(np.float32)).unsqueeze(0)  # [1, T]

        if sr != target_sr:
            orig_tensor = torchaudio.functional.resample(orig_tensor, sr, target_sr)
            rec_tensor = torchaudio.functional.resample(rec_tensor, sr, target_sr)

        with torch.no_grad():
            # Extract features: returns list of tensors, take last layer
            orig_features, _ = model.extract_features(orig_tensor)
            rec_features, _ = model.extract_features(rec_tensor)

            # Use last hidden state, mean-pool over time for speaker embedding
            orig_emb = orig_features[-1].mean(dim=1)  # [1, D]
            rec_emb = rec_features[-1].mean(dim=1)  # [1, D]

            # Cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(orig_emb, rec_emb, dim=1)

        return float(cos_sim.item())

    except Exception as e:
        _warn_once("speaker_error", f"Speaker similarity computation failed: {e}")
        return float("nan")


# ──────────────────────────────────────────────────────────
# EnhancedCodecEvaluator
# ──────────────────────────────────────────────────────────

class EnhancedCodecEvaluator:
    """Tiered evaluation for Stage 2 decoder training.

    Tier 1 (every eval_every steps): mel_distance, stft_distance, L1
    Tier 2 (every 5x eval_every): PESQ (primary), STOI, F0 corr
    Tier 3 (every 10x eval_every): UTMOS, WER, speaker_sim

    Heavy models (Whisper, WavLM, UTMOS) are loaded lazily on first use
    and cached at module level. All heavy inference runs on CPU via a
    ThreadPoolExecutor to avoid competing with the training GPU.

    Example:
        evaluator = EnhancedCodecEvaluator(eval_every=1000, num_samples=5)

        for step in range(num_steps):
            train_step(...)
            if evaluator.should_eval(step):
                metrics = evaluator.evaluate(codec, eval_wavs, step, device)
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        eval_every: int = 1000,
        num_samples: int = 5,
        log_audio: bool = True,
    ):
        self.sr = sample_rate
        self.eval_every = eval_every
        self.num_samples = num_samples
        self.log_audio = log_audio
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._pending_future: Optional[Future] = None

    def should_eval(self, step: int) -> bool:
        """Whether to run any eval at this step."""
        if step <= 0:
            return False
        return step % self.eval_every == 0

    def eval_tier(self, step: int) -> int:
        """Return the highest tier to run at this step: 1, 2, or 3.

        Higher tiers include all lower tiers.
        """
        if step % (10 * self.eval_every) == 0:
            return 3
        if step % (5 * self.eval_every) == 0:
            return 2
        return 1

    def _reconstruct_samples(
        self,
        codec,
        eval_wavs: List[torch.Tensor],
        device: torch.device,
    ) -> List[tuple]:
        """Run codec forward pass on eval samples.

        Returns list of (original_np, reconstructed_np) tuples.
        """
        codec.eval()
        pairs = []
        samples = eval_wavs[: self.num_samples]

        for wav in samples:
            wav = wav.to(device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # WaveformJEPAFSQVAE.forward returns (rec, indices, aux_loss, z_e)
                rec, _indices, _aux_loss, _z_e = codec(wav)

            orig_np = wav[0, 0].cpu().float().numpy()
            rec_np = rec[0, 0].cpu().float().numpy()

            # Match lengths
            min_len = min(len(orig_np), len(rec_np))
            orig_np = orig_np[:min_len]
            rec_np = rec_np[:min_len]

            pairs.append((orig_np, rec_np))

        codec.train()
        return pairs

    def _compute_tier1(self, pairs: List[tuple]) -> Dict[str, float]:
        """Tier 1: fast metrics (mel_distance, stft_distance, L1)."""
        results: Dict[str, List[float]] = {}

        for orig, rec in pairs:
            results.setdefault("mel_distance", []).append(
                mel_spectral_distance(orig, rec, self.sr)
            )
            results.setdefault("stft_distance", []).append(
                stft_distance(orig, rec)
            )
            results.setdefault("l1_error", []).append(
                l1_waveform_error(orig, rec)
            )

        return {k: float(np.mean(v)) for k, v in results.items()}

    def _compute_tier2(self, pairs: List[tuple]) -> Dict[str, float]:
        """Tier 2: medium metrics (PESQ, STOI, F0 correlation)."""
        results: Dict[str, List[float]] = {}

        for orig, rec in pairs:
            pesq_val = compute_pesq(orig, rec, self.sr)
            stoi_val = compute_stoi(orig, rec, self.sr)
            f0_val = compute_f0_correlation(orig, rec, self.sr)

            if not math.isnan(pesq_val):
                results.setdefault("pesq", []).append(pesq_val)
            if not math.isnan(stoi_val):
                results.setdefault("stoi", []).append(stoi_val)
            if not math.isnan(f0_val):
                results.setdefault("f0_corr", []).append(f0_val)

        return {k: float(np.mean(v)) for k, v in results.items() if v}

    def _compute_tier3(self, pairs: List[tuple]) -> Dict[str, float]:
        """Tier 3: heavy metrics (UTMOS, WER, speaker similarity)."""
        results: Dict[str, List[float]] = {}

        for orig, rec in pairs:
            utmos_val = compute_utmos(rec, self.sr)
            wer_val = compute_wer(orig, rec, self.sr)
            spk_val = compute_speaker_similarity(orig, rec, self.sr)

            if not math.isnan(utmos_val):
                results.setdefault("utmos", []).append(utmos_val)
            if not math.isnan(wer_val):
                results.setdefault("wer", []).append(wer_val)
            if not math.isnan(spk_val):
                results.setdefault("speaker_sim", []).append(spk_val)

        return {k: float(np.mean(v)) for k, v in results.items() if v}

    def _log_to_wandb(
        self,
        metrics: Dict[str, float],
        pairs: List[tuple],
        step: int,
    ):
        """Log metrics and audio samples to W&B."""
        try:
            import wandb
            if not wandb.run:
                return
        except ImportError:
            return

        wandb.log(
            {f"eval/{k}": v for k, v in metrics.items()},
            step=step,
        )

        if self.log_audio and pairs:
            num_audio = min(len(pairs), 5)
            audio_log = {}

            for i in range(num_audio):
                orig, rec = pairs[i]
                audio_log[f"eval/audio_original_{i}"] = wandb.Audio(
                    orig, sample_rate=self.sr, caption=f"Original #{i}"
                )
                audio_log[f"eval/audio_reconstructed_{i}"] = wandb.Audio(
                    rec, sample_rate=self.sr, caption=f"Reconstructed #{i}"
                )

            wandb.log(audio_log, step=step)

            log_audio_to_wandb(
                pairs[0][0], pairs[0][1], self.sr, step, prefix="eval"
            )

    def evaluate(
        self,
        codec,
        eval_wavs: List[torch.Tensor],
        step: int,
        device: torch.device,
    ) -> Dict[str, float]:
        """Run tiered evaluation and log to W&B.

        Args:
            codec: WaveformJEPAFSQVAE model instance.
            eval_wavs: List of [1, 1, T] waveform tensors.
            step: Current training step.
            device: Torch device the codec is on.

        Returns:
            Dict of metric name -> value for all computed metrics.
        """
        tier = self.eval_tier(step)
        pairs = self._reconstruct_samples(codec, eval_wavs, device)

        if not pairs:
            return {}

        metrics = self._compute_tier1(pairs)

        if tier >= 2:
            tier2 = self._compute_tier2(pairs)
            metrics.update(tier2)

        if tier >= 3:
            tier3 = self._compute_tier3(pairs)
            metrics.update(tier3)

        self._log_to_wandb(metrics, pairs, step)

        tier_label = f"Tier {tier}"
        metric_str = ", ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
        print(f"[eval] Step {step} ({tier_label}): {metric_str}")

        return metrics

    def evaluate_async(
        self,
        codec,
        eval_wavs: List[torch.Tensor],
        step: int,
        device: torch.device,
    ) -> Future:
        """Run evaluation asynchronously on a CPU thread.

        The codec forward pass (GPU) happens synchronously to get
        reconstructions, then metric computation is offloaded to a
        background CPU thread so training can resume immediately.

        Returns:
            Future that resolves to the metrics dict.
        """
        pairs = self._reconstruct_samples(codec, eval_wavs, device)
        tier = self.eval_tier(step)

        def _run_metrics():
            if not pairs:
                return {}

            metrics = self._compute_tier1(pairs)

            if tier >= 2:
                tier2 = self._compute_tier2(pairs)
                metrics.update(tier2)

            if tier >= 3:
                tier3 = self._compute_tier3(pairs)
                metrics.update(tier3)

            self._log_to_wandb(metrics, pairs, step)

            tier_label = f"Tier {tier}"
            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
            print(f"[eval] Step {step} ({tier_label}): {metric_str}")

            return metrics

        future = self._executor.submit(_run_metrics)
        self._pending_future = future
        return future

    def wait_for_pending(self, timeout: float = 600.0) -> Optional[Dict[str, float]]:
        """Wait for any pending async evaluation to complete."""
        if self._pending_future is None:
            return None
        try:
            result = self._pending_future.result(timeout=timeout)
            self._pending_future = None
            return result
        except Exception as e:
            print(f"[eval_metrics] Async eval failed: {e}")
            self._pending_future = None
            return None
