"""Train the JEPA audio tokenizer (Phase 1 of KoeTTS).

Production-grade training pipeline with:
- Two-stage training (JEPA encoder → FSQ+GAN decoder)
- DeepSpeed ZeRO-2 on 4x H200 via Modal
- AdamW-only default (proven stable), Muon optional
- Async CPU evaluation while GPU trains (no idle time)
- HuggingFace Hub for model weight storage (private repo)
- Auto-resume from HF checkpoint on preemption
- NaN/Inf detection → rollback to last checkpoint
- Discriminator collapse detection → adjust λ_gan/disc_lr
- Comprehensive W&B logging: losses, LR, throughput, memory, eval, audio
- Robust checkpointing to Modal Volume + HF Hub

Usage:
    # Stage 1 — JEPA encoder
    deepspeed --num_gpus=4 -m koe.train_tokenizer --stage 1 \
        --data_dir /data/librilight/medium \
        --output_dir /checkpoints/tokenizer \
        --deepspeed configs/ds_zero2_stage1.json

    # Stage 2 — FSQ + Decoder + GAN
    deepspeed --num_gpus=4 -m koe.train_tokenizer --stage 2 \
        --data_dir /data/librilight/medium \
        --output_dir /checkpoints/tokenizer \
        --stage1_ckpt hf://Andy004/koe-tokenizer/stage1_final.pt \
        --deepspeed configs/ds_zero2_stage2.json

    # Both stages sequentially
    deepspeed --num_gpus=4 -m koe.train_tokenizer --stage both \
        --data_dir /data/librilight/medium \
        --output_dir /checkpoints/tokenizer \
        --deepspeed configs/ds_zero2_stage1.json
"""

import argparse
import gc
import math
import os
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import deepspeed
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from koe.config import CodecConfig, TokenizerTrainConfig
from koe.codec_impl import (
    JEPAEncoder,
    WaveformJEPAFSQVAE,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    MRSTFTLoss,
    create_jepa_mask,
    jepa_time_len_from_wav,
    feature_loss,
    discriminator_loss,
    generator_loss,
    make_collate_fn,
    set_requires_grad,
)


# ──────────────────────────────────────────────────────────────
# Audio dataset — loads raw waveforms from a directory of audio files
# ──────────────────────────────────────────────────────────────

class AudioDataset(torch.utils.data.Dataset):
    """Simple audio dataset: recursively find audio files, load and crop.

    Features:
    - Random crop to max_seconds for variable-length training
    - Automatic resampling to target sample rate
    - Mono conversion for multi-channel audio
    """

    def __init__(self, data_dir: str, sample_rate: int = 24000, max_seconds: float = 15.0):
        self.sample_rate = sample_rate
        self.max_samples = int(max_seconds * sample_rate)
        self.files = []
        for ext in ("*.wav", "*.flac", "*.mp3", "*.ogg"):
            self.files.extend(sorted(Path(data_dir).rglob(ext)))
        if not self.files:
            raise ValueError(f"No audio files found in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        import torchaudio
        path = str(self.files[idx])
        try:
            info = torchaudio.info(path)
            sr = info.sample_rate
            total_frames = info.num_frames
            need_frames = int(self.max_samples * sr / self.sample_rate) if sr != self.sample_rate else self.max_samples

            # Partial read: only load the crop we need (huge speedup for long files)
            if total_frames > need_frames:
                start = torch.randint(0, total_frames - need_frames, (1,)).item()
                wav, sr = torchaudio.load(path, frame_offset=start, num_frames=need_frames)
            else:
                wav, sr = torchaudio.load(path)
        except Exception:
            # Fallback: load entire file if info/partial read fails
            wav, sr = torchaudio.load(path)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        wav = wav.squeeze(0)  # [T]
        if wav.shape[0] > self.max_samples:
            start = torch.randint(0, wav.shape[0] - self.max_samples, (1,)).item()
            wav = wav[start : start + self.max_samples]
        return wav


class CachedEncoderDataset(torch.utils.data.Dataset):
    """Dataset that loads pre-computed encoder outputs + raw audio.

    Skips the frozen encoder forward pass during Stage 2 training (~40% speedup).
    Each item returns (wav_tensor, z_e_tensor) where z_e is [D, T_z].
    """

    def __init__(self, data_dir: str, cache_dir: str, sample_rate: int = 24000,
                 max_seconds: float = 15.0, hop_length: int = 9600):
        self.sample_rate = sample_rate
        self.max_samples = int(max_seconds * sample_rate)
        self.hop_length = hop_length

        self.files = []
        for ext in ("*.wav", "*.flac", "*.mp3", "*.ogg"):
            self.files.extend(sorted(Path(data_dir).rglob(ext)))
        if not self.files:
            raise ValueError(f"No audio files found in {data_dir}")

        self.cache_dir = Path(cache_dir)
        # Verify cache exists
        n_cached = len(list(self.cache_dir.glob("*.pt")))
        if n_cached == 0:
            raise ValueError(f"No cached .pt files in {cache_dir}. Run with --cache_encoder first.")
        if n_cached != len(self.files):
            print(f"WARNING: {n_cached} cached files vs {len(self.files)} audio files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        import torchaudio
        # Load audio (still needed for discriminator)
        wav, sr = torchaudio.load(str(self.files[idx]))
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        wav = wav.squeeze(0)  # [T]

        # Load cached z_e
        cache_path = self.cache_dir / f"{idx:06d}.pt"
        z_e = torch.load(cache_path, map_location="cpu", weights_only=True)  # [D, T_z_full]

        # Random crop in both wav and z_e space (synchronized)
        if wav.shape[0] > self.max_samples:
            max_start = wav.shape[0] - self.max_samples
            start_sample = torch.randint(0, max_start, (1,)).item()
            wav = wav[start_sample : start_sample + self.max_samples]
            # Convert to z_e frame index
            start_frame = start_sample // self.hop_length
            n_frames = self.max_samples // self.hop_length
            z_e = z_e[:, start_frame : start_frame + n_frames]

        return wav, z_e


def _make_cached_collate_fn(sample_rate: int, hop_length: int):
    """Collate for (wav, z_e) pairs with synchronized padding."""

    def collate_fn(batch):
        if not batch:
            return None, None
        wavs, z_es = zip(*batch)
        # Pad wavs
        T = max(x.shape[0] for x in wavs)
        min_samples = max(int(sample_rate * 0.5), 4 * hop_length)
        T = max(T, min_samples)
        T = ((T + hop_length - 1) // hop_length) * hop_length
        wav_batch = torch.stack([F.pad(x, (0, T - x.shape[0])) for x in wavs], dim=0)
        # Pad z_e
        T_z = T // hop_length
        z_e_batch = torch.stack([
            F.pad(z, (0, T_z - z.shape[1])) if z.shape[1] < T_z else z[:, :T_z]
            for z in z_es
        ], dim=0)
        return wav_batch.unsqueeze(1), z_e_batch  # [B, 1, T], [B, D, T_z]

    return collate_fn


class HFAudioDataset(torch.utils.data.IterableDataset):
    """Stream audio directly from HuggingFace datasets — no download step needed.

    Uses HF datasets streaming mode to load audio on-the-fly.
    Supports distributed training (rank-based sharding) and infinite iteration.

    Usage:
        dataset = HFAudioDataset("openslr/librispeech_asr", split="train.clean.100",
                                 rank=rank, world_size=world_size)
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train.clean.100",
        sample_rate: int = 24000,
        max_seconds: float = 15.0,
        audio_column: str = "audio",
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.sample_rate = sample_rate
        self.max_samples = int(max_seconds * sample_rate)
        self.audio_column = audio_column
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __iter__(self):
        from datasets import load_dataset

        # Infinite iteration: restart stream when exhausted
        while True:
            ds = load_dataset(
                self.dataset_name, split=self.split,
                streaming=True,
            )
            ds = ds.shuffle(seed=self.seed + self._epoch, buffer_size=5000)

            # Distributed sharding: each rank skips to its slice
            for i, example in enumerate(ds):
                if i % self.world_size != self.rank:
                    continue
                try:
                    wav = self._process_example(example)
                    if wav is not None:
                        yield wav
                except Exception:
                    continue  # skip bad examples

            self._epoch += 1  # next pass gets different shuffle

    def _process_example(self, example):
        audio = example[self.audio_column]
        arr = torch.tensor(audio["array"], dtype=torch.float32)
        sr = audio["sampling_rate"]

        # Resample if needed
        if sr != self.sample_rate:
            import torchaudio
            arr = arr.unsqueeze(0)  # [1, T]
            arr = torchaudio.functional.resample(arr, sr, self.sample_rate)
            arr = arr.squeeze(0)  # [T]

        # Skip very short audio (< 1s)
        if arr.shape[0] < self.sample_rate:
            return None

        # Random crop
        if arr.shape[0] > self.max_samples:
            start = torch.randint(0, arr.shape[0] - self.max_samples, (1,)).item()
            arr = arr[start : start + self.max_samples]

        return arr


# ──────────────────────────────────────────────────────────────
# Optimizer setup
# ──────────────────────────────────────────────────────────────

def split_params_muon_adam(model, exclude_names=None):
    """Split model parameters into Muon-eligible (2D) and AdamW (1D/other)."""
    muon_params = []
    adam_params = []
    exclude_names = exclude_names or set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in exclude_names:
            continue
        if param.dim() >= 2 and param.shape[-1] > 1:
            muon_params.append(param)
        else:
            adam_params.append(param)

    return muon_params, adam_params


def build_optimizer(muon_params, adam_params, muon_lr, adam_lr, adam_betas, weight_decay, use_muon=True):
    """Build optimizer. AdamW by default, Muon+AdamW if use_muon=True and available."""
    if use_muon:
        try:
            from muon import MuonWithAuxAdam

            optimizer = MuonWithAuxAdam(
                muon_params=[{"params": muon_params, "lr": muon_lr}],
                lr=muon_lr,
                momentum=0.95,
                nesterov=True,
                aux_optimizer_cls=torch.optim.AdamW,
                aux_optimizer_kwargs={"weight_decay": weight_decay},
                aux_params=[{"params": adam_params, "lr": adam_lr, "betas": adam_betas}],
            )
            return optimizer, "muon"
        except ImportError:
            print("[WARN] Muon not found, falling back to AdamW for all params")

    optimizer = torch.optim.AdamW(
        [{"params": muon_params + adam_params, "lr": adam_lr, "betas": adam_betas}],
        weight_decay=weight_decay,
    )
    return optimizer, "adamw"


def cosine_warmup_schedule(step, warmup_steps, total_steps, min_ratio=0.1):
    """Cosine decay with linear warmup. Returns multiplier in [min_ratio, 1]."""
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress))


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _check_loss_health(loss_val: float, step: int) -> str:
    """Check if loss value is healthy. Returns 'ok', 'nan', or 'exploded'."""
    if math.isnan(loss_val) or math.isinf(loss_val):
        return "nan"
    if loss_val > 1e6:
        return "exploded"
    return "ok"


def _try_hf_resume(stage: str, hf_repo: Optional[str], device: str = "cpu"):
    """Try to resume from HuggingFace checkpoint. Returns ckpt dict or None."""
    if not hf_repo:
        return None
    try:
        from koe.hf_utils import pull_latest_checkpoint
        ckpt = pull_latest_checkpoint(stage, repo_id=hf_repo, device=device)
        if ckpt:
            print(f"[{stage}] Found HF checkpoint at step {ckpt.get('step', '?')}")
        return ckpt
    except Exception as e:
        print(f"[{stage}] HF resume failed: {e}")
        return None


def _try_local_resume(stage: str, output_dir: str, device: str = "cpu"):
    """Try to resume from latest local checkpoint. Returns ckpt dict or None."""
    import glob as glob_mod
    pattern = os.path.join(output_dir, f"{stage}_step*.pt")
    ckpts = sorted(glob_mod.glob(pattern))
    if not ckpts:
        return None
    latest = ckpts[-1]
    try:
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        print(f"[{stage}] Loaded local checkpoint: {latest} (step {ckpt.get('step', '?')})")
        return ckpt
    except Exception as e:
        print(f"[{stage}] Local resume failed: {e}")
        return None


def _push_to_hf(state_dict, config, step, stage, hf_repo):
    """Push checkpoint to HF in background. Non-blocking, best-effort."""
    if not hf_repo:
        return
    try:
        from koe.hf_utils import push_checkpoint_to_hf
        push_checkpoint_to_hf(state_dict, config, step, stage, repo_id=hf_repo)
        print(f"[{stage}] Pushed step {step} to HF")
    except Exception as e:
        print(f"[{stage}] HF push failed (non-fatal): {e}")


def _log_gpu_memory(step: int, prefix: str):
    """Log GPU memory usage to W&B."""
    try:
        import wandb
        if not wandb.run:
            return
        mem_alloc = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        wandb.log({
            f"{prefix}/gpu_mem_allocated_gb": mem_alloc,
            f"{prefix}/gpu_mem_reserved_gb": mem_reserved,
        }, step=step)
    except (ImportError, Exception):
        pass


# ──────────────────────────────────────────────────────────────
# Eval sample management
# ──────────────────────────────────────────────────────────────

def _prepare_eval_samples(
    dataset,
    n_samples: int = 100,
    sample_rate: int = 24000,
    hop_length: int = 9600,
) -> List[torch.Tensor]:
    """Pre-select eval samples from the dataset.

    Works with both map-style (AudioDataset) and iterable (HFAudioDataset) datasets.
    Returns list of [1, 1, T] waveform tensors (CPU).
    """
    import random
    eval_wavs = []

    if hasattr(dataset, '__len__'):
        # Map-style dataset: random indexing
        indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
        for idx in indices:
            item = dataset[idx]
            wav = item[0] if isinstance(item, tuple) else item  # handle CachedEncoderDataset
            remainder = wav.shape[0] % hop_length
            if remainder:
                wav = F.pad(wav, (0, hop_length - remainder))
            eval_wavs.append(wav.unsqueeze(0).unsqueeze(0))  # [1, 1, T]
    else:
        # Iterable dataset: take first n_samples
        for wav in dataset:
            remainder = wav.shape[0] % hop_length
            if remainder:
                wav = F.pad(wav, (0, hop_length - remainder))
            eval_wavs.append(wav.unsqueeze(0).unsqueeze(0))
            if len(eval_wavs) >= n_samples:
                break

    return eval_wavs


# ──────────────────────────────────────────────────────────────
# Stage 1: JEPA self-supervised encoder training
# ──────────────────────────────────────────────────────────────

def train_stage1(args, cfg: CodecConfig, tcfg: TokenizerTrainConfig):
    """Train JEPA encoder via masked prediction.

    Supports both DeepSpeed distributed and single-GPU modes.
    Single-GPU mode (--single_gpu): no DeepSpeed, no NCCL, no distributed.
    """
    use_ds = not getattr(args, 'single_gpu', False)

    if use_ds:
        from datetime import timedelta
        deepspeed.init_distributed(timeout=timedelta(minutes=30))
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    is_main = rank == 0
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Build model
    encoder = JEPAEncoder(
        sample_rate=cfg.sample_rate,
        code_dim=cfg.code_dim,
        channels=cfg.channels,
        strides=cfg.strides,
        n_res_blocks=cfg.n_res_blocks,
        n_conformer=cfg.n_conformer,
        conformer_heads=cfg.conformer_heads,
        use_gaatn=True,
    )

    if is_main:
        n_params = sum(p.numel() for p in encoder.parameters())
        n_train = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        print(f"[Stage 1] Encoder: {n_params/1e6:.1f}M total, {n_train/1e6:.1f}M trainable")

    # Try to resume: --resume_from takes priority, then HF
    # NOTE: All ranks must load the checkpoint (not just is_main) so DDP starts
    # with consistent weights across GPUs.
    start_step = 0
    resume_optimizer_state = None
    wandb_run_id = None
    if getattr(args, 'resume_from', None):
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        # Support both key names: "state_dict" (HF format) and "encoder" (local format)
        sd = ckpt.get("state_dict", ckpt.get("encoder"))
        if sd:
            encoder.load_state_dict(sd)
            start_step = ckpt.get("step", 0)
            resume_optimizer_state = ckpt.get("optimizer")
            wandb_run_id = ckpt.get("wandb_run_id")
            if is_main:
                print(f"[Stage 1] Resuming from local checkpoint: {args.resume_from} (step {start_step})")
        else:
            if is_main:
                print(f"[Stage 1] WARNING: No state_dict/encoder key in {args.resume_from}")
    elif not getattr(args, 'fresh', False):
        # Try HF first, then local checkpoints
        ckpt = None
        if args.hf_repo:
            ckpt = _try_hf_resume("stage1", args.hf_repo)
            if ckpt and is_main:
                print(f"[Stage 1] Resuming from HF step {ckpt.get('step', 0)}")
        if ckpt is None:
            ckpt = _try_local_resume("stage1", str(Path(args.output_dir)))
            if ckpt and is_main:
                print(f"[Stage 1] Resuming from local checkpoint step {ckpt.get('step', 0)}")
        if ckpt:
            sd = ckpt.get("state_dict", ckpt.get("encoder"))
            if sd:
                encoder.load_state_dict(sd)
                start_step = ckpt.get("step", 0)
                resume_optimizer_state = ckpt.get("optimizer")

    # For single-GPU: cast to bf16 BEFORE optimizer creation so param dtypes are
    # consistent with optimizer state buffers (avoids fused Adam dtype mismatch)
    if not use_ds:
        encoder = encoder.to(device, dtype=torch.bfloat16)

    # Optimizer — AdamW by default, Muon if requested
    use_muon = args.optimizer == "muon"
    muon_p, adam_p = split_params_muon_adam(encoder)
    optimizer, opt_type = build_optimizer(
        muon_p, adam_p,
        muon_lr=tcfg.stage1_muon_lr, adam_lr=tcfg.stage1_adam_lr,
        adam_betas=tcfg.adam_betas, weight_decay=tcfg.weight_decay,
        use_muon=use_muon,
    )
    if is_main:
        print(f"[Stage 1] Optimizer: {opt_type}")

    # Restore optimizer state if resuming from local checkpoint
    if resume_optimizer_state is not None:
        try:
            optimizer.load_state_dict(resume_optimizer_state)
            print(f"[Stage 1] Restored optimizer state from checkpoint")
        except Exception as e:
            print(f"[Stage 1] Could not restore optimizer state (non-fatal): {e}")

    # Optional torch.compile
    if args.compile:
        encoder = torch.compile(encoder)
        if is_main:
            print("[Stage 1] torch.compile enabled")

    # Model setup: DeepSpeed or plain PyTorch
    if use_ds:
        model_engine, optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=encoder,
            optimizer=optimizer,
            model_parameters=encoder.parameters(),
        )
        raw_encoder = model_engine.module
    else:
        model_engine = encoder
        raw_encoder = encoder
        if is_main:
            print("[Stage 1] Single-GPU mode (no DeepSpeed, bf16)")

    # Dataset
    collate = make_collate_fn(cfg.sample_rate, cfg.hop_length)
    per_gpu_bs = args.batch_size or (tcfg.stage1_batch_size // world_size)
    if args.hf_dataset:
        dataset = HFAudioDataset(
            args.hf_dataset, split=args.hf_split,
            sample_rate=cfg.sample_rate, max_seconds=tcfg.max_audio_seconds,
            rank=rank, world_size=world_size,
        )
        loader = DataLoader(
            dataset, batch_size=per_gpu_bs,
            collate_fn=collate, num_workers=0, pin_memory=True,
        )
        sampler = dataset  # duck-type: has set_epoch()
        if is_main:
            print(f"[Stage 1] Streaming from HF: {args.hf_dataset}/{args.hf_split}, "
                  f"{per_gpu_bs}/GPU, {world_size} GPUs")
    else:
        num_workers = args.num_workers if args.num_workers is not None else 16
        dataset = AudioDataset(args.data_dir, cfg.sample_rate, tcfg.max_audio_seconds)
        if world_size > 1:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        else:
            sampler = None  # single-GPU: use shuffle=True in DataLoader
        loader = DataLoader(
            dataset, batch_size=per_gpu_bs, sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=collate, num_workers=num_workers, pin_memory=True,
            drop_last=True, persistent_workers=True, prefetch_factor=4,
        )
        if is_main:
            print(f"[Stage 1] Dataset: {len(dataset)} files, {per_gpu_bs}/GPU, "
                  f"{world_size} GPUs, effective batch {per_gpu_bs * world_size}, "
                  f"workers={num_workers}, prefetch=4")

    # W&B
    if is_main and not args.no_wandb:
        try:
            import wandb
            wandb_kwargs = dict(
                project=tcfg.wandb_project,
                name=f"stage1_{args.run_name or 'v1'}",
                config={"stage": 1, **vars(tcfg), **vars(cfg),
                        "world_size": world_size, "optimizer": opt_type},
                resume="allow",
            )
            if wandb_run_id:
                wandb_kwargs["id"] = wandb_run_id
                print(f"[Stage 1] Resuming W&B run: {wandb_run_id}")
            wandb.init(**wandb_kwargs)
            wandb_run_id = wandb.run.id
        except ImportError:
            pass

    # Prepare eval samples
    eval_wavs = None
    evaluator = None
    if is_main and args.eval_every > 0:
        try:
            from koe.eval_codec import CodecEvaluator
            eval_wavs = _prepare_eval_samples(dataset, n_samples=20, sample_rate=cfg.sample_rate, hop_length=cfg.hop_length)
            evaluator = CodecEvaluator(cfg.sample_rate)
            print(f"[Stage 1] Prepared {len(eval_wavs)} eval samples")
        except Exception as e:
            print(f"[Stage 1] Eval setup failed (non-fatal): {e}")

    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step = start_step
    epoch = 0
    t0 = time.time()
    accum_loss = 0.0
    nan_count = 0
    max_nan_before_rollback = 5
    _sigterm_received = [False]

    def _sigterm_handler(signum, frame):
        if is_main:
            print(f"\n[Stage 1] SIGTERM received at step {step}, saving emergency checkpoint...")
        _sigterm_received[0] = True

    import signal
    signal.signal(signal.SIGTERM, _sigterm_handler)

    while step < tcfg.stage1_steps:
        if sampler is not None and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)

        for batch in loader:
            if step >= tcfg.stage1_steps or _sigterm_received[0]:
                break

            wav = batch.to(device, dtype=torch.bfloat16)  # [B, 1, T]

            # Create JEPA mask
            T_z = jepa_time_len_from_wav(wav.shape[-1], cfg.strides)
            mask = create_jepa_mask(
                batch_size=wav.shape[0], seq_len=T_z,
                mask_ratio=tcfg.stage1_mask_ratio, device=device,
            )

            # Forward (model is already in bf16 for both DS and single-GPU modes)
            z_context, z_pred, mask_out, z_target = model_engine(wav, mask)

            # Loss: MSE at masked positions
            inv_mask = (1.0 - mask_out.unsqueeze(1).float())
            n_masked = inv_mask.sum().clamp(min=1)
            loss = ((z_pred - z_target) ** 2 * inv_mask).sum() / (n_masked * z_pred.shape[1])

            # NaN detection
            loss_val = loss.item()
            health = _check_loss_health(loss_val, step)
            if health != "ok":
                nan_count += 1
                if is_main:
                    print(f"[Stage 1] WARNING: {health} loss at step {step} "
                          f"(count {nan_count}/{max_nan_before_rollback})")
                if nan_count >= max_nan_before_rollback:
                    if is_main:
                        print(f"[Stage 1] Too many NaN/Inf losses. Attempting rollback...")
                        ckpt = _try_hf_resume("stage1", args.hf_repo)
                        if ckpt is None:
                            ckpt = _try_local_resume("stage1", str(output_dir))
                        if ckpt is not None:
                            raw_encoder.load_state_dict(ckpt["state_dict"])
                            step = ckpt.get("step", 0)
                            nan_count = 0
                            print(f"[Stage 1] Rolled back to step {step}")
                        else:
                            print("[Stage 1] No checkpoint available for rollback — stopping training")
                            break
                    continue
                continue

            nan_count = 0  # Reset on healthy loss

            # Backward + step
            if use_ds:
                model_engine.backward(loss)
                model_engine.step()
            else:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(raw_encoder.parameters(), 1.0)
                optimizer.step()

            # EMA update
            raw_encoder.update_target_encoder(decay=tcfg.stage1_ema_decay)

            # LR schedule
            lr_mult = cosine_warmup_schedule(step, 1000, tcfg.stage1_steps)
            for pg in optimizer.param_groups:
                if "initial_lr" not in pg:
                    pg["initial_lr"] = pg["lr"]
                pg["lr"] = pg["initial_lr"] * lr_mult

            step += 1
            accum_loss += loss_val

            # Logging
            if is_main and step % tcfg.log_every == 0:
                elapsed = time.time() - t0
                avg_loss = accum_loss / tcfg.log_every
                accum_loss = 0.0
                steps_per_sec = (step - start_step) / elapsed if elapsed > 0 else 0
                samples_per_sec = steps_per_sec * per_gpu_bs * world_size

                print(f"[Stage 1] step {step}/{tcfg.stage1_steps} | "
                      f"loss={avg_loss:.4f} | lr={lr_mult:.4f} | "
                      f"{steps_per_sec:.1f} steps/s | {samples_per_sec:.0f} samples/s")
                try:
                    import wandb
                    if wandb.run:
                        log_dict = {
                            "stage1/loss": avg_loss,
                            "stage1/lr_mult": lr_mult,
                            "stage1/steps_per_sec": steps_per_sec,
                            "stage1/samples_per_sec": samples_per_sec,
                        }
                        # Representation diagnostics every 100 steps
                        if step % 100 == 0:
                            with torch.no_grad():
                                log_dict.update({
                                    "stage1/z_pred_std": z_pred.float().std().item(),
                                    "stage1/z_target_std": z_target.float().std().item(),
                                    "stage1/z_pred_mean_abs": z_pred.float().abs().mean().item(),
                                    "stage1/pred_target_cosine": F.cosine_similarity(
                                        z_pred.float().flatten(1), z_target.float().flatten(1), dim=1
                                    ).mean().item(),
                                })
                        wandb.log(log_dict, step=step)
                except (ImportError, Exception):
                    pass

                _log_gpu_memory(step, "stage1")

            # Checkpoint
            if is_main and step % args.save_every == 0:
                if use_ds:
                    # Barrier so all ranks finish before rank 0 saves
                    import torch.distributed as dist
                    if dist.is_initialized():
                        dist.barrier()
                    model_engine.save_checkpoint(
                        str(output_dir), tag=f"stage1_step{step}",
                        client_state={"step": step},
                    )
                else:
                    # Plain PyTorch checkpoint
                    ckpt_path = output_dir / f"stage1_step{step}.pt"
                    ckpt_data = {
                        "step": step,
                        "encoder": raw_encoder.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": vars(cfg),
                    }
                    if wandb_run_id:
                        ckpt_data["wandb_run_id"] = wandb_run_id
                    torch.save(ckpt_data, ckpt_path)

                # HuggingFace checkpoint for durability
                _push_to_hf(
                    raw_encoder.state_dict(), vars(cfg),
                    step, "stage1", args.hf_repo,
                )

                # Clean old checkpoints (keep last 3)
                if use_ds:
                    ckpt_dirs = sorted(
                        [d for d in output_dir.iterdir()
                         if d.is_dir() and d.name.startswith("stage1_step")],
                        key=lambda p: p.stat().st_mtime,
                    )
                    if len(ckpt_dirs) > 3:
                        import shutil
                        for old in ckpt_dirs[:-3]:
                            shutil.rmtree(old)
                else:
                    ckpt_files = sorted(
                        output_dir.glob("stage1_step*.pt"),
                        key=lambda p: p.stat().st_mtime,
                    )
                    if len(ckpt_files) > 3:
                        for old in ckpt_files[:-3]:
                            old.unlink()

                print(f"[Stage 1] Saved checkpoint at step {step}")

                if use_ds and dist.is_initialized():
                    dist.barrier()  # all ranks sync after save

            # Eval (async on CPU)
            if is_main and evaluator and eval_wavs and step % args.eval_every == 0:
                # Wait for any pending eval
                evaluator.wait_for_pending(timeout=30)
                print(f"[Stage 1] Note: Encoder-only eval not applicable for Stage 1 (no decoder yet)")

        epoch += 1
        if _sigterm_received[0]:
            break

    # Emergency save on SIGTERM
    if _sigterm_received[0] and is_main:
        ckpt_path = output_dir / f"stage1_step{step}.pt"
        torch.save({
            "step": step,
            "encoder": raw_encoder.state_dict(),
            "optimizer": optimizer.state_dict() if not use_ds else None,
            "config": vars(cfg),
        }, ckpt_path)
        # Also save as latest for easy resume
        latest_path = output_dir / "stage1_latest.pt"
        torch.save({
            "step": step,
            "encoder": raw_encoder.state_dict(),
            "config": vars(cfg),
        }, latest_path)
        print(f"[Stage 1] Emergency checkpoint saved at step {step}")

    # Save final consolidated weights
    if is_main:
        final_path = output_dir / "stage1_final.pt" if not _sigterm_received[0] else output_dir / "stage1_latest.pt"
        final_state = {
            "step": step,
            "encoder": raw_encoder.state_dict(),
            "config": vars(cfg),
        }
        torch.save(final_state, final_path)
        print(f"[Stage 1] {'Training complete' if not _sigterm_received[0] else 'SIGTERM save'}. Final: {final_path}")

        # Push final to HF
        try:
            from koe.hf_utils import push_final_model
            push_final_model(raw_encoder.state_dict(), vars(cfg), "stage1", repo_id=args.hf_repo)
            print(f"[Stage 1] Final model pushed to HF")
        except Exception as e:
            print(f"[Stage 1] HF final push failed: {e}")

    if use_ds:
        deepspeed.comm.destroy_process_group()

    return raw_encoder


# ──────────────────────────────────────────────────────────────
# Stage 2: FSQ-VAE + GAN decoder training
# ──────────────────────────────────────────────────────────────

def train_stage2(args, cfg: CodecConfig, tcfg: TokenizerTrainConfig, encoder: Optional[JEPAEncoder] = None):
    """Train FSQ quantizer + HiFi-GAN decoder.

    GAN training with generator + discriminator.
    Supports both DeepSpeed distributed and single-GPU modes.

    Features:
    - Discriminator warmup (no GAN loss for first N steps)
    - set_requires_grad for clean gen/disc step separation
    - Disc collapse detection (disc_loss < 0.1 for 500 steps)
    - Gen collapse detection (gen_loss > 10 for 500 steps)
    - Async CPU eval: PESQ, STOI, mel distance
    - Audio + spectrogram logging to W&B
    - HuggingFace checkpoint durability
    """
    use_ds = getattr(args, 'deepspeed', False) and not getattr(args, 'single_gpu', False)
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if use_ds:
        from datetime import timedelta
        if not deepspeed.comm.is_initialized():
            deepspeed.init_distributed(timeout=timedelta(minutes=30))
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif world_size > 1:
        # Plain DDP via torchrun (no DeepSpeed)
        import torch.distributed as _dist
        if not _dist.is_initialized():
            _dist.init_process_group(backend="nccl")
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        local_rank = 0

    is_main = rank == 0
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Load Stage 1 encoder if not passed directly
    if encoder is None:
        encoder = JEPAEncoder(
            sample_rate=cfg.sample_rate, code_dim=cfg.code_dim,
            channels=cfg.channels, strides=cfg.strides,
            n_res_blocks=cfg.n_res_blocks, n_conformer=cfg.n_conformer,
            conformer_heads=cfg.conformer_heads,
        )
        ckpt_path = args.stage1_ckpt
        # Support hf:// prefix for HuggingFace checkpoints
        if ckpt_path and ckpt_path.startswith("hf://"):
            repo_and_file = ckpt_path[5:]  # remove "hf://"
            parts = repo_and_file.split("/", 2)
            repo_id = f"{parts[0]}/{parts[1]}"
            filename = parts[2] if len(parts) > 2 else "stage1_final.pt"
            from koe.hf_utils import pull_checkpoint_from_hf
            ckpt = pull_checkpoint_from_hf(filename, repo_id=repo_id)
            if ckpt is None:
                raise RuntimeError(f"Could not download {ckpt_path} from HF")
            # Handle both checkpoint formats
            if "encoder" in ckpt:
                encoder.load_state_dict(ckpt["encoder"])
            elif "state_dict" in ckpt:
                encoder.load_state_dict(ckpt["state_dict"])
        elif ckpt_path:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "encoder" in ckpt:
                encoder.load_state_dict(ckpt["encoder"])
            elif "state_dict" in ckpt:
                encoder.load_state_dict(ckpt["state_dict"])
        else:
            raise ValueError("Must provide --stage1_ckpt for Stage 2")

        if is_main:
            print(f"[Stage 2] Loaded Stage 1 encoder from {ckpt_path}")

    # Build VAE
    finetune_enc = getattr(args, 'finetune_encoder', False)
    if is_main:
        print(f"[Stage 2] Encoder: {'FINE-TUNING' if finetune_enc else 'FROZEN'}")
    vae = WaveformJEPAFSQVAE(
        jepa_encoder=encoder,
        fsq_levels=cfg.fsq_levels, channels=cfg.channels, strides=cfg.strides,
        use_tanh=False, hifi_kernels=cfg.hifi_kernels,
        use_decoder_gaatn=cfg.use_decoder_gaatn, freeze_encoder=not finetune_enc,
        code_dim=cfg.code_dim, sample_rate=cfg.sample_rate,
        n_res_blocks=cfg.n_res_blocks, n_conformer=cfg.n_conformer,
        conformer_heads=cfg.conformer_heads,
    )

    # Discriminators — combined for DeepSpeed
    class DiscriminatorBundle(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mpd = MultiPeriodDiscriminator()
            self.msd = MultiScaleDiscriminator()
        def forward(self, y, y_hat):
            mpd_out = self.mpd(y, y_hat)
            msd_out = self.msd(y, y_hat)
            return mpd_out, msd_out

    disc_bundle = DiscriminatorBundle()
    stft_loss_fn = MRSTFTLoss().to(device)

    # Enable gradient checkpointing on discriminators
    if getattr(args, 'grad_ckpt', False):
        for d in disc_bundle.mpd.ds:
            d.use_grad_ckpt = True
        for d in disc_bundle.msd.ds:
            d.use_grad_ckpt = True
        if is_main:
            print("[Stage 2] Gradient checkpointing enabled on discriminators")

    if is_main:
        gen_params = sum(p.numel() for p in vae.trainable_params())
        disc_params_n = sum(p.numel() for p in disc_bundle.parameters())
        print(f"[Stage 2] Generator trainable: {gen_params/1e6:.1f}M | "
              f"Discriminators: {disc_params_n/1e6:.1f}M")

    # Try to resume from HF (skip if --fresh)
    start_step = 0
    wandb_run_id = None
    if getattr(args, 'resume_from', None):
        resume_ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if "state_dict" in resume_ckpt:
            vae.load_state_dict(resume_ckpt["state_dict"], strict=False)
        if "disc_state_dict" in resume_ckpt:
            disc_bundle.load_state_dict(resume_ckpt["disc_state_dict"], strict=False)
        start_step = resume_ckpt.get("step", 0)
        wandb_run_id = resume_ckpt.get("wandb_run_id")
        if is_main:
            print(f"[Stage 2] Resuming from local checkpoint: {args.resume_from} (step {start_step})")
    elif is_main and args.hf_repo and not getattr(args, 'fresh', False):
        hf_ckpt = _try_hf_resume("stage2", args.hf_repo)
        if hf_ckpt:
            if "state_dict" in hf_ckpt:
                vae.load_state_dict(hf_ckpt["state_dict"], strict=False)
            start_step = hf_ckpt.get("step", 0)
            print(f"[Stage 2] Resuming from HF step {start_step}")
    elif getattr(args, 'fresh', False) and is_main:
        print("[Stage 2] Fresh start — skipping HF/local resume")

    # For single-GPU: cast to bf16 before optimizer creation
    if not use_ds:
        vae = vae.to(device, dtype=torch.bfloat16)
        disc_bundle = disc_bundle.to(device, dtype=torch.bfloat16)

    # Generator optimizer
    gen_trainable = vae.trainable_params()
    trainable_set = set(id(p) for p in gen_trainable)
    use_muon = args.optimizer == "muon"
    enc_lr_scale = getattr(args, 'encoder_lr_scale', 1.0)

    # Track encoder param IDs for gradient scaling (differential LR via grad scaling)
    enc_param_ids = set()
    if finetune_enc and enc_lr_scale != 1.0:
        enc_param_ids = set(id(p) for p in vae.encoder.parameters() if p.requires_grad)
        if is_main:
            enc_lr = tcfg.stage2_adam_lr_gen * enc_lr_scale
            print(f"[Stage 2] Encoder grad scale={enc_lr_scale} "
                  f"(effective LR: {enc_lr:.1e} vs decoder {tcfg.stage2_adam_lr_gen:.1e})")

    muon_p, adam_p = split_params_muon_adam(vae)
    muon_p = [p for p in muon_p if id(p) in trainable_set]
    adam_p = [p for p in adam_p if id(p) in trainable_set]
    gen_optimizer, opt_type = build_optimizer(
        muon_p, adam_p,
        muon_lr=tcfg.stage2_muon_lr, adam_lr=tcfg.stage2_adam_lr_gen,
        adam_betas=tcfg.adam_betas, weight_decay=tcfg.weight_decay,
        use_muon=use_muon,
    )

    # Discriminator optimizer
    if getattr(args, 'optim_8bit', False):
        import bitsandbytes as bnb
        disc_optimizer = bnb.optim.AdamW8bit(
            disc_bundle.parameters(), lr=tcfg.stage2_adam_lr_disc,
            betas=tcfg.adam_betas, weight_decay=tcfg.weight_decay,
        )
        disc_opt_type = "AdamW8bit"
    else:
        disc_optimizer = torch.optim.AdamW(
            disc_bundle.parameters(), lr=tcfg.stage2_adam_lr_disc,
            betas=tcfg.adam_betas, weight_decay=tcfg.weight_decay,
        )
        disc_opt_type = "AdamW"

    if is_main:
        print(f"[Stage 2] Gen optimizer: {opt_type} | Disc optimizer: {disc_opt_type}")

    # Model setup: DeepSpeed, DDP, or single-GPU
    use_ddp_plain = (world_size > 1 and not use_ds)  # torchrun without --deepspeed
    if use_ds:
        gen_engine, gen_optimizer, _, _ = deepspeed.initialize(
            args=args, model=vae, optimizer=gen_optimizer,
            model_parameters=[p for p in vae.parameters() if p.requires_grad],
        )
        disc_engine, disc_optimizer, _, _ = deepspeed.initialize(
            args=args, model=disc_bundle, optimizer=disc_optimizer,
            model_parameters=disc_bundle.parameters(),
        )
        raw_vae = gen_engine.module
        raw_disc = disc_engine.module
    elif use_ddp_plain:
        from torch.nn.parallel import DistributedDataParallel as DDP
        vae = vae.to(device, dtype=torch.bfloat16)
        disc_bundle = disc_bundle.to(device, dtype=torch.bfloat16)
        gen_engine = DDP(vae, device_ids=[local_rank], find_unused_parameters=True)
        disc_engine = DDP(disc_bundle, device_ids=[local_rank])
        raw_vae = gen_engine.module
        raw_disc = disc_engine.module
        if is_main:
            print(f"[Stage 2] Plain DDP mode ({world_size} GPUs, bf16)")
    else:
        gen_engine = vae
        disc_engine = disc_bundle
        raw_vae = vae
        raw_disc = disc_bundle
        if is_main:
            print("[Stage 2] Single-GPU mode (no DeepSpeed, bf16)")

    # Register gradient scaling hooks for encoder differential LR
    # Scales encoder gradients by enc_lr_scale so same optimizer LR = lower effective LR
    _enc_grad_hooks = []
    if finetune_enc and enc_lr_scale != 1.0:
        _scale = enc_lr_scale  # capture for closure
        for p in raw_vae.encoder.parameters():
            if p.requires_grad:
                hook = p.register_hook(lambda grad, s=_scale: grad * s)
                _enc_grad_hooks.append(hook)
        if is_main:
            print(f"[Stage 2] Registered {len(_enc_grad_hooks)} encoder grad scale hooks (×{enc_lr_scale})")

    # torch.compile for Stage 2 (decoder + discriminator)
    if args.compile and not use_ds:
        if is_main:
            print("[Stage 2] torch.compile enabled for decoder + discriminator")
        gen_engine = torch.compile(gen_engine)
        disc_engine = torch.compile(disc_engine)

    # Dataset
    per_gpu_bs = args.batch_size or (tcfg.stage2_batch_size // world_size)
    use_cached = getattr(args, 'cached_z_dir', None) is not None
    if use_cached:
        cached_collate = _make_cached_collate_fn(cfg.sample_rate, cfg.hop_length)
        num_workers = args.num_workers if args.num_workers is not None else 16
        dataset = CachedEncoderDataset(
            args.data_dir, args.cached_z_dir,
            sample_rate=cfg.sample_rate, max_seconds=tcfg.max_audio_seconds,
            hop_length=cfg.hop_length,
        )
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
        loader = DataLoader(
            dataset, batch_size=max(per_gpu_bs, 1), sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=cached_collate, num_workers=num_workers, pin_memory=True,
            drop_last=True, persistent_workers=True, prefetch_factor=4,
        )
        if is_main:
            print(f"[Stage 2] CACHED mode: {len(dataset)} files, {per_gpu_bs}/GPU, "
                  f"encoder forward SKIPPED (~40% faster)")
    elif args.hf_dataset:
        collate = make_collate_fn(cfg.sample_rate, cfg.hop_length)
        dataset = HFAudioDataset(
            args.hf_dataset, split=args.hf_split,
            sample_rate=cfg.sample_rate, max_seconds=tcfg.max_audio_seconds,
            rank=rank, world_size=world_size,
        )
        loader = DataLoader(
            dataset, batch_size=max(per_gpu_bs, 1),
            collate_fn=collate, num_workers=0, pin_memory=True,
        )
        sampler = dataset
        if is_main:
            print(f"[Stage 2] Streaming from HF: {args.hf_dataset}/{args.hf_split}, "
                  f"{per_gpu_bs}/GPU, {world_size} GPUs")
    else:
        collate = make_collate_fn(cfg.sample_rate, cfg.hop_length)
        num_workers = args.num_workers if args.num_workers is not None else 16
        dataset = AudioDataset(args.data_dir, cfg.sample_rate, tcfg.max_audio_seconds)
        if world_size > 1:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        else:
            sampler = None
        loader = DataLoader(
            dataset, batch_size=max(per_gpu_bs, 1), sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=collate, num_workers=num_workers, pin_memory=True,
            drop_last=True, persistent_workers=True, prefetch_factor=4,
        )
        if is_main:
            print(f"[Stage 2] Dataset: {len(dataset)} files, {per_gpu_bs}/GPU, "
                  f"{world_size} GPUs, effective batch {per_gpu_bs * world_size}, "
                  f"workers={num_workers}, prefetch=4")

    # W&B
    if is_main and not args.no_wandb:
        try:
            import wandb
            if not wandb.run:
                wandb_kwargs = dict(
                    project=tcfg.wandb_project,
                    name=f"stage2_{args.run_name or 'v1'}",
                    config={"stage": 2, **vars(tcfg), **vars(cfg),
                            "world_size": world_size, "optimizer": opt_type},
                    resume="allow",
                )
                if wandb_run_id:
                    wandb_kwargs["id"] = wandb_run_id
                    print(f"[Stage 2] Resuming W&B run: {wandb_run_id}")
                wandb.init(**wandb_kwargs)
                wandb_run_id = wandb.run.id
        except ImportError:
            pass

    # Eval setup
    eval_wavs = None
    evaluator = None
    if is_main and args.eval_every > 0:
        try:
            from koe.eval_codec import CodecEvaluator
            eval_wavs = _prepare_eval_samples(dataset, n_samples=50, sample_rate=cfg.sample_rate, hop_length=cfg.hop_length)
            evaluator = CodecEvaluator(cfg.sample_rate)
            print(f"[Stage 2] Prepared {len(eval_wavs)} eval samples")
        except Exception as e:
            print(f"[Stage 2] Eval setup failed (non-fatal): {e}")

    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step = start_step
    epoch = 0
    t0 = time.time()
    nan_count = 0

    # Collapse detection tracking
    disc_loss_history = []
    gen_loss_history = []
    collapse_window = 500

    # Dynamic GAN hyperparameters (for collapse recovery)
    lambda_gan = tcfg.stage2_lambda_gan
    disc_warmup = tcfg.stage2_disc_warmup

    while step < tcfg.stage2_steps:
        if sampler is not None and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)

        for batch in loader:
            if step >= tcfg.stage2_steps:
                break

            if use_cached:
                wav, z_e_cached = batch
                wav = wav.to(device, dtype=torch.bfloat16)
                z_e_cached = z_e_cached.to(device, dtype=torch.bfloat16)
                # Skip encoder forward — use cached z_e
                wav_rec, indices, aux_loss, z_e = raw_vae.forward_from_z_e(
                    z_e_cached, wav.shape[-1]
                )
            else:
                wav = batch.to(device, dtype=torch.bfloat16)
                # ── Forward: VAE encode + decode ──
                wav_rec, indices, aux_loss, z_e = gen_engine(wav)

            # ── Generator step (GAN forward runs every step, matching reference) ──
            stft_loss = stft_loss_fn(wav_rec, wav)
            l1_loss = F.l1_loss(wav_rec, wav)

            # Disc forward for generator: always compute GAN loss (reference does this from step 0)
            set_requires_grad(raw_disc.mpd, False)
            set_requires_grad(raw_disc.msd, False)
            (mpd_r2, mpd_f2, mpd_fr, mpd_fg), (msd_r2, msd_f2, msd_fr, msd_fg) = disc_engine(wav, wav_rec)
            g_adv_loss = generator_loss(mpd_f2) + generator_loss(msd_f2)
            g_feat_loss = feature_loss(mpd_fr, mpd_fg) + feature_loss(msd_fr, msd_fg)

            effective_lambda_gan = lambda_gan

            g_loss = (
                tcfg.stage2_lambda_stft * stft_loss + l1_loss
                + effective_lambda_gan * (g_adv_loss + g_feat_loss) + aux_loss
            )

            # NaN detection (d_loss_val captured above before del)
            g_loss_val = g_loss.item()
            health = _check_loss_health(g_loss_val, step)
            if health != "ok":
                nan_count += 1
                if is_main:
                    print(f"[Stage 2] WARNING: {health} gen loss at step {step} (count {nan_count})")
                if nan_count >= 5:
                    if is_main:
                        print("[Stage 2] Too many NaN losses. Attempting rollback...")
                        ckpt = _try_hf_resume("stage2", args.hf_repo)
                        if ckpt is None:
                            ckpt = _try_local_resume("stage2", str(output_dir))
                        if ckpt is not None and "state_dict" in ckpt:
                            raw_vae.load_state_dict(ckpt["state_dict"], strict=False)
                            step = ckpt.get("step", 0)
                            nan_count = 0
                            print(f"[Stage 2] Rolled back to step {step}")
                        else:
                            print("[Stage 2] No checkpoint available for rollback — stopping training")
                            break
                    continue
                continue

            nan_count = 0

            if use_ds:
                gen_engine.backward(g_loss)
                gen_engine.step()
            else:
                gen_optimizer.zero_grad()
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(gen_trainable, 1.0)
                gen_optimizer.step()

            # ── Discriminator step (only update after warmup, matching reference) ──
            if step >= disc_warmup:
                set_requires_grad(raw_disc.mpd, True)
                set_requires_grad(raw_disc.msd, True)

                with torch.no_grad():
                    rec_det = wav_rec.detach()
                (mpd_real, mpd_fake, _, _), (msd_real, msd_fake, _, _) = disc_engine(wav, rec_det)
                mpd_d_loss = discriminator_loss(mpd_real, mpd_fake)
                msd_d_loss = discriminator_loss(msd_real, msd_fake)
                d_loss = mpd_d_loss + msd_d_loss

                if use_ds:
                    disc_engine.backward(d_loss)
                    disc_engine.step()
                else:
                    disc_optimizer.zero_grad()
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(disc_bundle.parameters(), 1.0)
                    disc_optimizer.step()

                d_loss_val = d_loss.item()
                del mpd_real, mpd_fake, msd_real, msd_fake, mpd_d_loss, msd_d_loss, d_loss

                set_requires_grad(raw_disc.mpd, False)
                set_requires_grad(raw_disc.msd, False)
            else:
                d_loss_val = 0.0

            step += 1

            # Collapse detection
            disc_loss_history.append(d_loss_val)
            gen_loss_history.append(g_loss_val)
            if len(disc_loss_history) > collapse_window:
                disc_loss_history.pop(0)
                gen_loss_history.pop(0)

            if is_main and len(disc_loss_history) == collapse_window:
                avg_d = sum(disc_loss_history) / collapse_window
                avg_g = sum(gen_loss_history) / collapse_window

                # Discriminator collapse: too strong or too weak (only check after warmup)
                if avg_d < 0.1 and step > disc_warmup + collapse_window:
                    print(f"[Stage 2] WARNING: Discriminator collapse detected "
                          f"(avg_d={avg_d:.3f} < 0.1). Increasing disc_lr 2x.")
                    for pg in disc_optimizer.param_groups:
                        pg["lr"] *= 2.0
                    disc_loss_history.clear()

                # Generator collapse
                if avg_g > 10.0 and step > disc_warmup:
                    print(f"[Stage 2] WARNING: Generator instability "
                          f"(avg_g={avg_g:.3f} > 10). Reducing λ_gan by 50%.")
                    lambda_gan *= 0.5
                    gen_loss_history.clear()

            # LR schedule
            lr_mult = cosine_warmup_schedule(step, 1000, tcfg.stage2_steps)
            for pg in gen_optimizer.param_groups:
                if "initial_lr" not in pg:
                    pg["initial_lr"] = pg["lr"]
                pg["lr"] = pg["initial_lr"] * lr_mult

            # Logging
            if is_main and step % tcfg.log_every == 0:
                elapsed = time.time() - t0
                real_steps = step - start_step
                steps_per_sec = real_steps / elapsed if elapsed > 0 else 0
                print(f"[Stage 2] step {step}/{tcfg.stage2_steps} | "
                      f"g={g_loss_val:.4f} d={d_loss_val:.4f} | "
                      f"stft={stft_loss.item():.4f} adv={g_adv_loss.item():.4f} "
                      f"feat={g_feat_loss.item():.4f} | "
                      f"λ_gan={effective_lambda_gan:.3f} | "
                      f"{steps_per_sec:.1f} steps/s")
                try:
                    import wandb
                    if wandb.run:
                        wandb.log({
                            "stage2/g_loss": g_loss_val,
                            "stage2/d_loss": d_loss_val,
                            "stage2/stft_loss": stft_loss.item(),
                            "stage2/l1_loss": l1_loss.item(),
                            "stage2/adv_loss": g_adv_loss.item(),
                            "stage2/feat_loss": g_feat_loss.item(),
                            "stage2/aux_loss": aux_loss.item(),
                            "stage2/fsq_entropy": raw_vae.fsq.entropy_metric(indices.detach()) if indices is not None else 0.0,
                            "stage2/fsq_pre_scale": raw_vae.fsq.pre_scale.item(),
                            "stage2/lambda_gan": lambda_gan,
                            "stage2/lr_mult": lr_mult,
                            "stage2/steps_per_sec": steps_per_sec,
                        }, step=step)
                except (ImportError, Exception):
                    pass

                _log_gpu_memory(step, "stage2")

            # Checkpoint
            if is_main and step % args.save_every == 0:
                if use_ds:
                    gen_engine.save_checkpoint(
                        str(output_dir), tag=f"stage2_gen_step{step}",
                        client_state={"step": step},
                    )
                    disc_engine.save_checkpoint(
                        str(output_dir), tag=f"stage2_disc_step{step}",
                        client_state={"step": step},
                    )
                else:
                    ckpt_path = output_dir / f"stage2_step{step}.pt"
                    ckpt_data = {
                        "step": step,
                        "state_dict": raw_vae.state_dict(),
                        "disc_state_dict": raw_disc.state_dict(),
                        "gen_optimizer": gen_optimizer.state_dict(),
                        "disc_optimizer": disc_optimizer.state_dict(),
                        "config": vars(cfg),
                    }
                    if wandb_run_id:
                        ckpt_data["wandb_run_id"] = wandb_run_id
                    torch.save(ckpt_data, ckpt_path)

                # HF push
                _push_to_hf(
                    raw_vae.state_dict(), vars(cfg),
                    step, "stage2", args.hf_repo,
                )

                # Clean old checkpoints (keep last 3)
                if use_ds:
                    for prefix in ("stage2_gen_step", "stage2_disc_step"):
                        ckpt_dirs = sorted(
                            [d for d in output_dir.iterdir()
                             if d.is_dir() and d.name.startswith(prefix)],
                            key=lambda p: p.stat().st_mtime,
                        )
                        if len(ckpt_dirs) > 3:
                            import shutil
                            for old in ckpt_dirs[:-3]:
                                shutil.rmtree(old)
                else:
                    ckpt_files = sorted(
                        output_dir.glob("stage2_step*.pt"),
                        key=lambda p: p.stat().st_mtime,
                    )
                    if len(ckpt_files) > 3:
                        for old in ckpt_files[:-3]:
                            old.unlink()

                print(f"[Stage 2] Saved checkpoints at step {step}")
                # Flush cache after checkpoint save (state_dict copies fragment memory)
                gc.collect()
                torch.cuda.empty_cache()

            # Barrier so all ranks wait for checkpoint save
            if world_size > 1 and step % args.save_every == 0:
                import torch.distributed as _dist
                if _dist.is_initialized():
                    _dist.barrier()

            # Eval (on CPU to avoid GPU memory spike)
            if is_main and evaluator and eval_wavs and step % args.eval_every == 0:
                # Wait for previous eval
                prev = evaluator.wait_for_pending(timeout=30)
                if prev:
                    print(f"[Stage 2] Eval results: {prev}")

                # Build a temporary KoeCodec for eval (on CPU to avoid GPU memory spike)
                include_medium = (step % (args.eval_every * 5) == 0)
                include_heavy = (step % (args.eval_every * 10) == 0)
                try:
                    from koe.codec import KoeCodec
                    eval_codec = KoeCodec(cfg)
                    eval_codec.model.load_state_dict(
                        {k: v.cpu() for k, v in raw_vae.state_dict().items()}
                    )
                    eval_codec.eval().cpu()
                    metrics = evaluator.evaluate_and_log(
                        eval_codec, eval_wavs[:10], step,
                        include_medium=include_medium,
                        include_heavy=include_heavy,
                        wandb_prefix="stage2_eval",
                    )
                    print(f"[Stage 2] Eval at step {step}: {metrics}")
                except Exception as e:
                    print(f"[Stage 2] Eval failed (non-fatal): {e}")
                finally:
                    # Explicitly free eval model and flush GPU cache
                    if 'eval_codec' in locals():
                        del eval_codec
                    gc.collect()
                    torch.cuda.empty_cache()

            # FSQ token analytics (every token_viz_every steps)
            if is_main and step % tcfg.token_viz_every == 0 and step > 0:
                try:
                    from koe.eval_codec import log_fsq_analytics_to_wandb
                    from koe.codec_impl import fsq_pack_indices
                    # Get token indices from last batch
                    tokens_packed = fsq_pack_indices(
                        indices.detach(), cfg.fsq_levels, cfg.group_size,
                    )
                    log_fsq_analytics_to_wandb(
                        tokens_packed.detach().cpu().numpy(), step,
                        num_groups=cfg.num_groups,
                        num_tokens=cfg.max_packed_value + 1,
                        prefix="stage2_fsq",
                    )
                except Exception as e:
                    print(f"[Stage 2] FSQ analytics failed (non-fatal): {e}")

                # Per-dimension codebook visualization
                try:
                    from koe.eval_codec import log_fsq_codebook_to_wandb
                    log_fsq_codebook_to_wandb(
                        indices.detach().cpu().numpy(),
                        z_e.detach().cpu().numpy() if z_e is not None else None,
                        step,
                        num_levels=cfg.fsq_levels[0],
                        group_size=cfg.group_size,
                        prefix="stage2_fsq",
                    )
                except Exception as e:
                    print(f"[Stage 2] Codebook viz failed (non-fatal): {e}")

        epoch += 1

    # Save final consolidated weights
    if is_main:
        final_path = output_dir / "tokenizer_final.pt"
        torch.save({
            "step": step,
            "vae": raw_vae.state_dict(),
            "config": vars(cfg),
        }, final_path)
        print(f"[Stage 2] Training complete. Final: {final_path}")

        # Push final to HF
        try:
            from koe.hf_utils import push_final_model
            push_final_model(raw_vae.state_dict(), vars(cfg), "stage2", repo_id=args.hf_repo)
            print(f"[Stage 2] Final model pushed to HF")
        except Exception as e:
            print(f"[Stage 2] HF final push failed: {e}")

        # Final eval
        if evaluator and eval_wavs:
            try:
                from koe.codec import KoeCodec
                eval_codec = KoeCodec(cfg)
                eval_codec.model.load_state_dict(raw_vae.state_dict())
                eval_codec.eval()
                final_metrics = evaluator.evaluate_and_log(
                    eval_codec, eval_wavs, step,
                    include_medium=True,
                    include_heavy=True,
                    wandb_prefix="stage2_final",
                )
                print(f"[Stage 2] Final eval: {final_metrics}")

                # Update HF model card with final metrics
                from koe.hf_utils import update_model_card
                update_model_card(repo_id=args.hf_repo, eval_metrics=final_metrics, step=step)
            except Exception as e:
                print(f"[Stage 2] Final eval failed: {e}")

    if use_ds:
        deepspeed.comm.destroy_process_group()


# ──────────────────────────────────────────────────────────────
# Encoder output caching (skip frozen encoder in Stage 2)
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def cache_encoder_outputs(args, cfg: CodecConfig):
    """Pre-compute encoder outputs for all training data.

    The encoder is frozen in Stage 2, so its outputs are deterministic.
    Caching them skips ~40% of per-step GPU compute.

    Saves: {cache_dir}/{idx:06d}.pt — each file is z_e [D, T_z] in bf16.
    Total cache size ~230MB for 28K files (LibriSpeech clean-100).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build encoder
    encoder = JEPAEncoder(
        sample_rate=cfg.sample_rate, code_dim=cfg.code_dim,
        channels=cfg.channels, strides=cfg.strides,
        n_res_blocks=cfg.n_res_blocks, n_conformer=cfg.n_conformer,
        conformer_heads=cfg.conformer_heads, use_gaatn=True,
    )

    # Load Stage 1 weights
    ckpt_path = args.stage1_ckpt
    if ckpt_path.startswith("hf://"):
        parts = ckpt_path[5:].split("/", 2)
        repo_id = f"{parts[0]}/{parts[1]}"
        filename = parts[2] if len(parts) > 2 else "stage1_final.pt"
        from koe.hf_utils import pull_checkpoint_from_hf
        ckpt = pull_checkpoint_from_hf(filename, repo_id=repo_id)
        if ckpt is None:
            raise RuntimeError(f"Could not download {ckpt_path} from HF")
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"])
    elif "state_dict" in ckpt:
        encoder.load_state_dict(ckpt["state_dict"])

    encoder = encoder.to(device, dtype=torch.bfloat16).eval()

    # Load dataset — use batch_size=1 for variable-length files, but process fast
    dataset = AudioDataset(args.data_dir, cfg.sample_rate, max_seconds=300.0)  # full length
    collate = make_collate_fn(cfg.sample_rate, cfg.hop_length)
    cache_batch_size = 32  # process 32 files at once (padded to max length in batch)
    loader = DataLoader(
        dataset, batch_size=cache_batch_size, collate_fn=collate,
        num_workers=16, pin_memory=True, prefetch_factor=4,
    )

    cache_dir = Path(args.cached_z_dir or f"{args.data_dir}_z_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Cache] Encoding {len(dataset)} files (batch={cache_batch_size}) → {cache_dir}")
    t0 = time.time()
    file_idx = 0
    for batch in loader:
        wav = batch.to(device, dtype=torch.bfloat16)
        z_e = encoder.encode(wav)  # [B, D, T_z]
        # Save each file individually
        for i in range(z_e.shape[0]):
            torch.save(z_e[i].cpu(), cache_dir / f"{file_idx:06d}.pt")
            file_idx += 1
        if file_idx % 5000 == 0:
            elapsed = time.time() - t0
            rate = file_idx / elapsed
            eta = (len(dataset) - file_idx) / rate
            print(f"[Cache] {file_idx}/{len(dataset)} files ({elapsed:.0f}s, {rate:.0f} files/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"[Cache] Done: {file_idx} files in {elapsed:.0f}s → {cache_dir}")
    return str(cache_dir)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train KoeTTS JEPA tokenizer")
    parser.add_argument("--stage", type=str, required=True, choices=["1", "2", "both"],
                        help="Which training stage to run")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing audio files (recursively searched)")
    parser.add_argument("--hf_dataset", type=str, default=None,
                        help="HuggingFace dataset to stream (e.g. openslr/librispeech_asr)")
    parser.add_argument("--hf_split", type=str, default="train.clean.100",
                        help="HF dataset split (default: train.clean.100)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for checkpoints")
    parser.add_argument("--stage1_ckpt", type=str, default=None,
                        help="Stage 1 checkpoint (local path or hf://repo/file)")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon"],
                        help="Optimizer: adamw (default, proven stable) or muon")
    parser.add_argument("--run_name", type=str, default=None,
                        help="W&B run name suffix")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")
    parser.add_argument("--hf_repo", type=str, default=None,
                        help="HuggingFace repo for checkpoint storage (e.g. Andy004/koe-tokenizer)")
    parser.add_argument("--eval_every", type=int, default=5000,
                        help="Run eval every N steps (0 to disable)")
    parser.add_argument("--save_every", type=int, default=2000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override per-GPU batch size (default: from config / world_size)")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for model forward pass")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader num_workers (default: 16 for local, 0 for streaming)")
    parser.add_argument("--single_gpu", action="store_true",
                        help="Single-GPU mode: no DeepSpeed, no distributed, no NCCL")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to a local checkpoint .pt file to resume from")
    parser.add_argument("--fresh", action="store_true",
                        help="Start fresh — skip auto-resume from HF/local checkpoints")
    parser.add_argument("--cache_encoder", action="store_true",
                        help="Pre-compute encoder outputs for all data, then exit")
    parser.add_argument("--cached_z_dir", type=str, default=None,
                        help="Directory of pre-computed encoder outputs (skips encoder forward)")
    parser.add_argument("--n_res_blocks", type=int, default=None,
                        help="Override encoder n_res_blocks (default: from CodecConfig)")
    parser.add_argument("--fsq_levels", type=str, default=None,
                        help="Override FSQ levels as comma-separated ints, e.g. '8,8,8,8'")
    parser.add_argument("--strides", type=str, default=None,
                        help="Override encoder strides as comma-separated ints, e.g. '4,4,4,5,6'")
    parser.add_argument("--stage1_steps", type=int, default=None,
                        help="Override Stage 1 total steps (default: 24000)")
    parser.add_argument("--stage2_steps", type=int, default=None,
                        help="Override Stage 2 total steps (default: 29000)")
    parser.add_argument("--max_seconds", type=float, default=None,
                        help="Override max audio clip length in seconds (default: 15.0)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override base learning rate for AdamW (default: 1.5e-4 stage1, 1.5e-4 stage2 gen)")
    parser.add_argument("--grad_ckpt", action="store_true",
                        help="Enable gradient checkpointing on discriminators (saves ~30%% VRAM)")
    parser.add_argument("--optim_8bit", action="store_true",
                        help="Use 8-bit AdamW (requires bitsandbytes)")
    parser.add_argument("--finetune_encoder", action="store_true",
                        help="Fine-tune encoder during Stage 2 (default: frozen)")
    parser.add_argument("--encoder_lr_scale", type=float, default=1.0,
                        help="LR multiplier for encoder params when fine-tuning (e.g. 0.1 = 10x lower)")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Set by torchrun/deepspeed launcher (use LOCAL_RANK env var)")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    if not args.data_dir and not args.hf_dataset:
        parser.error("Either --data_dir or --hf_dataset is required")

    cfg = CodecConfig()
    tcfg = TokenizerTrainConfig()

    # CLI overrides for architecture
    if args.n_res_blocks is not None:
        cfg.n_res_blocks = args.n_res_blocks
    if args.fsq_levels is not None:
        cfg.fsq_levels = [int(x) for x in args.fsq_levels.split(",")]
    if args.strides is not None:
        cfg.strides = [int(x) for x in args.strides.split(",")]
    if args.stage1_steps is not None:
        tcfg.stage1_steps = args.stage1_steps
    if args.stage2_steps is not None:
        tcfg.stage2_steps = args.stage2_steps
    if args.max_seconds is not None:
        tcfg.max_audio_seconds = args.max_seconds
    if args.lr is not None:
        tcfg.stage1_adam_lr = args.lr
        tcfg.stage2_adam_lr_gen = args.lr

    # Performance tuning
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    # Encoder caching mode: pre-compute z_e for all data, then exit
    if getattr(args, 'cache_encoder', False):
        if not args.stage1_ckpt:
            parser.error("--cache_encoder requires --stage1_ckpt")
        if not args.data_dir:
            parser.error("--cache_encoder requires --data_dir")
        cache_encoder_outputs(args, cfg)
        return

    if args.stage in ("1", "both"):
        encoder = train_stage1(args, cfg, tcfg)

    if args.stage in ("2", "both"):
        if args.stage == "both":
            train_stage2(args, cfg, tcfg, encoder=encoder)
        else:
            train_stage2(args, cfg, tcfg)


if __name__ == "__main__":
    main()
