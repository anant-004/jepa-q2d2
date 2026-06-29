"""Stage 2 decoder training — single GPU, optimized for H100.

Trains the full codec (encoder + FSQ/Q2D2 + HiFi-GAN decoder) with:
- Encoder fine-tuning at 0.1× decoder LR (the key breakthrough)
- Multi-resolution STFT + L1 + GAN losses
- Optional WavLM perceptual loss and phase-aware STFT
- Optional Q2D2 quantizer (drop-in FSQ replacement)
- Comprehensive W&B logging with audio samples
- GCS checkpoint backup for spot instances
- SIGTERM handling for graceful preemption

Usage:
    python -m koe.fast.train_stage2 --data_dir /data/librilight \
        --output_dir ./checkpoints --stage1_ckpt /data/stage1_final.pt

    # With WavLM perceptual loss:
    python -m koe.fast.train_stage2 --data_dir /data/librilight \
        --output_dir ./checkpoints --stage1_ckpt /data/stage1_final.pt \
        --lambda_perceptual 0.1

    # With Q2D2 quantizer:
    python -m koe.fast.train_stage2 --data_dir /data/librilight \
        --output_dir ./checkpoints --stage1_ckpt /data/stage1_final.pt \
        --quantizer q2d2
"""

import argparse
import math
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from koe.codec_impl import (
    WaveformJEPAFSQVAE,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    MultiScaleSTFTDiscriminator,
    MRSTFTLoss,
    discriminator_loss,
    generator_loss,
    feature_loss,
    set_requires_grad,
)
from koe.config import CodecConfig


# ═══════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════
SAMPLE_RATE = 24000


# ═══════════════════════════════════════════════════════════
# LR Schedule
# ═══════════════════════════════════════════════════════════

def cosine_warmup(step: int, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    progress = min(1.0, (step - warmup) / max(total - warmup, 1))
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress))


# ═══════════════════════════════════════════════════════════
# Audio Dataset
# ═══════════════════════════════════════════════════════════

class AudioDataset(torch.utils.data.Dataset):
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
        import soundfile as sf
        import torchaudio
        # Retry with random fallback on corrupt files
        for attempt in range(3):
            try:
                cur = idx if attempt == 0 else torch.randint(0, len(self.files), (1,)).item()
                path = str(self.files[cur])
                info = sf.info(path)
                sr = info.samplerate
                total = info.frames
                need = int(self.max_samples * sr / self.sample_rate) if sr != self.sample_rate else self.max_samples
                if total > need:
                    start = torch.randint(0, total - need, (1,)).item()
                    data, sr = sf.read(path, start=start, stop=start + need, dtype='float32')
                else:
                    data, sr = sf.read(path, dtype='float32')
                if data.ndim == 1:
                    wav = torch.from_numpy(data).unsqueeze(0)  # [1, T]
                else:
                    wav = torch.from_numpy(data.T)  # [C, T]
                if sr != self.sample_rate:
                    wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
                if wav.shape[0] > 1:
                    wav = wav.mean(0, keepdim=True)
                wav = wav.squeeze(0)
                if wav.shape[0] > self.max_samples:
                    start = torch.randint(0, wav.shape[0] - self.max_samples, (1,)).item()
                    wav = wav[start:start + self.max_samples]
                return wav
            except Exception:
                continue
        # Last resort: return silence
        return torch.zeros(self.max_samples)


def make_collate(sample_rate: int, hop_length: int):
    min_samples = max(int(sample_rate * 0.5), 4 * hop_length)

    def collate_fn(batch):
        if not batch:
            return None
        lengths = [x.shape[0] for x in batch]
        T = max(max(lengths), min_samples)
        T = ((T + hop_length - 1) // hop_length) * hop_length
        out = torch.zeros(len(batch), 1, T)
        for i, x in enumerate(batch):
            out[i, 0, :x.shape[0]] = x
        return out

    return collate_fn


# ═══════════════════════════════════════════════════════════
# Eval helpers
# ═══════════════════════════════════════════════════════════

def make_eval_samples(data_dir: str, hop_length: int, n: int = 10, seed: int = 42):
    import random
    rng = random.Random(seed)
    ds = AudioDataset(data_dir, SAMPLE_RATE, 10.0)
    idxs = rng.sample(range(len(ds)), min(n, len(ds)))
    wavs = []
    for i in idxs:
        wav = ds[i]
        rem = wav.shape[0] % hop_length
        if rem:
            wav = F.pad(wav, (0, hop_length - rem))
        wavs.append(wav.unsqueeze(0).unsqueeze(0))
    return wavs


# ═══════════════════════════════════════════════════════════
# Attention Pooling for Teacher Distillation
# ═══════════════════════════════════════════════════════════

class AttentionPool(nn.Module):
    """Learned attention pooling: N teacher frames → 1 student frame.

    Used for distillation from a higher frame-rate teacher (e.g. 12.5 Hz v9)
    to a lower frame-rate student (e.g. 2.5 Hz). Each window of pool_size
    teacher frames is summarized into 1 frame via learned attention weights.
    """

    def __init__(self, dim: int = 128):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.proj = nn.Linear(dim, dim)

    def forward(self, z: torch.Tensor, pool_size: int = 5) -> torch.Tensor:
        B, D, T = z.shape
        T_out = T // pool_size
        z = z[:, :, :T_out * pool_size]
        # [B, D, T_out, pool_size] → [B, T_out, pool_size, D]
        z = z.reshape(B, D, T_out, pool_size).permute(0, 2, 3, 1)
        attn = (z @ self.query.transpose(-1, -2)).softmax(dim=2)  # [B, T_out, pool_size, 1]
        pooled = (z * attn).sum(dim=2)  # [B, T_out, D]
        return self.proj(pooled).permute(0, 2, 1)  # [B, D, T_out]


# ═══════════════════════════════════════════════════════════
# Main training function
# ═══════════════════════════════════════════════════════════

def train(args):
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    # GPU optimizations
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # ── Config ──
    strides = [int(s) for s in args.strides.split(",")]
    fsq_levels = [int(l) for l in args.fsq_levels.split(",")]
    cfg = CodecConfig(
        strides=strides,
        fsq_levels=fsq_levels,
        n_res_blocks=args.n_res_blocks,
        n_conformer=args.n_conformer,
        conformer_heads=args.conformer_heads,
    )
    hop = cfg.hop_length
    print(f"[train] Config: strides={strides}, hop={hop}, "
          f"frame_rate={cfg.frame_rate:.1f} Hz, fsq_levels={fsq_levels}")

    # ── Build model ──
    model = WaveformJEPAFSQVAE(
        sample_rate=SAMPLE_RATE,
        code_dim=128,
        channels=cfg.channels,
        strides=strides,
        n_res_blocks=args.n_res_blocks,
        n_conformer=args.n_conformer,
        conformer_heads=args.conformer_heads,
        fsq_levels=fsq_levels,
        hifi_kernels=cfg.hifi_kernels,
        use_decoder_gaatn=cfg.use_decoder_gaatn,
    )

    # ── Load Stage 1 encoder checkpoint ──
    if args.stage1_ckpt and Path(args.stage1_ckpt).exists():
        ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
        enc_sd = ckpt.get("state_dict", ckpt.get("encoder", {}))
        if enc_sd:
            try:
                model.encoder.load_state_dict(enc_sd, strict=True)
                print(f"[train] Loaded Stage 1 encoder from {args.stage1_ckpt}")
            except RuntimeError:
                model.encoder.load_state_dict(enc_sd, strict=False)
                print(f"[train] Loaded Stage 1 encoder (partial) from {args.stage1_ckpt}")

    # ── Q2D2 quantizer swap ──
    if args.quantizer == "q2d2":
        from koe.fast.q2d2 import Q2D2Quantizer
        model.fsq = Q2D2Quantizer(
            dim=128,
            num_levels=args.q2d2_levels,
            grid_type=args.q2d2_grid,
        )
        print(f"[train] Swapped FSQ → Q2D2 (levels={args.q2d2_levels}, grid={args.q2d2_grid})")

    # ── Teacher distillation setup ──
    teacher_encoder = None
    attention_pool = None
    distill_pool_size = 5
    if args.teacher_ckpt and Path(args.teacher_ckpt).exists():
        teacher_strides = [int(s) for s in args.teacher_strides.split(",")]
        teacher_hop = 1
        for s in teacher_strides:
            teacher_hop *= s
        distill_pool_size = hop // teacher_hop  # 9600/1920=5 for v9→2.5Hz

        teacher_model = WaveformJEPAFSQVAE(
            sample_rate=SAMPLE_RATE, code_dim=128,
            channels=cfg.channels, strides=teacher_strides,
            n_res_blocks=args.n_res_blocks, n_conformer=args.n_conformer,
            conformer_heads=args.conformer_heads,
            fsq_levels=[8, 8, 8, 8], hifi_kernels=cfg.hifi_kernels,
        )
        t_ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
        t_sd = t_ckpt.get("state_dict", {})
        enc_sd = {k.replace("encoder.", ""): v for k, v in t_sd.items() if k.startswith("encoder.")}
        teacher_model.encoder.load_state_dict(enc_sd, strict=False)
        teacher_encoder = teacher_model.encoder
        teacher_encoder.eval()
        teacher_encoder.requires_grad_(False)
        del teacher_model, t_ckpt, t_sd, enc_sd

        if args.distill_pool == "learned":
            attention_pool = AttentionPool(dim=128)
        print(f"[train] Teacher distillation: strides={teacher_strides}, "
              f"pool_size={distill_pool_size}, λ={args.lambda_distill}, "
              f"loss={args.distill_loss}, pool={args.distill_pool}, "
              f"start={args.distill_start_step}, ramp={args.distill_anneal_steps}")

    # ── Build discriminators ──
    mpd = MultiPeriodDiscriminator()
    msd = MultiScaleDiscriminator()
    msstftd = None
    if args.use_msstftd:
        msstftd = MultiScaleSTFTDiscriminator()
        print(f"[train] MS-STFT discriminator enabled ({sum(p.numel() for p in msstftd.parameters())/1e3:.0f}K params)")

    # ── Resume from Stage 2 checkpoint ──
    start_step = 0
    wandb_run_id = args.wandb_id
    best_pesq = 0.0

    resume_path = _find_resume_checkpoint(args)
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"], strict=False)
            print(f"[train] Loaded model from {resume_path}")
        if "disc_state_dict" in ckpt:
            disc_sd = ckpt["disc_state_dict"]
            mpd_sd = {k.replace("mpd.", ""): v for k, v in disc_sd.items() if k.startswith("mpd.")}
            msd_sd = {k.replace("msd.", ""): v for k, v in disc_sd.items() if k.startswith("msd.")}
            if mpd_sd:
                mpd.load_state_dict(mpd_sd, strict=False)
            if msd_sd:
                msd.load_state_dict(msd_sd, strict=False)
            if msstftd is not None:
                stftd_sd = {k.replace("msstftd.", ""): v for k, v in disc_sd.items() if k.startswith("msstftd.")}
                if stftd_sd:
                    msstftd.load_state_dict(stftd_sd, strict=False)
            print("[train] Loaded discriminator states")
        start_step = ckpt.get("step", 0)
        wandb_run_id = ckpt.get("wandb_run_id", wandb_run_id)
        best_pesq = ckpt.get("metrics", {}).get("pesq", 0.0)
        print(f"[train] Resuming from step {start_step}")

    # ── Move to GPU ──
    model = model.to(device, dtype=torch.bfloat16)
    mpd = mpd.to(device, dtype=torch.bfloat16)
    msd = msd.to(device, dtype=torch.bfloat16)
    if msstftd is not None:
        msstftd = msstftd.to(device, dtype=torch.bfloat16)
    if teacher_encoder is not None:
        teacher_encoder = teacher_encoder.to(device, dtype=torch.bfloat16)
    if attention_pool is not None:
        attention_pool = attention_pool.to(device, dtype=torch.bfloat16)
        # Load attention_pool state from checkpoint if resuming
        if resume_path and "attention_pool" in ckpt:
            attention_pool.load_state_dict(ckpt["attention_pool"])
            print("[train] Loaded attention_pool from checkpoint")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[train] Model: {n_params:.1f}M params")

    # ── Encoder fine-tuning with gradient scaling ──
    enc_hooks = []
    if args.encoder_lr_scale != 1.0:
        for p in model.encoder.parameters():
            if p.requires_grad:
                enc_hooks.append(
                    p.register_hook(lambda g, s=args.encoder_lr_scale: g * s)
                )
        print(f"[train] Encoder LR scale: {args.encoder_lr_scale}×")

    # ── Optimizers ──
    gen_params = [p for p in model.parameters() if p.requires_grad]
    if attention_pool is not None:
        gen_params += [p for p in attention_pool.parameters() if p.requires_grad]
    gen_opt = torch.optim.AdamW(
        gen_params, lr=args.lr_gen, betas=(0.8, 0.99), weight_decay=1e-3, fused=True,
    )
    disc_params = list(mpd.parameters()) + list(msd.parameters())
    if msstftd is not None:
        disc_params += list(msstftd.parameters())
    disc_opt = torch.optim.AdamW(
        disc_params, lr=args.lr_disc, betas=(0.8, 0.99), weight_decay=1e-3, fused=True,
    )

    # Restore optimizer states
    if resume_path and "gen_optimizer" in ckpt:
        try:
            gen_opt.load_state_dict(ckpt["gen_optimizer"])
            print("[train] Restored generator optimizer")
        except Exception:
            pass
    if resume_path and "disc_optimizer" in ckpt:
        try:
            disc_opt.load_state_dict(ckpt["disc_optimizer"])
            print("[train] Restored discriminator optimizer")
        except Exception:
            pass

    # ── Loss functions ──
    stft_loss_fn = MRSTFTLoss().to(device)

    # Optional WavLM perceptual loss
    perceptual_loss_fn = None
    if args.lambda_perceptual > 0:
        from koe.fast.losses import WavLMPerceptualLoss
        perceptual_loss_fn = WavLMPerceptualLoss(device=str(device))
        print(f"[train] WavLM perceptual loss enabled (λ={args.lambda_perceptual})")

    # Optional phase-aware STFT loss
    phase_loss_fn = None
    if args.lambda_phase > 0:
        from koe.fast.losses import PhaseAwareSTFTLoss
        phase_loss_fn = PhaseAwareSTFTLoss().to(device)
        print(f"[train] Phase-aware STFT loss enabled (λ={args.lambda_phase})")

    # ── Data ──
    dataset = AudioDataset(args.data_dir, SAMPLE_RATE, args.max_seconds)
    collate = make_collate(SAMPLE_RATE, hop)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=4,
    )
    print(f"[train] Dataset: {len(dataset)} files, batch={args.batch_size}")

    # ── Eval setup ──
    eval_wavs = make_eval_samples(args.data_dir, hop, n=args.eval_samples)
    evaluator = None
    try:
        from koe.fast.eval_metrics import EnhancedCodecEvaluator
        evaluator = EnhancedCodecEvaluator(
            sample_rate=SAMPLE_RATE,
            eval_every=args.eval_every,
            num_samples=args.eval_samples,
            log_audio=True,
        )
        print(f"[train] Enhanced evaluator ready ({len(eval_wavs)} samples)")
    except ImportError as e:
        print(f"[train] Enhanced evaluator unavailable ({e}), using basic eval")

    # ── GCS checkpoint manager ──
    gcs_manager = None
    if args.gcs_bucket:
        from koe.fast.gcs_checkpoint import GCSCheckpointManager
        gcs_manager = GCSCheckpointManager(
            bucket=args.gcs_bucket,
            prefix=args.gcs_prefix or args.run_name or "default",
            local_dir=args.output_dir,
            keep_last_n=3,
        )
        print(f"[train] GCS checkpoint: gs://{args.gcs_bucket}/{args.gcs_prefix or args.run_name or 'default'}")

    # ── Output dir ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── W&B ──
    if not args.no_wandb:
        try:
            import wandb
            wandb_kwargs = dict(
                project="koe-tokenizer",
                name=args.run_name or "stage2_decoder",
                config=vars(args),
                resume="allow",
            )
            if wandb_run_id:
                wandb_kwargs["id"] = wandb_run_id
            wandb.init(**wandb_kwargs)
            wandb_run_id = wandb.run.id
            print(f"[train] W&B run: {wandb.run.url}")
        except ImportError:
            print("[train] W&B not available")

    # ── SIGTERM handler ──
    sigterm_received = [False]

    def _on_sigterm(signum, frame):
        print(f"\n[train] SIGTERM received at step {step}. Saving checkpoint...")
        sigterm_received[0] = True

    signal.signal(signal.SIGTERM, _on_sigterm)

    # ── EMA ──
    ema_shadow = None
    if args.ema_decay > 0:
        ema_shadow = {}
        for n, p in model.named_parameters():
            ema_shadow[n] = p.data.float().clone()
        if resume_path and "ema_state" in ckpt:
            for k, v in ckpt["ema_state"].items():
                if k in ema_shadow:
                    ema_shadow[k].copy_(v.float().to(ema_shadow[k].device))
            print("[train] Restored EMA state")
        print(f"[train] EMA enabled (decay={args.ema_decay}, tracking {len(ema_shadow)} keys, float32)")

    # ── Collapse detection state ──
    collapse_window = 500
    disc_loss_history = deque(maxlen=collapse_window)
    gen_loss_history = deque(maxlen=collapse_window)
    lambda_gan = args.lambda_gan

    # ── Training loop ──
    step = start_step
    t0 = time.time()
    t_last_log = t0
    accum = {
        "g_loss": 0.0, "d_loss": 0.0, "stft": 0.0, "l1": 0.0, "distill": 0.0,
        "adv": 0.0, "feat": 0.0, "aux": 0.0, "perceptual": 0.0, "phase": 0.0,
    }
    nan_count = 0

    print(f"[train] Starting from step {step}, target {args.total_steps} steps")

    while step < args.total_steps:
        for batch in loader:
            if step >= args.total_steps or sigterm_received[0]:
                break

            wav = batch.to(device, dtype=torch.bfloat16, non_blocking=True)

            # ── Generator forward ──
            wav_rec, indices, aux_loss, z_e = model(wav)
            stft = stft_loss_fn(wav_rec, wav)
            l1 = F.l1_loss(wav_rec, wav)

            # GAN losses (disc frozen for gen step)
            set_requires_grad(mpd, False)
            set_requires_grad(msd, False)
            mpd_out = mpd(wav, wav_rec)
            msd_out = msd(wav, wav_rec)
            g_adv = generator_loss(mpd_out[1]) + generator_loss(msd_out[1])
            g_feat = feature_loss(mpd_out[2], mpd_out[3]) + feature_loss(msd_out[2], msd_out[3])
            if msstftd is not None:
                set_requires_grad(msstftd, False)
                stftd_out = msstftd(wav, wav_rec)
                g_adv = g_adv + args.msstftd_weight * generator_loss(stftd_out[1])
                g_feat = g_feat + args.msstftd_weight * feature_loss(stftd_out[2], stftd_out[3])

            # Total generator loss
            g_loss = (
                args.lambda_stft * stft + l1
                + lambda_gan * (g_adv + g_feat)
                + aux_loss
            )

            # Optional perceptual loss
            perceptual_val = 0.0
            if perceptual_loss_fn is not None:
                p_loss = perceptual_loss_fn(wav_rec, wav)
                g_loss = g_loss + args.lambda_perceptual * p_loss
                perceptual_val = p_loss.item()

            # Optional phase loss
            phase_val = 0.0
            if phase_loss_fn is not None:
                ph_loss = phase_loss_fn(wav_rec, wav)
                g_loss = g_loss + args.lambda_phase * ph_loss
                phase_val = ph_loss.item()

            # Optional teacher distillation loss
            distill_val = 0.0
            if teacher_encoder is not None and step >= args.distill_start_step:
                with torch.no_grad():
                    teacher_z = teacher_encoder.encode(wav)
                    # Avg pool or learned attention pool
                    if attention_pool is not None:
                        teacher_pooled = attention_pool(teacher_z, pool_size=distill_pool_size)
                    else:
                        teacher_pooled = F.avg_pool1d(teacher_z.float(), distill_pool_size, distill_pool_size).to(teacher_z.dtype)
                min_t = min(z_e.shape[-1], teacher_pooled.shape[-1])
                if args.distill_loss == "cosine":
                    dl = 1.0 - F.cosine_similarity(
                        z_e[:, :, :min_t], teacher_pooled[:, :, :min_t].detach(), dim=1
                    ).mean()
                else:
                    dl = F.l1_loss(z_e[:, :, :min_t], teacher_pooled[:, :, :min_t].detach())
                # Ramp up from 0 to lambda_distill over distill_anneal_steps after start
                ramp_step = step - args.distill_start_step
                ramp = min(1.0, ramp_step / max(args.distill_anneal_steps, 1))
                g_loss = g_loss + args.lambda_distill * ramp * dl
                distill_val = dl.item()

            # NaN check
            g_loss_val = g_loss.item()
            if math.isnan(g_loss_val) or math.isinf(g_loss_val) or g_loss_val > 1e6:
                nan_count += 1
                print(f"[train] Bad gen loss {g_loss_val} at step {step} (count {nan_count})")
                if nan_count >= 5:
                    print("[train] Too many NaN losses, stopping")
                    sigterm_received[0] = True
                    break
                continue
            nan_count = 0

            gen_opt.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen_params, 1.0)
            gen_opt.step()

            # ── EMA update (float32 to avoid bfloat16 precision loss) ──
            if ema_shadow is not None:
                decay = args.ema_decay
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        ema_shadow[n].lerp_(p.data.float(), 1 - decay)

            # ── Discriminator step ──
            if step >= args.disc_warmup:
                set_requires_grad(mpd, True)
                set_requires_grad(msd, True)
                rec_det = wav_rec.detach()
                mpd_out_d = mpd(wav, rec_det)
                msd_out_d = msd(wav, rec_det)
                d_loss = (
                    discriminator_loss(mpd_out_d[0], mpd_out_d[1])
                    + discriminator_loss(msd_out_d[0], msd_out_d[1])
                )
                if msstftd is not None:
                    set_requires_grad(msstftd, True)
                    stftd_out_d = msstftd(wav, rec_det)
                    d_loss = d_loss + discriminator_loss(stftd_out_d[0], stftd_out_d[1])
                disc_opt.zero_grad()
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(disc_params, 1.0)
                disc_opt.step()
                set_requires_grad(mpd, False)
                set_requires_grad(msd, False)
                if msstftd is not None:
                    set_requires_grad(msstftd, False)
                d_loss_val = d_loss.item()
            else:
                d_loss_val = 0.0

            step += 1

            # Accumulate for logging
            accum["g_loss"] += g_loss_val
            accum["d_loss"] += d_loss_val
            accum["stft"] += stft.item()
            accum["l1"] += l1.item()
            accum["adv"] += g_adv.item()
            accum["feat"] += g_feat.item()
            accum["aux"] += aux_loss.item()
            accum["perceptual"] += perceptual_val
            accum["distill"] += distill_val
            accum["phase"] += phase_val

            # ── Collapse detection ──
            disc_loss_history.append(d_loss_val)
            gen_loss_history.append(g_loss_val)

            if len(disc_loss_history) == collapse_window and step > args.disc_warmup + collapse_window:
                avg_d = sum(disc_loss_history) / collapse_window
                avg_g = sum(gen_loss_history) / collapse_window

                if avg_d < 0.1:
                    print(f"[train] Disc collapse (avg_d={avg_d:.3f}). Increasing disc_lr 2×.")
                    for pg in disc_opt.param_groups:
                        pg["lr"] *= 2.0
                    disc_loss_history.clear()

                if avg_g > 10.0:
                    print(f"[train] Gen instability (avg_g={avg_g:.3f}). Reducing λ_gan 50%.")
                    lambda_gan *= 0.5
                    gen_loss_history.clear()

            # ── LR schedule ──
            schedule_total = args.lr_schedule_total if args.lr_schedule_total else args.total_steps
            lr_mult = cosine_warmup(step, args.warmup_steps, schedule_total)
            for pg in gen_opt.param_groups:
                if "initial_lr" not in pg:
                    pg["initial_lr"] = pg["lr"]
                pg["lr"] = pg["initial_lr"] * lr_mult

            # ── Logging ──
            if step % args.log_every == 0:
                t_now = time.time()
                dt = max(t_now - t_last_log, 1e-6)
                n = args.log_every
                sps = n / dt

                log_dict = {
                    "train/g_loss": accum["g_loss"] / n,
                    "train/d_loss": accum["d_loss"] / n,
                    "train/stft_loss": accum["stft"] / n,
                    "train/l1_loss": accum["l1"] / n,
                    "train/adv_loss": accum["adv"] / n,
                    "train/feat_loss": accum["feat"] / n,
                    "train/aux_loss": accum["aux"] / n,
                    "train/lambda_gan": lambda_gan,
                    "train/lr_mult": lr_mult,
                    "train/lr_gen": gen_opt.param_groups[0]["lr"],
                    "train/steps_per_sec": sps,
                    "train/gpu_mem_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                    "train/gpu_mem_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                }
                if perceptual_loss_fn is not None:
                    log_dict["train/perceptual_loss"] = accum["perceptual"] / n
                if phase_loss_fn is not None:
                    log_dict["train/phase_loss"] = accum["phase"] / n
                if teacher_encoder is not None:
                    log_dict["train/distill_loss"] = accum["distill"] / n
                    ramp_s = max(0, step - args.distill_start_step)
                    log_dict["train/distill_ramp"] = min(1.0, ramp_s / max(args.distill_anneal_steps, 1)) * args.lambda_distill

                print(
                    f"step {step}/{args.total_steps} | "
                    f"g={accum['g_loss']/n:.4f} d={accum['d_loss']/n:.4f} "
                    f"stft={accum['stft']/n:.4f} l1={accum['l1']/n:.4f} | "
                    f"{sps:.2f} sps"
                )

                try:
                    import wandb
                    if wandb.run:
                        wandb.log(log_dict, step=step)
                except (ImportError, Exception):
                    pass

                for k in accum:
                    accum[k] = 0.0
                t_last_log = t_now

            # ── Evaluation ──
            if step % args.eval_every == 0:
                if ema_shadow is not None:
                    ema_backup = {}
                    for n, p in model.named_parameters():
                        ema_backup[n] = p.data.clone()
                        p.data.copy_(ema_shadow[n])
                metrics = _run_eval(model, eval_wavs, device, step, evaluator)
                if ema_shadow is not None:
                    for n, p in model.named_parameters():
                        p.data.copy_(ema_backup[n])
                    del ema_backup
                if metrics and metrics.get("pesq", 0) > best_pesq:
                    best_pesq = metrics["pesq"]
                    print(f"[train] New best PESQ: {best_pesq:.4f}")
                torch.cuda.empty_cache()

            # ── Checkpoint ──
            if step % args.save_every == 0:
                _save_checkpoint(
                    model, mpd, msd, gen_opt, disc_opt,
                    step, wandb_run_id, best_pesq, args, cfg,
                    output_dir, gcs_manager, blocking=False,
                    attention_pool=attention_pool,
                    msstftd=msstftd, ema_shadow=ema_shadow,
                )

        if sigterm_received[0] or step >= args.total_steps:
            break

    # ── Cleanup ──
    for h in enc_hooks:
        h.remove()

    # Final save (blocking for SIGTERM)
    _save_checkpoint(
        model, mpd, msd, gen_opt, disc_opt,
        step, wandb_run_id, best_pesq, args, cfg,
        output_dir, gcs_manager, blocking=True,
        attention_pool=attention_pool,
        msstftd=msstftd, ema_shadow=ema_shadow,
    )

    peak = torch.cuda.max_memory_allocated() / 1e9
    elapsed = time.time() - t0
    print(f"\n[train] Done. step={step}, elapsed={elapsed:.0f}s, peak_vram={peak:.1f}GB")


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

def _find_resume_checkpoint(args) -> Optional[str]:
    """Find checkpoint to resume from: explicit path → local latest → GCS latest."""
    if args.resume_from and Path(args.resume_from).exists():
        return args.resume_from

    # Check local latest
    output_dir = Path(args.output_dir)
    latest = output_dir / "stage2_latest.pt"
    if latest.exists():
        return str(latest)

    # Try GCS
    if args.gcs_bucket:
        try:
            from koe.fast.gcs_checkpoint import GCSCheckpointManager
            mgr = GCSCheckpointManager(
                bucket=args.gcs_bucket,
                prefix=args.gcs_prefix or args.run_name or "default",
                local_dir=args.output_dir,
            )
            path = mgr.download_latest()
            if path:
                return path
        except Exception as e:
            print(f"[train] GCS resume failed: {e}")

    return None


def _run_eval(model, eval_wavs, device, step, evaluator) -> Optional[Dict[str, float]]:
    """Run evaluation and log results."""
    model.eval()

    if evaluator is not None:
        try:
            metrics = evaluator.evaluate(model, eval_wavs, step, device)
            model.train()
            return metrics
        except Exception as e:
            print(f"[train] Enhanced eval failed: {e}")

    # Fallback: basic PESQ/STOI eval
    import numpy as np
    from koe.eval_codec import compute_pesq, compute_stoi, mel_spectral_distance

    stois, pesqs, mels = [], [], []

    for wav_t in eval_wavs:
        wav_in = wav_t.to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            rec, indices, _, z_e = model(wav_in)
        orig = wav_t[0, 0].numpy()
        rec_np = rec[0, 0].float().cpu().numpy()
        n = min(len(orig), len(rec_np))
        orig, rec_np = orig[:n], rec_np[:n]
        stois.append(compute_stoi(orig, rec_np, SAMPLE_RATE))
        pesqs.append(compute_pesq(orig, rec_np, SAMPLE_RATE))
        mels.append(mel_spectral_distance(orig, rec_np, SAMPLE_RATE))

    pesqs_clean = [p for p in pesqs if not math.isnan(p)]
    metrics = {
        "stoi": float(np.mean(stois)),
        "pesq": float(np.mean(pesqs_clean)) if pesqs_clean else 0.0,
        "mel_distance": float(np.mean(mels)),
    }

    print(f"[eval] step {step} | PESQ={metrics['pesq']:.4f} "
          f"STOI={metrics['stoi']:.4f} mel={metrics['mel_distance']:.4f}")

    try:
        import wandb
        if wandb.run:
            wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)
    except (ImportError, Exception):
        pass

    model.train()
    return metrics


def _save_checkpoint(
    model, mpd, msd, gen_opt, disc_opt,
    step, wandb_run_id, best_pesq, args, cfg,
    output_dir, gcs_manager, blocking=False,
    attention_pool=None, msstftd=None, ema_shadow=None,
):
    """Save checkpoint locally and optionally upload to GCS."""
    disc_sd = {}
    for k, v in mpd.state_dict().items():
        disc_sd[f"mpd.{k}"] = v
    for k, v in msd.state_dict().items():
        disc_sd[f"msd.{k}"] = v
    if msstftd is not None:
        for k, v in msstftd.state_dict().items():
            disc_sd[f"msstftd.{k}"] = v

    ckpt = {
        "step": step,
        "state_dict": model.state_dict(),
        "disc_state_dict": disc_sd,
        "gen_optimizer": gen_opt.state_dict(),
        "disc_optimizer": disc_opt.state_dict(),
        "config": {
            "strides": [int(s) for s in args.strides.split(",")],
            "fsq_levels": [int(l) for l in args.fsq_levels.split(",")],
            "n_res_blocks": args.n_res_blocks,
            "quantizer": args.quantizer,
        },
        "wandb_run_id": wandb_run_id,
        "metrics": {"pesq": best_pesq},
    }
    if attention_pool is not None:
        ckpt["attention_pool"] = attention_pool.state_dict()
    if ema_shadow is not None:
        ckpt["ema_state"] = {n: v.cpu() for n, v in ema_shadow.items()}

    # Save locally
    for fname in [f"stage2_step{step:07d}.pt", "stage2_latest.pt"]:
        torch.save(ckpt, output_dir / fname)

    # Cleanup old local checkpoints
    step_ckpts = sorted(output_dir.glob("stage2_step*.pt"))
    for old in step_ckpts[:-3]:
        old.unlink()

    print(f"[train] Saved checkpoint at step {step}")

    # Upload to GCS
    if gcs_manager is not None:
        gcs_manager.upload_checkpoint(
            str(output_dir / "stage2_latest.pt"),
            step=step,
            blocking=blocking,
        )

    try:
        import wandb
        if wandb.run:
            wandb.log({
                "checkpoint/step": step,
                "checkpoint/best_pesq": best_pesq,
            }, step=step)
    except (ImportError, Exception):
        pass


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Stage 2 decoder training")

    # Required
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--stage1_ckpt", type=str, default=None,
                        help="Stage 1 encoder checkpoint (for fresh start)")

    # Resume
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Explicit Stage 2 checkpoint to resume from")

    # Model architecture
    parser.add_argument("--strides", type=str, default="8,8,5,5,6")
    parser.add_argument("--fsq_levels", type=str, default="4,4,4,4")
    parser.add_argument("--n_res_blocks", type=int, default=8)
    parser.add_argument("--n_conformer", type=int, default=8)
    parser.add_argument("--conformer_heads", type=int, default=16)

    # Quantizer
    parser.add_argument("--quantizer", type=str, default="fsq",
                        choices=["fsq", "q2d2"])
    parser.add_argument("--q2d2_levels", type=int, default=4)
    parser.add_argument("--q2d2_grid", type=str, default="rhombic",
                        choices=["square", "rhombic"])

    # Training
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr_gen", type=float, default=1.5e-4)
    parser.add_argument("--lr_disc", type=float, default=7.5e-5)
    parser.add_argument("--encoder_lr_scale", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--lr_schedule_total", type=int, default=None,
                        help="Override total_steps for LR schedule. Use original value when extending runs.")
    parser.add_argument("--max_seconds", type=float, default=15.0)
    parser.add_argument("--num_workers", type=int, default=4)

    # Loss weights
    parser.add_argument("--lambda_stft", type=float, default=2.0)
    parser.add_argument("--lambda_gan", type=float, default=0.1)
    parser.add_argument("--lambda_perceptual", type=float, default=0.0,
                        help="WavLM perceptual loss weight (0=disabled)")
    parser.add_argument("--lambda_phase", type=float, default=0.0,
                        help="Phase-aware STFT loss weight (0=disabled)")

    # GAN schedule
    parser.add_argument("--disc_warmup", type=int, default=5000,
                        help="Steps before disc updates (gen GAN loss from step 0)")
    parser.add_argument("--use_msstftd", action="store_true",
                        help="Enable multi-scale STFT discriminator")
    parser.add_argument("--msstftd_weight", type=float, default=0.1,
                        help="Weight for MS-STFT-D loss contribution (default 0.1)")
    parser.add_argument("--ema_decay", type=float, default=0.0,
                        help="EMA decay for model averaging (0=disabled, 0.999=typical)")

    # Eval
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--eval_samples", type=int, default=10)

    # Checkpointing
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=10)

    # Teacher distillation
    parser.add_argument("--teacher_ckpt", type=str, default=None,
                        help="v9 stage2 checkpoint for encoder distillation")
    parser.add_argument("--teacher_strides", type=str, default="4,4,4,5,6",
                        help="Teacher encoder strides (v9=4,4,4,5,6)")
    parser.add_argument("--lambda_distill", type=float, default=0.1,
                        help="Distillation loss weight")
    parser.add_argument("--distill_start_step", type=int, default=10000,
                        help="Step to begin distillation (let model warmup first)")
    parser.add_argument("--distill_anneal_steps", type=int, default=20000,
                        help="Steps to ramp distill weight from 0 to lambda_distill after start")
    parser.add_argument("--distill_loss", type=str, default="cosine",
                        choices=["cosine", "l1"],
                        help="Distillation loss type")
    parser.add_argument("--distill_pool", type=str, default="avg",
                        choices=["avg", "learned"],
                        help="Pooling method for teacher features (avg or learned attention)")

    # GCS
    parser.add_argument("--gcs_bucket", type=str, default=None)
    parser.add_argument("--gcs_prefix", type=str, default=None)

    # W&B
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    main()
