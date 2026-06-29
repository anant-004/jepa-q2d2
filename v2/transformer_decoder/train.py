"""Transformer decoder training — torchrun compatible.

Trains: JEPA encoder (frozen/FT) → proj_down → Q2D2 → TransformerDecoder → waveform

Two-phase training (MOSS-inspired):
  Phase 1 (non-adversarial): STFT + L1 + mel losses only, no discriminator
  Phase 2 (adversarial): Add GAN losses (MPD + MSD)

Usage (single GPU):
    CUDA_VISIBLE_DEVICES=3 python v2/transformer_decoder/train.py --data_dir /data ...

Usage (multi-GPU with torchrun):
    CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=2 v2/transformer_decoder/train.py --data_dir /data ...
"""

import argparse
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from koe.codec_impl import (
    JEPAEncoder,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    MRSTFTLoss,
    discriminator_loss,
    generator_loss,
    feature_loss,
    set_requires_grad,
)
from v2.transformer_decoder.model import TransformerCodecDecoder, TransformerDecoderConfig


SAMPLE_RATE = 24000


# ═══════════════════════════════════════════════════════════
# V2 Transformer Codec
# ═══════════════════════════════════════════════════════════

class V2TransformerCodec(nn.Module):
    """encoder → proj_down → Q2D2 → TransformerDecoder → waveform."""

    def __init__(self, encoder: JEPAEncoder, code_dim: int = 32,
                 decoder_cfg: TransformerDecoderConfig = None):
        super().__init__()
        self.encoder = encoder
        self.enc_dim = encoder.code_dim  # 128
        self.code_dim = code_dim
        self.hop_length = encoder.hop_length

        self.proj_down = nn.Conv1d(self.enc_dim, code_dim, 1)
        self.quantizer = None  # Set externally
        self.decoder = TransformerCodecDecoder(decoder_cfg)

    def forward(self, wav: torch.Tensor):
        original_length = wav.shape[-1]
        z_e = self.encoder.encode(wav)        # [B, 128, T]
        z_proj = self.proj_down(z_e)          # [B, code_dim, T]
        z_q, indices, aux_loss = self.quantizer(z_proj)
        rec = self.decoder(z_q, target_len=original_length)  # [B, 1, T_wav]
        return rec, indices, aux_loss, z_proj


# ═══════════════════════════════════════════════════════════
# Multi-Scale Mel Loss (MOSS-style)
# ═══════════════════════════════════════════════════════════

class MultiScaleMelLoss(nn.Module):
    """Multi-scale mel-spectrogram L1 loss."""

    def __init__(self, sample_rate: int = 24000,
                 window_sizes=(32, 64, 128, 256, 512, 1024, 2048),
                 n_mels_list=(5, 10, 20, 40, 80, 160, 320)):
        super().__init__()
        import torchaudio
        self.transforms = nn.ModuleList()
        for win, n_mels in zip(window_sizes, n_mels_list):
            hop = win // 4
            self.transforms.append(
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate, n_fft=win, hop_length=hop,
                    win_length=win, n_mels=n_mels, power=1.0,
                )
            )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target: [B, 1, T] waveforms."""
        pred_1d = pred.squeeze(1) if pred.dim() == 3 else pred
        tgt_1d = target.squeeze(1) if target.dim() == 3 else target
        loss = torch.tensor(0.0, device=pred.device, dtype=torch.float32)
        for mel_fn in self.transforms:
            mel_fn = mel_fn.to(pred.device)
            p = mel_fn(pred_1d.float())
            t = mel_fn(tgt_1d.float())
            loss = loss + F.l1_loss(p, t)
        return loss / len(self.transforms)


# ═══════════════════════════════════════════════════════════
# Audio Dataset (same as train_v2_stage2)
# ═══════════════════════════════════════════════════════════

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, sample_rate: int = 24000, max_seconds: float = 10.0):
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
                wav = torch.from_numpy(data).unsqueeze(0) if data.ndim == 1 else torch.from_numpy(data.T)
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
        return torch.zeros(self.max_samples)


def make_collate(hop_length: int):
    min_samples = max(int(SAMPLE_RATE * 0.5), 4 * hop_length)

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
# LR Schedule
# ═══════════════════════════════════════════════════════════

def cosine_warmup(step: int, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress))


# ═══════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════

def train(args):
    # ── DDP setup ──
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
    else:
        rank, local_rank, world_size = 0, 0, 1

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    is_main = rank == 0

    # ── Build encoder ──
    strides = [int(s) for s in args.strides.split(",")]
    hop = math.prod(strides)
    channels = [64, 128, 256, 384, 512, 512]

    encoder = JEPAEncoder(
        sample_rate=SAMPLE_RATE, code_dim=128,
        channels=channels, strides=strides,
        n_res_blocks=args.n_res_blocks,
        n_conformer=args.n_conformer,
        conformer_heads=args.conformer_heads,
        use_gaatn=True,
    )
    if args.stage1_ckpt and Path(args.stage1_ckpt).exists():
        ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
        enc_sd = ckpt.get("state_dict", ckpt.get("encoder", {}))
        if enc_sd:
            try:
                encoder.load_state_dict(enc_sd, strict=True)
            except RuntimeError:
                encoder.load_state_dict(enc_sd, strict=False)
            if is_main:
                print(f"[train] Loaded Stage 1 encoder from {args.stage1_ckpt}")

    # ── Build model ──
    dec_cfg = TransformerDecoderConfig(
        code_dim=args.code_dim,
        hop_length=hop,
        final_patch_size=hop // 4,  # 960/4=240 for 25Hz
    )
    model = V2TransformerCodec(encoder, code_dim=args.code_dim, decoder_cfg=dec_cfg)

    # ── Quantizer ──
    from koe.fast.q2d2 import Q2D2Quantizer
    model.quantizer = Q2D2Quantizer(
        dim=args.code_dim, num_levels=args.q2d2_levels, grid_type="rhombic",
    )
    if is_main:
        print(f"[train] Q2D2 (dim={args.code_dim}, K={args.q2d2_levels})")

    # ── Discriminators ──
    mpd = MultiPeriodDiscriminator()
    msd = MultiScaleDiscriminator()

    # ── Resume ──
    start_step = 0
    resume_path = Path(args.output_dir) / "tf_latest.pt"
    if args.resume_from and Path(args.resume_from).exists():
        resume_path = Path(args.resume_from)
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"], strict=False)
            if is_main:
                print(f"[train] Loaded model from {resume_path}")
        if "disc_state_dict" in ckpt:
            disc_sd = ckpt["disc_state_dict"]
            mpd_sd = {k.replace("mpd.", ""): v for k, v in disc_sd.items() if k.startswith("mpd.")}
            msd_sd = {k.replace("msd.", ""): v for k, v in disc_sd.items() if k.startswith("msd.")}
            if mpd_sd:
                mpd.load_state_dict(mpd_sd, strict=False)
            if msd_sd:
                msd.load_state_dict(msd_sd, strict=False)
        start_step = ckpt.get("step", 0)
        if is_main:
            print(f"[train] Resuming from step {start_step}")

    # ── Move to GPU ──
    model = model.to(device, dtype=torch.bfloat16)
    mpd = mpd.to(device, dtype=torch.bfloat16)
    msd = msd.to(device, dtype=torch.bfloat16)

    if ddp:
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if ddp else model

    if is_main:
        n_total = sum(p.numel() for p in raw_model.parameters()) / 1e6
        n_dec = sum(p.numel() for p in raw_model.decoder.parameters()) / 1e6
        print(f"[train] Total: {n_total:.1f}M, Decoder: {n_dec:.1f}M")

    # ── Encoder fine-tuning ──
    enc_hooks = []
    if args.encoder_lr_scale != 1.0 and not args.freeze_encoder:
        for p in raw_model.encoder.parameters():
            if p.requires_grad:
                enc_hooks.append(
                    p.register_hook(lambda g, s=args.encoder_lr_scale: g * s)
                )
        if is_main:
            print(f"[train] Encoder LR scale: {args.encoder_lr_scale}x")
    if args.freeze_encoder:
        raw_model.encoder.requires_grad_(False)
        raw_model.encoder.eval()
        if is_main:
            print("[train] Encoder FROZEN")

    # ── Optimizers ──
    gen_params = [p for p in model.parameters() if p.requires_grad]
    gen_opt = torch.optim.AdamW(gen_params, lr=args.lr_gen, betas=(0.8, 0.99), weight_decay=0.01)
    disc_params = list(mpd.parameters()) + list(msd.parameters())
    disc_opt = torch.optim.AdamW(disc_params, lr=args.lr_disc, betas=(0.8, 0.99), weight_decay=0.01)

    if resume_path.exists() and "gen_optimizer" in ckpt:
        try:
            gen_opt.load_state_dict(ckpt["gen_optimizer"])
        except Exception:
            pass
    if resume_path.exists() and "disc_optimizer" in ckpt:
        try:
            disc_opt.load_state_dict(ckpt["disc_optimizer"])
        except Exception:
            pass

    # ── Losses ──
    stft_loss_fn = MRSTFTLoss().to(device)
    mel_loss_fn = MultiScaleMelLoss(SAMPLE_RATE).to(device)

    perceptual_loss_fn = None
    if args.lambda_perceptual > 0:
        from koe.fast.losses import WavLMPerceptualLoss
        perceptual_loss_fn = WavLMPerceptualLoss(device=str(device))
        if is_main:
            print(f"[train] WavLM perceptual loss (lambda={args.lambda_perceptual})")

    # ── Data ──
    dataset = AudioDataset(args.data_dir, SAMPLE_RATE, args.max_seconds)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    collate = make_collate(hop)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(sampler is None),
        collate_fn=collate, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=4,
        sampler=sampler,
    )
    if is_main:
        print(f"[train] Dataset: {len(dataset)} files, batch={args.batch_size}, world={world_size}")

    # ── Eval ──
    eval_wavs = make_eval_samples(args.data_dir, hop, n=args.eval_samples) if is_main else []

    # ── Output dir ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── W&B ──
    if is_main and not args.no_wandb:
        try:
            import wandb
            wandb.init(project="koe-v2-tf", name=args.run_name or "tf_decoder",
                       config=vars(args), resume="allow")
            print(f"[train] W&B: {wandb.run.url}")
        except ImportError:
            pass

    # ── SIGTERM ──
    sigterm = [False]
    step = start_step

    def _on_sigterm(signum, frame):
        if is_main:
            print(f"\n[train] SIGTERM at step {step}")
        sigterm[0] = True
    signal.signal(signal.SIGTERM, _on_sigterm)

    # ── Training loop ──
    t0 = time.time()
    t_last_log = t0
    accum = {"g": 0.0, "d": 0.0, "stft": 0.0, "mel": 0.0, "l1": 0.0, "adv": 0.0, "feat": 0.0, "perc": 0.0}
    nan_count = 0

    if is_main:
        print(f"[train] Start step={step}, target={args.total_steps}, "
              f"adv_start={args.adv_start_step}")

    while step < args.total_steps:
        if sampler is not None:
            sampler.set_epoch(step // len(loader))
        for batch in loader:
            if step >= args.total_steps or sigterm[0]:
                break

            wav = batch.to(device, dtype=torch.bfloat16, non_blocking=True)
            use_gan = step >= args.adv_start_step

            # ── Generator forward ──
            wav_rec, indices, aux_loss, z_proj = model(wav)
            stft = stft_loss_fn(wav_rec, wav)
            mel = mel_loss_fn(wav_rec, wav)
            l1 = F.l1_loss(wav_rec, wav)

            g_loss = args.lambda_stft * stft + args.lambda_mel * mel + l1 + aux_loss

            # GAN losses (Phase 2 only)
            g_adv_val, g_feat_val = 0.0, 0.0
            if use_gan:
                set_requires_grad(mpd, False)
                set_requires_grad(msd, False)
                mpd_out = mpd(wav, wav_rec)
                msd_out = msd(wav, wav_rec)
                g_adv = generator_loss(mpd_out[1]) + generator_loss(msd_out[1])
                g_feat = feature_loss(mpd_out[2], mpd_out[3]) + feature_loss(msd_out[2], msd_out[3])
                g_loss = g_loss + args.lambda_gan * (g_adv + args.lambda_feat * g_feat)
                g_adv_val = g_adv.item()
                g_feat_val = g_feat.item()

            # Perceptual
            perc_val = 0.0
            if perceptual_loss_fn is not None:
                p_loss = perceptual_loss_fn(wav_rec, wav)
                g_loss = g_loss + args.lambda_perceptual * p_loss
                perc_val = p_loss.item()

            # NaN check
            g_val = g_loss.item()
            if math.isnan(g_val) or math.isinf(g_val) or g_val > 1e6:
                nan_count += 1
                if is_main:
                    print(f"[train] Bad loss {g_val} step {step} (#{nan_count})")
                if nan_count >= 5:
                    sigterm[0] = True
                    break
                continue
            nan_count = 0

            gen_opt.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen_params, 1.0)
            gen_opt.step()

            # ── Disc step (Phase 2 only) ──
            d_val = 0.0
            if use_gan:
                set_requires_grad(mpd, True)
                set_requires_grad(msd, True)
                rec_det = wav_rec.detach()
                mpd_out_d = mpd(wav, rec_det)
                msd_out_d = msd(wav, rec_det)
                d_loss = discriminator_loss(mpd_out_d[0], mpd_out_d[1]) + \
                         discriminator_loss(msd_out_d[0], msd_out_d[1])
                disc_opt.zero_grad()
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(disc_params, 1.0)
                disc_opt.step()
                set_requires_grad(mpd, False)
                set_requires_grad(msd, False)
                d_val = d_loss.item()

            step += 1

            # Accumulate
            accum["g"] += g_val
            accum["d"] += d_val
            accum["stft"] += stft.item()
            accum["mel"] += mel.item()
            accum["l1"] += l1.item()
            accum["adv"] += g_adv_val
            accum["feat"] += g_feat_val
            accum["perc"] += perc_val

            # LR schedule
            lr_mult = cosine_warmup(step, args.warmup_steps, args.total_steps)
            for pg in gen_opt.param_groups:
                if "initial_lr" not in pg:
                    pg["initial_lr"] = pg["lr"]
                pg["lr"] = pg["initial_lr"] * lr_mult

            # ── Logging ──
            if step % args.log_every == 0 and is_main:
                dt = max(time.time() - t_last_log, 1e-6)
                n = args.log_every
                sps = n / dt
                print(f"step {step}/{args.total_steps} | "
                      f"g={accum['g']/n:.4f} d={accum['d']/n:.4f} "
                      f"stft={accum['stft']/n:.4f} mel={accum['mel']/n:.4f} "
                      f"l1={accum['l1']/n:.4f} | {sps:.2f} sps"
                      f"{' [GAN]' if use_gan else ''}")
                try:
                    import wandb
                    if wandb.run:
                        wandb.log({f"train/{k}": v / n for k, v in accum.items()}, step=step)
                        wandb.log({"train/sps": sps, "train/lr": gen_opt.param_groups[0]["lr"]}, step=step)
                except Exception:
                    pass
                for k in accum:
                    accum[k] = 0.0
                t_last_log = time.time()

            # ── Eval ──
            if step % args.eval_every == 0 and is_main:
                _run_eval(raw_model, eval_wavs, device, step)
                torch.cuda.empty_cache()

            # ── Save ──
            if step % args.save_every == 0 and is_main:
                _save(raw_model, mpd, msd, gen_opt, disc_opt, step, args, output_dir)

        if sigterm[0] or step >= args.total_steps:
            break

    # Cleanup
    for h in enc_hooks:
        h.remove()
    if is_main:
        _save(raw_model, mpd, msd, gen_opt, disc_opt, step, args, output_dir)
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n[train] Done. step={step}, elapsed={time.time()-t0:.0f}s, peak={peak:.1f}GB")
    if ddp:
        dist.destroy_process_group()


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

def _run_eval(model, eval_wavs, device, step):
    import numpy as np
    from koe.eval_codec import compute_pesq, compute_stoi, mel_spectral_distance
    model.eval()
    stois, pesqs, mels = [], [], []
    for wav_t in eval_wavs:
        wav_in = wav_t.to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            rec, _, _, _ = model(wav_in)
        orig = wav_t[0, 0].numpy()
        rec_np = rec[0, 0].float().cpu().numpy()
        n = min(len(orig), len(rec_np))
        stois.append(compute_stoi(orig[:n], rec_np[:n], SAMPLE_RATE))
        pesqs.append(compute_pesq(orig[:n], rec_np[:n], SAMPLE_RATE))
        mels.append(mel_spectral_distance(orig[:n], rec_np[:n], SAMPLE_RATE))
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
    except Exception:
        pass
    model.train()
    if hasattr(model, 'encoder') and not any(p.requires_grad for p in model.encoder.parameters()):
        model.encoder.eval()
    return metrics


def _save(model, mpd, msd, gen_opt, disc_opt, step, args, output_dir):
    disc_sd = {}
    for k, v in mpd.state_dict().items():
        disc_sd[f"mpd.{k}"] = v
    for k, v in msd.state_dict().items():
        disc_sd[f"msd.{k}"] = v
    ckpt = {
        "step": step,
        "state_dict": model.state_dict(),
        "disc_state_dict": disc_sd,
        "gen_optimizer": gen_opt.state_dict(),
        "disc_optimizer": disc_opt.state_dict(),
        "config": {"code_dim": args.code_dim, "decoder": "transformer"},
    }
    for fname in [f"tf_step{step:07d}.pt", "tf_latest.pt"]:
        torch.save(ckpt, output_dir / fname)
    step_ckpts = sorted(output_dir.glob("tf_step*.pt"))
    for old in step_ckpts[:-3]:
        old.unlink()
    print(f"[train] Saved step {step}")


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Transformer decoder training")

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--stage1_ckpt", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)

    # Architecture
    parser.add_argument("--code_dim", type=int, default=32)
    parser.add_argument("--strides", type=str, default="4,4,4,5,3")
    parser.add_argument("--n_res_blocks", type=int, default=8)
    parser.add_argument("--n_conformer", type=int, default=8)
    parser.add_argument("--conformer_heads", type=int, default=16)
    parser.add_argument("--q2d2_levels", type=int, default=4)

    # Training
    parser.add_argument("--total_steps", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr_gen", type=float, default=5e-5)
    parser.add_argument("--lr_disc", type=float, default=2.5e-5)
    parser.add_argument("--encoder_lr_scale", type=float, default=0.1)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--max_seconds", type=float, default=10.0)
    parser.add_argument("--num_workers", type=int, default=2)

    # Two-phase: non-adversarial pretraining then GAN
    parser.add_argument("--adv_start_step", type=int, default=50000,
                        help="Step to begin adversarial training (Phase 2)")

    # Loss weights
    parser.add_argument("--lambda_stft", type=float, default=2.0)
    parser.add_argument("--lambda_mel", type=float, default=10.0)
    parser.add_argument("--lambda_gan", type=float, default=1.0)
    parser.add_argument("--lambda_feat", type=float, default=2.0)
    parser.add_argument("--lambda_perceptual", type=float, default=0.1)

    # Eval / logging
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--eval_samples", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=10)

    # W&B
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
