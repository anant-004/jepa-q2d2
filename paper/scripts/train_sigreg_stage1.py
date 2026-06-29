"""SIGReg-JEPA Stage 1 training — JEPA + Gaussian regularization.

Adds SIGReg loss to the standard JEPA training loop, pushing encoder
embeddings toward N(0, I) — the optimal source for lattice quantizers.

The training objective is:
    L = L_jepa + lambda_sigreg * L_sigreg

where L_jepa is the standard masked prediction loss and L_sigreg
regularizes z_context (online encoder output) toward isotropy.

Based on koe/fast/train_stage1.py with minimal modifications.

Usage:
    python -m paper.scripts.train_sigreg_stage1 \
        --data_dir /data/librilight_9k \
        --output_dir /checkpoints/sigreg_stage1 \
        --lambda_sigreg 0.1
"""

import argparse
import math
import os
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from koe.codec_impl import (
    JEPAEncoder,
    create_jepa_mask,
    jepa_time_len_from_wav,
)
from koe.fast.train_stage1 import (
    cosine_warmup,
    make_fast_collate,
    vectorized_ema_update,
)
from paper.scripts.sigreg import SIGRegLoss, gaussianity_metrics


class AudioDataset(torch.utils.data.Dataset):
    """Load audio files via soundfile (no torchaudio/ffmpeg dependency)."""

    def __init__(self, data_dir: str, sample_rate: int = 24000, max_seconds: float = 15.0):
        import soundfile as sf
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
        for attempt_idx in [idx] + [torch.randint(0, len(self), (1,)).item() for _ in range(3)]:
            path = str(self.files[attempt_idx])
            try:
                info = sf.info(path)
                sr = info.samplerate
                total = info.frames
                need = int(self.max_samples * sr / self.sample_rate) if sr != self.sample_rate else self.max_samples
                if total > need:
                    start = torch.randint(0, total - need, (1,)).item()
                    wav, sr = sf.read(path, start=start, stop=start + need, dtype="float32")
                else:
                    wav, sr = sf.read(path, dtype="float32")

                wav = torch.from_numpy(wav)
                if wav.ndim > 1:
                    wav = wav.mean(dim=-1)

                if sr != self.sample_rate:
                    import torchaudio
                    wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, self.sample_rate).squeeze(0)

                if wav.shape[0] > self.max_samples:
                    start = torch.randint(0, wav.shape[0] - self.max_samples, (1,)).item()
                    wav = wav[start:start + self.max_samples]
                return wav
            except Exception:
                continue
        return torch.zeros(self.max_samples)


def train(args):
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    strides = [int(s) for s in args.strides.split(",")]
    hop_length = math.prod(strides)
    frame_rate = 24000 / hop_length

    print(f"[SIGReg Stage 1] strides={strides}, hop={hop_length}, "
          f"frame_rate={frame_rate:.1f} Hz, lambda_sigreg={args.lambda_sigreg}")

    encoder = JEPAEncoder(
        sample_rate=24000,
        code_dim=128,
        channels=[64, 128, 256, 384, 512, 512],
        strides=strides,
        n_res_blocks=args.n_res_blocks,
        n_conformer=8,
        conformer_heads=16,
        use_gaatn=True,
    )

    n_params = sum(p.numel() for p in encoder.parameters())
    n_train = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"[SIGReg Stage 1] {n_params/1e6:.1f}M total, {n_train/1e6:.1f}M trainable")

    # SIGReg loss
    sigreg = SIGRegLoss(
        lambda_var=1.0,
        lambda_cov=0.04,
        lambda_sw=1.0,
        n_slices=64,
    )

    # Resume
    start_step = 0
    opt_state = None
    wandb_run_id = None
    if args.resume_from and Path(args.resume_from).exists():
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt.get("encoder"))
        if sd:
            try:
                encoder.load_state_dict(sd, strict=True)
                start_step = ckpt.get("step", 0)
                opt_state = ckpt.get("optimizer")
                wandb_run_id = ckpt.get("wandb_run_id")
                print(f"[SIGReg Stage 1] Resumed from step {start_step}")
            except RuntimeError as e:
                print(f"[SIGReg Stage 1] Cannot load checkpoint: {e}")

    encoder = encoder.to(device, dtype=torch.bfloat16)

    compiled_encoder = encoder
    if args.compile:
        print("[SIGReg Stage 1] Compiling encoder...")
        compiled_encoder = torch.compile(encoder, mode="reduce-overhead")

    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=args.lr,
        betas=(0.8, 0.99),
        weight_decay=1e-3,
        fused=True,
    )
    if opt_state is not None:
        try:
            optimizer.load_state_dict(opt_state)
        except Exception:
            pass

    dataset = AudioDataset(args.data_dir, sample_rate=24000, max_seconds=args.max_seconds)
    collate = make_fast_collate(24000, hop_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    print(f"[SIGReg Stage 1] {len(dataset)} files, batch={args.batch_size}")

    if not args.no_wandb:
        try:
            import wandb
            wandb_kwargs = dict(
                project="koe-tokenizer",
                name=args.run_name or "sigreg_stage1",
                config=vars(args),
                resume="allow",
            )
            if wandb_run_id:
                wandb_kwargs["id"] = wandb_run_id
            wandb.init(**wandb_kwargs)
            wandb_run_id = wandb.run.id
        except ImportError:
            pass

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sigterm = [False]
    def _sigterm(signum, frame):
        print(f"\n[SIGReg Stage 1] SIGTERM at step {step}, saving...")
        sigterm[0] = True
    signal.signal(signal.SIGTERM, _sigterm)

    step = start_step
    t0 = time.time()
    t_last_log = t0
    steps_at_last_log = start_step
    accum_jepa = 0.0
    accum_sigreg = 0.0
    mask_ratio = 0.5
    ema_decay = 0.996
    warmup_steps = 1000
    log_every = 10

    print(f"[SIGReg Stage 1] Starting from step {step}/{args.stage1_steps}")

    while step < args.stage1_steps:
        for batch in loader:
            if step >= args.stage1_steps or sigterm[0]:
                break

            wav = batch.to(device, dtype=torch.bfloat16, non_blocking=True)

            T_z = jepa_time_len_from_wav(wav.shape[-1], strides)
            mask = create_jepa_mask(
                batch_size=wav.shape[0], seq_len=T_z,
                mask_ratio=mask_ratio, device=device,
            )

            z_context, z_pred, mask_out, z_target = compiled_encoder(wav, mask)

            # JEPA loss: MSE at masked positions
            inv_mask = (1.0 - mask_out.unsqueeze(1).float())
            n_masked = inv_mask.sum().clamp(min=1)
            jepa_loss = ((z_pred - z_target) ** 2 * inv_mask).sum() / (n_masked * z_pred.shape[1])

            # SIGReg loss on encoder embeddings (z_context is what gets quantized)
            # Ramp up SIGReg over first 1000 steps to avoid gradient explosion
            sigreg_warmup = min(1.0, step / 1000.0)
            z_flat = z_context.permute(0, 2, 1).reshape(-1, z_context.shape[1]).float()
            sigreg_loss, sigreg_m = sigreg(z_flat)

            loss = jepa_loss + args.lambda_sigreg * sigreg_warmup * sigreg_loss.to(jepa_loss.dtype)

            loss_val = loss.item()
            if math.isnan(loss_val) or math.isinf(loss_val) or loss_val > 1e6:
                print(f"[SIGReg Stage 1] Bad loss {loss_val} at step {step}, skipping")
                optimizer.zero_grad(set_to_none=True)
                step += 1
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()

            vectorized_ema_update(encoder, ema_decay)

            lr_mult = cosine_warmup(step, warmup_steps, args.stage1_steps)
            for pg in optimizer.param_groups:
                if "initial_lr" not in pg:
                    pg["initial_lr"] = pg["lr"]
                pg["lr"] = pg["initial_lr"] * lr_mult

            step += 1
            accum_jepa += jepa_loss.item()
            accum_sigreg += sigreg_loss.item()

            if step % log_every == 0:
                t_now = time.time()
                dt = t_now - t_last_log
                avg_jepa = accum_jepa / log_every
                avg_sigreg = accum_sigreg / log_every
                accum_jepa = 0.0
                accum_sigreg = 0.0
                sps = (step - steps_at_last_log) / dt if dt > 0 else 0
                t_last_log = t_now
                steps_at_last_log = step

                print(f"[SIGReg Stage 1] step {step}/{args.stage1_steps} | "
                      f"jepa={avg_jepa:.4f} sig={avg_sigreg:.4f} | "
                      f"lr={lr_mult:.4f} | {sps:.2f} sps")

                try:
                    import wandb
                    if wandb.run:
                        log_d = {
                            "stage1/jepa_loss": avg_jepa,
                            "stage1/sigreg_loss": avg_sigreg,
                            "stage1/total_loss": avg_jepa + args.lambda_sigreg * avg_sigreg,
                            "stage1/lr_mult": lr_mult,
                            "stage1/steps_per_sec": sps,
                        }
                        log_d.update({f"stage1/{k}": v for k, v in sigreg_m.items()})

                        # Gaussianity metrics every 100 steps
                        if step % 100 == 0:
                            with torch.no_grad():
                                g_metrics = gaussianity_metrics(z_flat[:1000])
                                log_d.update({f"stage1/{k}": v for k, v in g_metrics.items()})
                        wandb.log(log_d, step=step)
                except (ImportError, Exception):
                    pass

            # Gaussianity eval
            if args.eval_every > 0 and step % args.eval_every == 0:
                with torch.no_grad():
                    g = gaussianity_metrics(z_flat[:2000])
                    print(f"[SIGReg Stage 1] Gaussianity @ step {step}:")
                    print(f"  erank={g['erank']:.1f}/{z_context.shape[1]} "
                          f"mean_var={g['mean_var']:.3f} "
                          f"max_corr={g['max_abs_corr']:.3f} "
                          f"kurtosis={g['mean_kurtosis']:.3f} "
                          f"sw={g['sw_distance']:.4f}")

            if step % args.save_every == 0:
                ckpt = {
                    "step": step,
                    "encoder": encoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": {
                        "strides": strides,
                        "n_res_blocks": args.n_res_blocks,
                        "hop_length": hop_length,
                        "lambda_sigreg": args.lambda_sigreg,
                        "sigreg": True,
                    },
                }
                if wandb_run_id:
                    ckpt["wandb_run_id"] = wandb_run_id

                for fname in [f"stage1_step{step}.pt", "stage1_latest.pt"]:
                    torch.save(ckpt, output_dir / fname)

                try:
                    import modal
                    modal.Volume.from_name("koe-checkpoints").commit()
                except Exception:
                    pass

                step_ckpts = sorted(output_dir.glob("stage1_step*.pt"))
                for old in step_ckpts[:-3]:
                    old.unlink()

                print(f"[SIGReg Stage 1] Saved step {step}")

        if sigterm[0]:
            break

    final_ckpt = {
        "step": step,
        "encoder": encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": {
            "strides": strides,
            "n_res_blocks": args.n_res_blocks,
            "hop_length": hop_length,
            "lambda_sigreg": args.lambda_sigreg,
            "sigreg": True,
        },
    }
    if wandb_run_id:
        final_ckpt["wandb_run_id"] = wandb_run_id
    for fname in [f"stage1_step{step}.pt", "stage1_latest.pt", "stage1_final.pt"]:
        torch.save(final_ckpt, output_dir / fname)
    try:
        import modal
        modal.Volume.from_name("koe-checkpoints").commit()
    except Exception:
        pass
    print(f"[SIGReg Stage 1] Complete at step {step}")


def main():
    parser = argparse.ArgumentParser(description="SIGReg-JEPA Stage 1 training")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")

    parser.add_argument("--strides", type=str, default="4,4,4,5,3")
    parser.add_argument("--n_res_blocks", type=int, default=8)

    parser.add_argument("--stage1_steps", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_seconds", type=float, default=10.0)
    parser.add_argument("--num_workers", type=int, default=16)

    parser.add_argument("--lambda_sigreg", type=float, default=0.1)

    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--no_compile", dest="compile", action="store_false")
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=5000)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
