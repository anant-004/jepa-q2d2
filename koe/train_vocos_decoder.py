"""Train Vocos decoder: JEPA encoder → VocosDecoder → waveform.

End-to-end training with multi-resolution STFT loss + GAN (MPD+MSD).
The key advantage: fully differentiable through iSTFT, no external vocoder needed.

Usage:
    python -m koe.train_vocos_decoder \
        --stage1_ckpt /checkpoints/tokenizer_v9/stage1_final.pt \
        --data_dir /data/librispeech/all \
        --strides 4,4,4,5,6
"""

import os
import math
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

from koe.config import CodecConfig
from koe.codec_impl import (
    JEPAEncoder, MRSTFTLoss,
    MultiPeriodDiscriminator, MultiScaleDiscriminator,
    discriminator_loss, generator_loss, feature_loss, set_requires_grad,
)
from koe.vocos_decoder import VocosDecoder, VocosDecoderConfig


def build_encoder(args, cfg, device="cuda"):
    """Load frozen JEPA encoder from stage1 checkpoint."""
    encoder = JEPAEncoder(
        sample_rate=cfg.sample_rate, code_dim=cfg.code_dim,
        channels=cfg.channels, strides=cfg.strides,
        n_res_blocks=cfg.n_res_blocks, n_conformer=cfg.n_conformer,
        conformer_heads=cfg.conformer_heads,
    )
    ckpt_path = args.stage1_ckpt
    if ckpt_path.startswith("hf://"):
        parts = ckpt_path[5:].split("/", 2)
        repo_id = f"{parts[0]}/{parts[1]}"
        filename = parts[2] if len(parts) > 2 else "stage1_final.pt"
        from koe.hf_utils import pull_checkpoint_from_hf
        ckpt = pull_checkpoint_from_hf(filename, repo_id=repo_id)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"])
    elif "state_dict" in ckpt:
        encoder.load_state_dict(ckpt["state_dict"])
    print(f"Loaded encoder from {ckpt_path}")
    encoder.requires_grad_(False)
    encoder = encoder.to(device, dtype=torch.bfloat16).eval()
    print(f"Encoder: {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M params (frozen)")
    return encoder


@torch.no_grad()
def _evaluate(decoder, encoder, eval_wavs, step, device, cfg, wandb_module):
    """Evaluate: encoder → decoder → waveform → PESQ/STOI."""
    decoder.eval()
    from pesq import pesq
    from pystoi import stoi

    pesq_scores, stoi_scores = [], []

    for i, wav_cpu in enumerate(eval_wavs):
        wav = wav_cpu.unsqueeze(0).unsqueeze(0).to(device, dtype=torch.bfloat16)
        z_e = encoder.encode(wav)
        target_len = wav_cpu.shape[0]
        wav_hat = decoder(z_e.float(), target_len=target_len)
        wav_hat_np = wav_hat.squeeze().cpu().numpy()
        wav_ref_np = wav_cpu.numpy()
        min_len = min(len(wav_hat_np), len(wav_ref_np))
        wav_hat_np, wav_ref_np = wav_hat_np[:min_len], wav_ref_np[:min_len]

        try:
            w16_h = torchaudio.functional.resample(
                torch.from_numpy(wav_hat_np).unsqueeze(0), 24000, 16000).squeeze().numpy()
            w16_r = torchaudio.functional.resample(
                torch.from_numpy(wav_ref_np).unsqueeze(0), 24000, 16000).squeeze().numpy()
            pesq_scores.append(pesq(16000, w16_r, w16_h, "wb"))
        except Exception as e:
            print(f"  PESQ failed sample {i}: {e}")
        try:
            stoi_scores.append(stoi(wav_ref_np, wav_hat_np, 24000, extended=False))
        except Exception as e:
            print(f"  STOI failed sample {i}: {e}")

    import numpy as np
    avg_pesq = np.mean(pesq_scores) if pesq_scores else 0
    avg_stoi = np.mean(stoi_scores) if stoi_scores else 0
    print(f"[eval step {step}] PESQ={avg_pesq:.3f} ({len(pesq_scores)}/{len(eval_wavs)}) | "
          f"STOI={avg_stoi:.4f} ({len(stoi_scores)}/{len(eval_wavs)})", flush=True)

    if wandb_module:
        wandb_module.log({"eval/pesq": avg_pesq, "eval/stoi": avg_stoi}, step=step)
        # Log audio samples
        try:
            wav = eval_wavs[0].unsqueeze(0).unsqueeze(0).to(device, dtype=torch.bfloat16)
            z_e = encoder.encode(wav)
            wav_hat = decoder(z_e.float(), target_len=eval_wavs[0].shape[0])
            wandb_module.log({
                "eval/audio_reconstructed": wandb_module.Audio(
                    wav_hat.squeeze().cpu().numpy(), sample_rate=24000),
                "eval/audio_original": wandb_module.Audio(
                    eval_wavs[0].numpy(), sample_rate=24000),
            }, step=step)
        except Exception:
            pass

    decoder.train()
    return avg_stoi


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = CodecConfig()
    if args.n_res_blocks is not None:
        cfg.n_res_blocks = args.n_res_blocks
    if args.strides:
        cfg.strides = [int(x) for x in args.strides.split(",")]

    # Encoder
    encoder = build_encoder(args, cfg, device)

    # Vocos decoder
    jepa_hop = math.prod(cfg.strides)
    vocos_cfg = VocosDecoderConfig(jepa_hop=jepa_hop)
    decoder = VocosDecoder(vocos_cfg).to(device)  # float32 — iSTFT needs numerical precision
    n_dec = sum(p.numel() for p in decoder.parameters()) / 1e6
    print(f"VocosDecoder: {n_dec:.1f}M params (trainable)")

    # Discriminators (reuse from codec_impl)
    mpd = MultiPeriodDiscriminator().to(device)  # float32
    msd = MultiScaleDiscriminator().to(device)  # float32
    n_disc = (sum(p.numel() for p in mpd.parameters()) + sum(p.numel() for p in msd.parameters())) / 1e6
    print(f"Discriminators: {n_disc:.1f}M params (MPD+MSD)")

    # Losses
    stft_loss_fn = MRSTFTLoss().to(device)

    # Optimizers
    gen_opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=0.01)
    disc_params = list(mpd.parameters()) + list(msd.parameters())
    disc_opt = torch.optim.AdamW(disc_params, lr=args.lr * 0.5, weight_decay=0.01)

    # LR schedule
    warmup_steps = min(1000, args.total_steps // 10)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(args.total_steps - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    gen_sched = torch.optim.lr_scheduler.LambdaLR(gen_opt, lr_lambda)
    disc_sched = torch.optim.lr_scheduler.LambdaLR(disc_opt, lr_lambda)

    disc_start_step = args.disc_start_step

    # Dataset
    from koe.train_tokenizer import AudioDataset
    from koe.codec_impl import make_collate_fn
    dataset = AudioDataset(args.data_dir, sample_rate=cfg.sample_rate, max_seconds=args.max_seconds)
    collate_fn = make_collate_fn(cfg.sample_rate, jepa_hop)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_fn,
    )
    print(f"Dataset: {len(dataset)} files, batch_size={args.batch_size}")

    # W&B
    try:
        import wandb
        wandb.init(
            project="koe-tokenizer", entity="andy404-bits-pilani",
            name=f"vocos_{args.run_name}",
            config={"decoder": vars(vocos_cfg), "codec": vars(cfg),
                    "lr": args.lr, "batch_size": args.batch_size},
            resume="allow",
        )
    except Exception:
        wandb = None

    # Eval samples
    eval_wavs = [dataset[i] for i in range(min(10, len(dataset)))]

    # Resume
    start_step = 0
    if args.hf_repo:
        try:
            from koe.hf_utils import pull_checkpoint_from_hf
            ckpt = pull_checkpoint_from_hf("vocos_latest.pt", repo_id=args.hf_repo)
            if ckpt is not None:
                decoder.load_state_dict(ckpt["decoder"], strict=False)
                start_step = ckpt.get("step", 0)
                if "gen_opt" in ckpt:
                    gen_opt.load_state_dict(ckpt["gen_opt"])
                if "mpd" in ckpt:
                    mpd.load_state_dict(ckpt["mpd"])
                if "msd" in ckpt:
                    msd.load_state_dict(ckpt["msd"])
                if "disc_opt" in ckpt:
                    disc_opt.load_state_dict(ckpt["disc_opt"])
                for _ in range(start_step):
                    gen_sched.step()
                    disc_sched.step()
                print(f"Resumed from step {start_step}")
        except Exception as e:
            print(f"Resume failed (starting fresh): {e}")

    # Training loop
    step = start_step
    t0 = time.time()
    running_loss = 0.0

    print(f"\nStarting from step {step}, total {args.total_steps}")
    print(f"Disc starts at step {disc_start_step}, eval every {args.eval_every}\n")

    while step < args.total_steps:
        for batch in loader:
            if step >= args.total_steps:
                break

            wav = batch.to(device, dtype=torch.bfloat16)  # [B, 1, T]

            with torch.no_grad():
                z_e = encoder.encode(wav)

            target_len = wav.size(-1)
            wav_hat = decoder(z_e.float(), target_len=target_len)  # [B, T_wav]
            wav_hat = wav_hat.unsqueeze(1)  # [B, 1, T_wav]

            wav_f = wav.float()
            wav_hat_f = wav_hat.float()

            # Multi-resolution STFT loss
            stft_l = stft_loss_fn(wav_hat_f, wav_f)

            # Discriminator step (always forward, update after disc_start_step)
            disc_active = step >= start_step + disc_start_step

            if disc_active:
                set_requires_grad(mpd, True)
                set_requires_grad(msd, True)
                mpd_r, mpd_g, _, _ = mpd(wav_f.detach(), wav_hat_f.detach())
                msd_r, msd_g, _, _ = msd(wav_f.detach(), wav_hat_f.detach())
                d_loss = discriminator_loss(mpd_r, mpd_g) + discriminator_loss(msd_r, msd_g)

                disc_opt.zero_grad()
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(disc_params, 1.0)
                disc_opt.step()
                disc_sched.step()
            else:
                d_loss = torch.tensor(0.0)

            # Generator step
            set_requires_grad(mpd, False)
            set_requires_grad(msd, False)

            loss = stft_l * 2.0  # STFT weight

            if disc_active:
                mpd_r, mpd_g, mpd_fr, mpd_fg = mpd(wav_f, wav_hat_f)
                msd_r, msd_g, msd_fr, msd_fg = msd(wav_f, wav_hat_f)
                g_adv = generator_loss(mpd_g) + generator_loss(msd_g)
                g_feat = feature_loss(mpd_fr, mpd_fg) + feature_loss(msd_fr, msd_fg)
                loss = loss + 0.1 * g_adv + g_feat
            else:
                g_adv = torch.tensor(0.0)
                g_feat = torch.tensor(0.0)

            gen_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            gen_opt.step()
            gen_sched.step()

            running_loss += loss.item()
            step += 1

            # Log
            if step % 10 == 0:
                elapsed = time.time() - t0
                sps = (step - start_step) / elapsed if elapsed > 0 else 0
                avg = running_loss / 10
                running_loss = 0.0
                disc_str = "ON" if disc_active else "warmup"
                print(f"[step {step}/{args.total_steps}] loss={avg:.4f} stft={stft_l.item():.3f} "
                      f"d={d_loss.item():.3f} g_adv={g_adv.item():.3f} disc={disc_str} | "
                      f"{sps:.1f} sps", flush=True)
                if wandb:
                    wandb.log({
                        "vocos/loss": avg, "vocos/stft": stft_l.item(),
                        "vocos/d_loss": d_loss.item(), "vocos/g_adv": g_adv.item(),
                        "vocos/g_feat": g_feat.item(), "vocos/sps": sps,
                        "vocos/lr": gen_sched.get_last_lr()[0],
                    }, step=step)

            # Early audio sample at step 100
            if step == start_step + 100:
                _evaluate(decoder, encoder, eval_wavs[:3], step, device, cfg, wandb)

            # Eval
            if step % args.eval_every == 0:
                _evaluate(decoder, encoder, eval_wavs, step, device, cfg, wandb)

            # Save
            if step % args.save_every == 0:
                _save(decoder, gen_opt, disc_opt, mpd, msd, step, args)

    _save(decoder, gen_opt, disc_opt, mpd, msd, step, args)
    print(f"Training complete at step {step}")


def _save(decoder, gen_opt, disc_opt, mpd, msd, step, args):
    if not args.hf_repo:
        return
    try:
        import tempfile
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(args.hf_repo, private=True, exist_ok=True)
        ckpt = {
            "decoder": decoder.state_dict(), "gen_opt": gen_opt.state_dict(),
            "disc_opt": disc_opt.state_dict(), "mpd": mpd.state_dict(),
            "msd": msd.state_dict(), "step": step,
        }
        for fname in [f"vocos_step{step}.pt", "vocos_latest.pt"]:
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                torch.save(ckpt, f)
                tmp = f.name
            try:
                api.upload_file(path_or_fileobj=tmp, path_in_repo=f"checkpoints/{fname}",
                                repo_id=args.hf_repo)
            finally:
                os.unlink(tmp)
        print(f"Saved step {step} to {args.hf_repo}")
    except Exception as e:
        print(f"Save failed: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--run_name", type=str, default="vocos_v9")
    p.add_argument("--hf_repo", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--total_steps", type=int, default=10000)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--n_res_blocks", type=int, default=8)
    p.add_argument("--strides", type=str, default=None)
    p.add_argument("--disc_start_step", type=int, default=2000)
    p.add_argument("--max_seconds", type=float, default=10.0)
    main_args = p.parse_args()
    train(main_args)


if __name__ == "__main__":
    main()
