"""Train BigVGAN adapter: JEPA encoder → FSQ → MelAdapter → BigVGAN.

Usage:
    python -m koe.train_bigvgan_adapter \
        --stage1_ckpt hf://Andy004/koe-tokenizer-v5/stage1_latest.pt \
        --data_dir /data/librispeech/all \
        --run_name bigvgan_v5 \
        --hf_repo Andy004/koe-bigvgan-v5
"""

import os
import sys
import math
import argparse
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torchaudio

from koe.config import CodecConfig
from koe.codec_impl import JEPAEncoder
from koe.bigvgan_adapter import (
    BigVGANAdapterConfig, MelAdapter, MelDiscriminator, extract_mel,
    load_bigvgan, bigvgan_synthesize,
    mel_disc_loss, mel_gen_loss, mel_feat_match_loss,
)


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
        repo_and_file = ckpt_path[5:]
        parts = repo_and_file.split("/", 2)
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
    print(f"Loaded encoder from {ckpt_path}")

    finetune = getattr(args, 'finetune_encoder', False)
    if not finetune:
        encoder.requires_grad_(False)
    encoder = encoder.to(device, dtype=torch.bfloat16)
    if not finetune:
        encoder.eval()

    n_params = sum(p.numel() for p in encoder.parameters()) / 1e6
    mode = "FINE-TUNING" if finetune else "frozen"
    print(f"Encoder: {n_params:.1f}M params ({mode})")
    return encoder


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = CodecConfig()

    if args.n_res_blocks is not None:
        cfg.n_res_blocks = args.n_res_blocks
    if args.fsq_levels:
        cfg.fsq_levels = [int(x) for x in args.fsq_levels.split(",")]
    if args.strides:
        cfg.strides = [int(x) for x in args.strides.split(",")]

    # Build frozen encoder (no FSQ — use raw z_e for maximum signal)
    encoder = build_encoder(args, cfg, device)

    # Build adapter (hop length determines upsampling ratio)
    jepa_hop = 1
    for s in cfg.strides:
        jepa_hop *= s
    adapter_cfg = BigVGANAdapterConfig(jepa_hop=jepa_hop)
    print(f"Adapter: jepa_hop={jepa_hop}, upsample_factors={adapter_cfg.upsample_factors}, "
          f"subsample={adapter_cfg.subsample}, effective_ratio={jepa_hop / 256:.1f}x")
    adapter = MelAdapter(adapter_cfg).to(device, dtype=torch.bfloat16)
    n_adapter = sum(p.numel() for p in adapter.parameters()) / 1e6
    print(f"MelAdapter: {n_adapter:.1f}M params (trainable)")

    # Mel discriminator for adversarial training (fp32 — small model, avoids dtype mismatches)
    disc = MelDiscriminator(n_mels=adapter_cfg.n_mels, n_scales=3).to(device, dtype=torch.float32)
    n_disc = sum(p.numel() for p in disc.parameters()) / 1e6
    print(f"MelDiscriminator: {n_disc:.1f}M params (trainable)")

    # Load BigVGAN for evaluation
    print("Loading BigVGAN for evaluation...")
    bigvgan_model = load_bigvgan(device)
    bigvgan_model = bigvgan_model.to(dtype=torch.float32)  # BigVGAN needs fp32
    n_bigvgan = sum(p.numel() for p in bigvgan_model.parameters()) / 1e6
    print(f"BigVGAN: {n_bigvgan:.1f}M params (frozen)")

    # Separate optimizers for generator (adapter + optional encoder) and discriminator
    finetune_enc = getattr(args, 'finetune_encoder', False)
    enc_lr_scale = getattr(args, 'encoder_lr_scale', 1.0)
    gen_params = [{"params": adapter.parameters(), "lr": args.lr}]
    if finetune_enc:
        enc_lr = args.lr * enc_lr_scale
        gen_params.append({"params": encoder.parameters(), "lr": enc_lr})
        print(f"Encoder LR: {enc_lr:.1e} (scale={enc_lr_scale})")
    optimizer = torch.optim.AdamW(gen_params, weight_decay=0.01)
    disc_optimizer = torch.optim.AdamW(disc.parameters(), lr=args.lr * 0.5, weight_decay=0.01)

    # Cosine LR schedule with warmup
    warmup_steps = min(1000, args.total_steps // 10)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(args.total_steps - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))  # decays to 10% of peak
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    disc_scheduler = torch.optim.lr_scheduler.LambdaLR(disc_optimizer, lr_lambda)

    # GAN warmup: disc trains from step 0, but adapter only gets GAN loss after disc_warmup_steps
    disc_warmup_steps = args.disc_warmup_steps

    # Dataset
    from koe.train_tokenizer import AudioDataset
    dataset = AudioDataset(args.data_dir, sample_rate=cfg.sample_rate, max_seconds=15.0)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=lambda batch: torch.nn.utils.rnn.pad_sequence(batch, batch_first=True),
    )
    print(f"Dataset: {len(dataset)} files, batch_size={args.batch_size}")

    # Use BigVGAN's own mel extraction for exact compatibility
    # Import BigVGAN's mel function (available when BigVGAN is on PYTHONPATH)
    try:
        from meldataset import get_mel_spectrogram as bigvgan_mel_fn
        import bigvgan as bigvgan_module
        # Load BigVGAN's config (AttrDict with mel params)
        bigvgan_h = bigvgan_model.h  # config from pretrained model
        use_bigvgan_mel = True
        print("Using BigVGAN's native mel extraction for training")
    except ImportError:
        use_bigvgan_mel = False
        print("BigVGAN mel import failed, falling back to torchaudio")

    # W&B
    try:
        import wandb
        wandb.init(
            project="koe-tokenizer", entity="andy404-bits-pilani",
            name=f"bigvgan_{args.run_name}",
            config={"adapter": vars(adapter_cfg), "codec": vars(cfg),
                    "lr": args.lr, "batch_size": args.batch_size,
                    "data_dir": args.data_dir},
            resume="allow",
        )
    except Exception:
        wandb = None

    # Eval samples
    eval_wavs = []
    for i, wav in enumerate(dataset):
        if i >= 10:
            break
        eval_wavs.append(wav)

    # Resume from local save_dir or HF
    start_step = 0
    resume_ckpt = None
    save_dir = getattr(args, 'save_dir', None)
    if save_dir:
        from pathlib import Path
        local_latest = Path(save_dir) / "adapter_latest.pt"
        if local_latest.exists():
            resume_ckpt = torch.load(local_latest, map_location="cpu", weights_only=False)
            print(f"Found local checkpoint at {local_latest}")
    if resume_ckpt is None and args.hf_repo:
        try:
            from koe.hf_utils import pull_checkpoint_from_hf
            resume_ckpt = pull_checkpoint_from_hf("adapter_latest.pt", repo_id=args.hf_repo)
            if resume_ckpt is not None:
                print("Found HF checkpoint")
        except Exception as e:
            print(f"HF resume failed: {e}")
    if resume_ckpt is not None:
        adapter.load_state_dict(resume_ckpt["adapter"], strict=False)
        start_step = resume_ckpt.get("step", 0)
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        if "disc" in resume_ckpt:
            disc.load_state_dict(resume_ckpt["disc"])
        if "disc_optimizer" in resume_ckpt:
            disc_optimizer.load_state_dict(resume_ckpt["disc_optimizer"])
        if "scheduler" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler"])
        else:
            for _ in range(start_step):
                scheduler.step()
        if "disc_scheduler" in resume_ckpt:
            disc_scheduler.load_state_dict(resume_ckpt["disc_scheduler"])
        else:
            for _ in range(start_step):
                disc_scheduler.step()
        if "encoder" in resume_ckpt and getattr(args, 'finetune_encoder', False):
            encoder.load_state_dict(resume_ckpt["encoder"])
            print("Restored fine-tuned encoder weights")
        print(f"Resumed from step {start_step}")

    # Training loop
    step = start_step
    t0 = time.time()
    running_loss = 0.0
    total_steps = args.total_steps

    print(f"\nStarting training from step {step}, total {total_steps}")
    print(f"Eval every {args.eval_every}, save every {args.save_every}\n")

    while step < total_steps:
        for batch in loader:
            if step >= total_steps:
                break

            wav = batch.unsqueeze(1).to(device, dtype=torch.bfloat16)  # [B, 1, T]

            # Forward through encoder (frozen or fine-tuned)
            if finetune_enc:
                z_e = encoder.encode(wav)       # [B, 128, T_z]
            else:
                with torch.no_grad():
                    z_e = encoder.encode(wav)

            # Ground truth mel from audio (use BigVGAN's extraction for exact match)
            with torch.no_grad():
                wav_float = wav.squeeze(1).float()  # [B, T] fp32 for mel
                if use_bigvgan_mel:
                    gt_mel = bigvgan_mel_fn(wav_float, bigvgan_h)  # [B, 100, T_mel]
                else:
                    gt_mel = extract_mel(wav_float, adapter_cfg)
                gt_mel = gt_mel.to(dtype=torch.bfloat16)

            # Adapter forward
            pred_mel = adapter(z_e, target_len=gt_mel.size(-1))  # [B, 100, T_mel]

            # Combined mel losses: L1 + L2 + temporal gradient penalty
            pred_f = pred_mel.float()
            gt_f = gt_mel.float()
            mel_l1 = F.l1_loss(pred_f, gt_f)
            mel_l2 = F.mse_loss(pred_f, gt_f)
            # Temporal smoothness: penalize difference in mel deltas
            if pred_f.size(-1) > 1:
                pred_delta = pred_f[:, :, 1:] - pred_f[:, :, :-1]
                gt_delta = gt_f[:, :, 1:] - gt_f[:, :, :-1]
                delta_loss = F.l1_loss(pred_delta, gt_delta)
            else:
                delta_loss = torch.tensor(0.0, device=device)

            recon_loss = mel_l1 + 0.5 * mel_l2 + 0.5 * delta_loss

            # --- Discriminator step (always runs, matches Stage 2 pattern) ---
            disc_real = disc(gt_f.detach())
            disc_fake = disc(pred_f.detach())
            d_loss = mel_disc_loss(disc_real, disc_fake)

            disc_optimizer.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            disc_optimizer.step()
            disc_scheduler.step()

            # --- Generator step (adapter) ---
            # GAN loss: disc forward on predicted mels (with adapter gradients)
            # Re-run disc on real for fresh features (disc weights just updated)
            with torch.no_grad():
                disc_real_for_gen = disc(gt_f)
            disc_fake_for_gen = disc(pred_f)
            g_adv = mel_gen_loss(disc_fake_for_gen)
            g_feat = mel_feat_match_loss(disc_real_for_gen, disc_fake_for_gen)

            # Gate GAN loss during warmup (disc trains but adapter ignores GAN signal)
            if step < start_step + disc_warmup_steps:
                gan_weight = 0.0
            else:
                gan_weight = args.lambda_gan

            loss = recon_loss + gan_weight * (g_adv + g_feat)

            optimizer.zero_grad()
            loss.backward()
            clip_params = list(adapter.parameters())
            if finetune_enc:
                clip_params += list(encoder.parameters())
            torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            running_loss += loss_val
            step += 1

            # Log
            if step % 10 == 0:
                elapsed = time.time() - t0
                sps = (step - start_step) / elapsed if elapsed > 0 else 0
                avg_loss = running_loss / 10
                running_loss = 0.0
                gan_active = "ON" if gan_weight > 0 else "warmup"
                print(f"[step {step}/{total_steps}] loss={avg_loss:.4f} d={d_loss.item():.3f} "
                      f"g_adv={g_adv.item():.3f} gan={gan_active} | {sps:.1f} steps/s",
                      flush=True)
                if wandb:
                    log_dict = {
                        "adapter/loss": avg_loss,
                        "adapter/recon_loss": recon_loss.item(),
                        "adapter/d_loss": d_loss.item(),
                        "adapter/g_adv": g_adv.item(),
                        "adapter/g_feat": g_feat.item(),
                        "adapter/gan_weight": gan_weight,
                        "adapter/steps_per_sec": sps,
                        "adapter/lr": scheduler.get_last_lr()[0],
                    }
                    wandb.log(log_dict, step=step)

            # Eval
            if step % args.eval_every == 0:
                bigvgan_mel_info = (use_bigvgan_mel,
                                   bigvgan_mel_fn if use_bigvgan_mel else None,
                                   bigvgan_h if use_bigvgan_mel else None)
                _evaluate(adapter, encoder, None, bigvgan_model, bigvgan_mel_info,
                          eval_wavs, step, device, adapter_cfg, wandb)

            # Save
            if step % args.save_every == 0:
                _save_checkpoint(adapter, optimizer, step, args, scheduler, disc, disc_optimizer, disc_scheduler, encoder=encoder)

    # Final save
    _save_checkpoint(adapter, optimizer, step, args, scheduler, disc, disc_optimizer, disc_scheduler, encoder=encoder)
    print(f"Training complete at step {step}")


@torch.no_grad()
def _evaluate(adapter, encoder, fsq, bigvgan_model, bigvgan_mel_info,
              eval_wavs, step, device, adapter_cfg, wandb_module):
    """Evaluate: predict mel → BigVGAN → waveform → PESQ/STOI."""
    adapter.eval()
    from pesq import pesq
    from pystoi import stoi

    use_bigvgan_mel, bigvgan_mel_fn, bigvgan_h = bigvgan_mel_info

    pesq_scores = []
    stoi_scores = []
    mel_losses = []

    for wav_cpu in eval_wavs:
        wav = wav_cpu.unsqueeze(0).unsqueeze(0).to(device, dtype=torch.bfloat16)

        # Encode (use raw z_e if no FSQ)
        z_e = encoder.encode(wav)
        z_q = z_e if fsq is None else fsq(z_e)[0]

        # Ground truth mel
        wav_float = wav.squeeze(1).float()
        if use_bigvgan_mel:
            gt_mel = bigvgan_mel_fn(wav_float, bigvgan_h)
        else:
            gt_mel = extract_mel(wav_float, adapter_cfg)

        # Predict mel
        pred_mel = adapter(z_q.to(torch.bfloat16), target_len=gt_mel.size(-1))
        mel_loss = F.l1_loss(pred_mel.float(), gt_mel).item()
        mel_losses.append(mel_loss)

        # Synthesize via BigVGAN (needs fp32)
        wav_hat = bigvgan_synthesize(bigvgan_model, pred_mel.float())
        wav_hat = wav_hat.squeeze().cpu().numpy()
        wav_ref = wav_cpu.numpy()

        # Trim to same length
        min_len = min(len(wav_hat), len(wav_ref))
        wav_hat = wav_hat[:min_len]
        wav_ref = wav_ref[:min_len]

        # Metrics — PESQ needs 16kHz
        try:
            wav_hat_16k = torchaudio.functional.resample(
                torch.from_numpy(wav_hat).unsqueeze(0), 24000, 16000).squeeze().numpy()
            wav_ref_16k = torchaudio.functional.resample(
                torch.from_numpy(wav_ref).unsqueeze(0), 24000, 16000).squeeze().numpy()
            p = pesq(16000, wav_ref_16k, wav_hat_16k, "wb")
            pesq_scores.append(p)
        except Exception as e:
            print(f"  PESQ failed: {e}")
        try:
            s = stoi(wav_ref, wav_hat, 24000, extended=False)
            stoi_scores.append(s)
        except Exception as e:
            print(f"  STOI failed: {e}")

    n_eval = len(eval_wavs)
    avg_mel = sum(mel_losses) / len(mel_losses) if mel_losses else 0
    avg_pesq = sum(pesq_scores) / len(pesq_scores) if pesq_scores else 0
    avg_stoi = sum(stoi_scores) / len(stoi_scores) if stoi_scores else 0

    print(f"[eval step {step}] mel_loss={avg_mel:.4f} | PESQ={avg_pesq:.3f} ({len(pesq_scores)}/{n_eval}) | STOI={avg_stoi:.4f} ({len(stoi_scores)}/{n_eval})",
          flush=True)

    if wandb_module:
        wandb_module.log({
            "eval/mel_loss": avg_mel,
            "eval/pesq": avg_pesq,
            "eval/stoi": avg_stoi,
        }, step=step)

        # Log one audio sample
        try:
            wav = eval_wavs[0].unsqueeze(0).unsqueeze(0).to(device, dtype=torch.bfloat16)
            z_e = encoder.encode(wav)
            z_q = z_e if fsq is None else fsq(z_e)[0]
            wav_float = wav.squeeze(1).float()
            if use_bigvgan_mel:
                gt_mel = bigvgan_mel_fn(wav_float, bigvgan_h)
            else:
                gt_mel = extract_mel(wav_float, adapter_cfg)
            pred_mel = adapter(z_q.to(torch.bfloat16), target_len=gt_mel.size(-1))
            wav_hat = bigvgan_synthesize(bigvgan_model, pred_mel.float())
            wandb_module.log({
                "eval/audio_reconstructed": wandb_module.Audio(
                    wav_hat.squeeze().cpu().numpy(), sample_rate=24000),
                "eval/audio_original": wandb_module.Audio(
                    eval_wavs[0].numpy(), sample_rate=24000),
            }, step=step)
        except Exception:
            pass

    adapter.train()


def _save_checkpoint(adapter, optimizer, step, args, scheduler=None,
                     disc=None, disc_optimizer=None, disc_scheduler=None,
                     encoder=None):
    """Save checkpoint to local dir (Modal volume) and optionally HF."""
    ckpt = {
        "adapter": adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    if disc is not None:
        ckpt["disc"] = disc.state_dict()
    if disc_optimizer is not None:
        ckpt["disc_optimizer"] = disc_optimizer.state_dict()
    if disc_scheduler is not None:
        ckpt["disc_scheduler"] = disc_scheduler.state_dict()
    if encoder is not None and getattr(args, 'finetune_encoder', False):
        ckpt["encoder"] = encoder.state_dict()

    # Save to local directory (Modal volume)
    save_dir = getattr(args, 'save_dir', None)
    if save_dir:
        from pathlib import Path
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        for fname in [f"adapter_step{step}.pt", "adapter_latest.pt"]:
            torch.save(ckpt, out / fname)
        # Keep only last 3 step checkpoints
        step_ckpts = sorted(out.glob("adapter_step*.pt"))
        for old in step_ckpts[:-3]:
            old.unlink()
        # Commit volume if available
        try:
            import modal
            modal.Volume.from_name("checkpoints").commit()
        except Exception:
            pass
        print(f"Saved checkpoint at step {step} to {save_dir}")

    # Also try HF upload
    if args.hf_repo:
        try:
            import tempfile
            from huggingface_hub import HfApi
            api = HfApi(token=os.environ.get("HF_TOKEN"))
            api.create_repo(args.hf_repo, private=True, exist_ok=True)
            for filename in [f"adapter_step{step}.pt", "adapter_latest.pt"]:
                with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                    torch.save(ckpt, f)
                    tmp_path = f.name
                try:
                    api.upload_file(
                        path_or_fileobj=tmp_path,
                        path_in_repo=f"checkpoints/{filename}",
                        repo_id=args.hf_repo,
                    )
                finally:
                    os.unlink(tmp_path)
            print(f"Also uploaded to {args.hf_repo}")
        except Exception as e:
            print(f"HF upload failed (non-fatal): {e}")


def main():
    parser = argparse.ArgumentParser(description="Train BigVGAN adapter")
    parser.add_argument("--stage1_ckpt", type=str, required=True,
                        help="Path to Stage 1 encoder checkpoint (hf:// or local)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="bigvgan_adapter")
    parser.add_argument("--hf_repo", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--total_steps", type=int, default=50000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--n_res_blocks", type=int, default=8)
    parser.add_argument("--fsq_levels", type=str, default=None,
                        help="FSQ levels e.g. '8,8,8,8'")
    parser.add_argument("--strides", type=str, default=None,
                        help="Override encoder strides e.g. '4,4,4,5,6' for 12.5 Hz")
    parser.add_argument("--disc_warmup_steps", type=int, default=2000,
                        help="Steps before GAN loss activates for adapter (disc trains from step 0)")
    parser.add_argument("--lambda_gan", type=float, default=0.1,
                        help="Weight for adversarial + feature matching loss after warmup")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Local directory for checkpoint saving (e.g. Modal volume)")
    parser.add_argument("--finetune_encoder", action="store_true",
                        help="Fine-tune encoder jointly with adapter")
    parser.add_argument("--encoder_lr_scale", type=float, default=0.1,
                        help="LR multiplier for encoder when fine-tuning (default: 0.1)")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
