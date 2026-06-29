"""Train flow matching decoder: JEPA encoder → FlowMatchingDecoder → mel → BigVGAN → waveform.

Conditional flow matching: learns to transport Gaussian noise to mel spectrograms,
conditioned on JEPA encoder features. At inference, Euler ODE integration produces
sharp mel predictions (unlike regression-based adapters).

Usage:
    python -m koe.train_flow_decoder \
        --stage1_ckpt /checkpoints/tokenizer_v9/stage1_final.pt \
        --data_dir /data/librispeech/all \
        --strides 4,4,4,5,6
"""

import os
import math
import argparse
import time

import torch
import torch.nn.functional as F
import torchaudio

from koe.config import CodecConfig
from koe.codec_impl import JEPAEncoder
from koe.flow_decoder import FlowMatchingDecoder, FlowDecoderConfig, extract_mel


def build_encoder(args, cfg, device="cuda"):
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
    finetune = getattr(args, 'finetune_encoder', False)
    if not finetune:
        encoder.requires_grad_(False)
    encoder = encoder.to(device, dtype=torch.bfloat16)
    if not finetune:
        encoder.eval()
    mode = "FINE-TUNING" if finetune else "frozen"
    print(f"Encoder: {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M params ({mode})")
    return encoder


@torch.no_grad()
def _evaluate(flow_decoder, encoder, bigvgan_model, bigvgan_mel_info,
              eval_wavs, step, device, flow_cfg, wandb_module, n_ode_steps=32):
    """Evaluate: encoder → flow decoder → mel → BigVGAN → waveform → PESQ/STOI."""
    flow_decoder.eval()
    from pesq import pesq
    from pystoi import stoi
    import numpy as np

    use_bigvgan_mel, bigvgan_mel_fn, bigvgan_h = bigvgan_mel_info

    pesq_scores, stoi_scores, mel_losses = [], [], []

    for i, wav_cpu in enumerate(eval_wavs):
        wav = wav_cpu.unsqueeze(0).unsqueeze(0).to(device, dtype=torch.bfloat16)
        z_e = encoder.encode(wav)

        # Ground truth mel
        wav_float = wav.squeeze(1).float()
        if use_bigvgan_mel:
            gt_mel = bigvgan_mel_fn(wav_float, bigvgan_h)
        else:
            gt_mel = extract_mel(wav_float, flow_cfg)

        # Flow matching inference
        pred_mel = flow_decoder.sample(z_e.float(), n_steps=n_ode_steps, target_len=gt_mel.size(-1))
        mel_l1 = F.l1_loss(pred_mel, gt_mel.float()).item()
        mel_losses.append(mel_l1)

        # BigVGAN synthesis
        from koe.bigvgan_adapter import bigvgan_synthesize
        wav_hat = bigvgan_synthesize(bigvgan_model, pred_mel.float())
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

    avg_mel = np.mean(mel_losses) if mel_losses else 0
    avg_pesq = np.mean(pesq_scores) if pesq_scores else 0
    avg_stoi = np.mean(stoi_scores) if stoi_scores else 0

    print(f"[eval step {step}] mel_l1={avg_mel:.4f} | PESQ={avg_pesq:.3f} ({len(pesq_scores)}/{len(eval_wavs)}) | "
          f"STOI={avg_stoi:.4f} ({len(stoi_scores)}/{len(eval_wavs)})", flush=True)

    if wandb_module:
        wandb_module.log({
            "eval/mel_l1": avg_mel, "eval/pesq": avg_pesq, "eval/stoi": avg_stoi,
        }, step=step)
        # Log audio
        try:
            wav = eval_wavs[0].unsqueeze(0).unsqueeze(0).to(device, dtype=torch.bfloat16)
            z_e = encoder.encode(wav)
            wav_float = wav.squeeze(1).float()
            if use_bigvgan_mel:
                gt_mel = bigvgan_mel_fn(wav_float, bigvgan_h)
            else:
                gt_mel = extract_mel(wav_float, flow_cfg)
            pred_mel = flow_decoder.sample(z_e.float(), n_steps=n_ode_steps, target_len=gt_mel.size(-1))
            wav_hat = bigvgan_synthesize(bigvgan_model, pred_mel)
            wandb_module.log({
                "eval/audio_reconstructed": wandb_module.Audio(
                    wav_hat.squeeze().cpu().numpy(), sample_rate=24000),
                "eval/audio_original": wandb_module.Audio(
                    eval_wavs[0].numpy(), sample_rate=24000),
            }, step=step)
        except Exception:
            pass

    flow_decoder.train()
    return avg_stoi


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = CodecConfig()
    if args.n_res_blocks is not None:
        cfg.n_res_blocks = args.n_res_blocks
    if args.strides:
        cfg.strides = [int(x) for x in args.strides.split(",")]

    encoder = build_encoder(args, cfg, device)

    jepa_hop = math.prod(cfg.strides)
    flow_cfg = FlowDecoderConfig(jepa_hop=jepa_hop)
    flow_decoder = FlowMatchingDecoder(flow_cfg).to(device)  # float32 for numerical stability
    n_flow = sum(p.numel() for p in flow_decoder.parameters()) / 1e6
    print(f"FlowMatchingDecoder: {n_flow:.1f}M params")

    # BigVGAN for eval
    print("Loading BigVGAN for evaluation...")
    from koe.bigvgan_adapter import load_bigvgan
    bigvgan_model = load_bigvgan(device).to(dtype=torch.float32)
    print(f"BigVGAN: {sum(p.numel() for p in bigvgan_model.parameters())/1e6:.1f}M (frozen)")

    try:
        from meldataset import get_mel_spectrogram as bigvgan_mel_fn
        bigvgan_h = bigvgan_model.h
        use_bigvgan_mel = True
        print("Using BigVGAN's native mel extraction")
    except ImportError:
        use_bigvgan_mel = False
        bigvgan_mel_fn, bigvgan_h = None, None
        print("Falling back to torchaudio mel extraction")

    bigvgan_mel_info = (use_bigvgan_mel, bigvgan_mel_fn, bigvgan_h)

    finetune_enc = getattr(args, 'finetune_encoder', False)
    enc_lr_scale = getattr(args, 'encoder_lr_scale', 1.0)
    gen_params = [{"params": flow_decoder.parameters(), "lr": args.lr}]
    if finetune_enc:
        enc_lr = args.lr * enc_lr_scale
        gen_params.append({"params": encoder.parameters(), "lr": enc_lr})
        print(f"Encoder LR: {enc_lr:.1e} (scale={enc_lr_scale})")
    optimizer = torch.optim.AdamW(gen_params, weight_decay=0.01)
    warmup_steps = min(1000, args.total_steps // 10)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(args.total_steps - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Dataset
    from koe.train_tokenizer import AudioDataset
    dataset = AudioDataset(args.data_dir, sample_rate=cfg.sample_rate, max_seconds=args.max_seconds)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=lambda batch: torch.nn.utils.rnn.pad_sequence(batch, batch_first=True),
    )
    print(f"Dataset: {len(dataset)} files, batch_size={args.batch_size}")

    # W&B
    try:
        import wandb
        wandb.init(
            project="koe-tokenizer", entity="andy404-bits-pilani",
            name=f"flow_{args.run_name}",
            config={"flow": vars(flow_cfg), "codec": vars(cfg),
                    "lr": args.lr, "batch_size": args.batch_size},
            resume="allow",
        )
    except Exception:
        wandb = None

    eval_wavs = [dataset[i] for i in range(min(10, len(dataset)))]

    # Resume from local save_dir or HF
    start_step = 0
    resume_ckpt = None
    save_dir = getattr(args, 'save_dir', None)
    if save_dir:
        from pathlib import Path
        local_latest = Path(save_dir) / "flow_latest.pt"
        if local_latest.exists():
            resume_ckpt = torch.load(local_latest, map_location="cpu", weights_only=False)
            print(f"Found local checkpoint at {local_latest}")
    if resume_ckpt is None and args.hf_repo:
        try:
            from koe.hf_utils import pull_checkpoint_from_hf
            resume_ckpt = pull_checkpoint_from_hf("flow_latest.pt", repo_id=args.hf_repo)
            if resume_ckpt is not None:
                print("Found HF checkpoint")
        except Exception as e:
            print(f"HF resume failed: {e}")
    if resume_ckpt is not None:
        flow_decoder.load_state_dict(resume_ckpt["decoder"], strict=False)
        start_step = resume_ckpt.get("step", 0)
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        for _ in range(start_step):
            scheduler.step()
        if "encoder" in resume_ckpt and getattr(args, 'finetune_encoder', False):
            encoder.load_state_dict(resume_ckpt["encoder"])
            print("Restored fine-tuned encoder weights")
        print(f"Resumed from step {start_step}")

    step = start_step
    t0 = time.time()
    running_loss = 0.0

    print(f"\nStarting from step {step}, total {args.total_steps}")
    print(f"Eval every {args.eval_every}, ODE steps for eval: {args.n_ode_steps}\n")

    while step < args.total_steps:
        for batch in loader:
            if step >= args.total_steps:
                break

            wav = batch.unsqueeze(1).to(device, dtype=torch.bfloat16)  # [B, 1, T]

            if finetune_enc:
                z_e = encoder.encode(wav)
                with torch.no_grad():
                    wav_float = wav.squeeze(1).float()
                    if use_bigvgan_mel:
                        gt_mel = bigvgan_mel_fn(wav_float, bigvgan_h)
                    else:
                        gt_mel = extract_mel(wav_float, flow_cfg)
            else:
                with torch.no_grad():
                    z_e = encoder.encode(wav)
                    wav_float = wav.squeeze(1).float()
                    if use_bigvgan_mel:
                        gt_mel = bigvgan_mel_fn(wav_float, bigvgan_h)
                    else:
                        gt_mel = extract_mel(wav_float, flow_cfg)

            loss = flow_decoder.compute_loss(z_e.float(), gt_mel.float())

            optimizer.zero_grad()
            loss.backward()
            clip_params = list(flow_decoder.parameters())
            if finetune_enc:
                clip_params += list(encoder.parameters())
            torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            step += 1

            if step % 10 == 0:
                elapsed = time.time() - t0
                sps = (step - start_step) / elapsed if elapsed > 0 else 0
                avg = running_loss / 10
                running_loss = 0.0
                print(f"[step {step}/{args.total_steps}] flow_loss={avg:.4f} | {sps:.1f} sps",
                      flush=True)
                if wandb:
                    wandb.log({
                        "flow/loss": avg, "flow/sps": sps,
                        "flow/lr": scheduler.get_last_lr()[0],
                    }, step=step)

            # Early eval at step 100
            if step == start_step + 100:
                _evaluate(flow_decoder, encoder, bigvgan_model, bigvgan_mel_info,
                          eval_wavs[:3], step, device, flow_cfg, wandb, args.n_ode_steps)

            if step % args.eval_every == 0:
                _evaluate(flow_decoder, encoder, bigvgan_model, bigvgan_mel_info,
                          eval_wavs, step, device, flow_cfg, wandb, args.n_ode_steps)

            if step % args.save_every == 0:
                _save(flow_decoder, optimizer, step, args, encoder=encoder)

    _save(flow_decoder, optimizer, step, args, encoder=encoder)
    print(f"Training complete at step {step}")


def _save(decoder, optimizer, step, args, encoder=None):
    ckpt = {"decoder": decoder.state_dict(), "optimizer": optimizer.state_dict(), "step": step}
    if encoder is not None and getattr(args, 'finetune_encoder', False):
        ckpt["encoder"] = encoder.state_dict()

    # Save to local directory (Modal volume)
    save_dir = getattr(args, 'save_dir', None)
    if save_dir:
        from pathlib import Path
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        for fname in [f"flow_step{step}.pt", "flow_latest.pt"]:
            torch.save(ckpt, out / fname)
        # Keep only last 3 step checkpoints
        step_ckpts = sorted(out.glob("flow_step*.pt"))
        for old in step_ckpts[:-3]:
            old.unlink()
        try:
            import modal
            modal.Volume.from_name("checkpoints").commit()
        except Exception:
            pass
        print(f"Saved step {step} to {save_dir}")

    # Also try HF upload
    if args.hf_repo:
        try:
            import tempfile
            from huggingface_hub import HfApi
            api = HfApi(token=os.environ.get("HF_TOKEN"))
            api.create_repo(args.hf_repo, private=True, exist_ok=True)
            for fname in [f"flow_step{step}.pt", "flow_latest.pt"]:
                with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                    torch.save(ckpt, f)
                    tmp = f.name
                try:
                    api.upload_file(path_or_fileobj=tmp, path_in_repo=f"checkpoints/{fname}",
                                    repo_id=args.hf_repo)
                finally:
                    os.unlink(tmp)
            print(f"Also uploaded to {args.hf_repo}")
        except Exception as e:
            print(f"HF upload failed (non-fatal): {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--run_name", type=str, default="flow_v9")
    p.add_argument("--hf_repo", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--total_steps", type=int, default=10000)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--n_res_blocks", type=int, default=8)
    p.add_argument("--strides", type=str, default=None)
    p.add_argument("--n_ode_steps", type=int, default=32)
    p.add_argument("--max_seconds", type=float, default=10.0)
    p.add_argument("--save_dir", type=str, default=None,
                   help="Local directory for checkpoint saving (e.g. Modal volume)")
    p.add_argument("--finetune_encoder", action="store_true",
                   help="Fine-tune encoder jointly with flow decoder")
    p.add_argument("--encoder_lr_scale", type=float, default=0.1,
                   help="LR multiplier for encoder when fine-tuning")
    main_args = p.parse_args()
    train(main_args)


if __name__ == "__main__":
    main()
