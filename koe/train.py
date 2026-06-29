"""Train the KoeTTS AR speech language model (Phase 2).

Trains on pre-encoded audio tokens (from scripts/pre_encode.py).
Uses DeepSpeed ZeRO-2 for distributed training on 4x H200s.

Usage:
    # Single GPU (DeepSpeed with 1 GPU)
    deepspeed --num_gpus=1 -m koe.train \
        --manifest data/libritts_r/encoded_manifest.jsonl \
        --output_dir checkpoints/ar_model \
        --deepspeed configs/ds_zero2.json

    # 4x GPU
    deepspeed --num_gpus=4 -m koe.train \
        --manifest data/libritts_r/encoded_manifest.jsonl \
        --output_dir checkpoints/ar_model \
        --deepspeed configs/ds_zero2.json

    # Resume from checkpoint
    deepspeed --num_gpus=4 -m koe.train \
        --manifest data/libritts_r/encoded_manifest.jsonl \
        --output_dir checkpoints/ar_model \
        --deepspeed configs/ds_zero2.json \
        --resume checkpoints/ar_model/step_10000
"""

import argparse
import math
import os
import time
from functools import partial
from pathlib import Path

import deepspeed
import torch
from torch.utils.data import DataLoader, DistributedSampler

from koe.config import ModelConfig, TokenConfig, TrainConfig
from koe.dataset import TTSDataset, collate_fn
from koe.model import KoeTTS
from koe.text import CharTokenizer


# ──────────────────────────────────────────────────────────────
# Optimizer setup — Muon for 2D, AdamW for 1D
# ──────────────────────────────────────────────────────────────

def split_params(model: KoeTTS):
    """Split model params into Muon-eligible (2D) and AdamW (1D/other)."""
    muon_params = []
    adam_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2 and param.shape[-1] > 1:
            muon_params.append(param)
        else:
            adam_params.append(param)
    return muon_params, adam_params


def build_optimizer(model: KoeTTS, cfg: TrainConfig):
    """Build Muon + AdamW optimizer, or fall back to pure AdamW.

    DeepSpeed's ZeRO-2 shards optimizer states across GPUs.
    We set zero_allow_untested_optimizer=true in the DS config
    to allow Muon (a non-standard optimizer) to work with ZeRO.
    """
    muon_params, adam_params = split_params(model)

    try:
        from muon import MuonWithAuxAdam

        optimizer = MuonWithAuxAdam(
            muon_params=[{"params": muon_params, "lr": cfg.muon_lr}],
            lr=cfg.muon_lr,
            momentum=cfg.muon_momentum,
            nesterov=cfg.muon_nesterov,
            aux_optimizer_cls=torch.optim.AdamW,
            aux_optimizer_kwargs={"weight_decay": cfg.weight_decay},
            aux_params=[{"params": adam_params, "lr": cfg.adam_lr, "betas": cfg.adam_betas}],
        )
        opt_type = "muon"
    except ImportError:
        print("[WARN] Muon not found, falling back to AdamW")
        optimizer = torch.optim.AdamW(
            [
                {"params": muon_params, "lr": cfg.adam_lr, "betas": cfg.adam_betas},
                {"params": adam_params, "lr": cfg.adam_lr, "betas": cfg.adam_betas},
            ],
            weight_decay=cfg.weight_decay,
        )
        opt_type = "adamw"

    return optimizer, opt_type, muon_params, adam_params


def cosine_warmup_schedule(step, warmup_steps, total_steps, min_ratio=0.1):
    """Cosine decay with linear warmup. Returns multiplier in [min_ratio, 1]."""
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress))


# ──────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────

def train(args):
    # DeepSpeed handles distributed init
    deepspeed.init_distributed()

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = rank == 0

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Configs
    model_cfg = ModelConfig()
    token_cfg = TokenConfig()
    train_cfg = TrainConfig()

    if args.max_steps:
        train_cfg.max_steps = args.max_steps
    if args.max_epochs:
        train_cfg.max_epochs = args.max_epochs

    # Build model
    model = KoeTTS(model_cfg, token_cfg)
    if is_main:
        n_params = model.estimate_params()
        print(f"Model: {n_params/1e6:.1f}M params")

    # Build optimizer (before DeepSpeed init — DS wraps it)
    optimizer, opt_type, muon_p, adam_p = build_optimizer(model, train_cfg)
    if is_main:
        print(f"Optimizer: {opt_type} | Muon: {sum(p.numel() for p in muon_p)/1e6:.1f}M, "
              f"AdamW: {sum(p.numel() for p in adam_p)/1e6:.1f}M")

    # Initialize DeepSpeed engine
    # DeepSpeed wraps model + optimizer + LR scheduler + AMP
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters(),
    )

    # Dataset
    text_tokenizer = CharTokenizer(token_cfg)
    dataset = TTSDataset(
        manifest_path=args.manifest,
        text_tokenizer=text_tokenizer,
        token_config=token_cfg,
        max_seq_len=model_cfg.max_seq_len,
        groups_per_frame=model_cfg.groups_per_frame,
        cfg_dropout=train_cfg.cfg_dropout,
    )
    if is_main:
        print(f"Dataset: {len(dataset)} samples")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=model_engine.train_micro_batch_size_per_gpu(),
        sampler=sampler,
        collate_fn=partial(collate_fn, pad_id=token_cfg.pad_id),
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Resume from DeepSpeed checkpoint
    start_step = 0
    start_epoch = 0
    if args.resume:
        _, client_state = model_engine.load_checkpoint(args.resume)
        if client_state:
            start_step = client_state.get("step", 0)
            start_epoch = client_state.get("epoch", 0)
        if is_main:
            print(f"Resumed from {args.resume} at step {start_step}")

    # Estimate total steps
    grad_accum = model_engine.gradient_accumulation_steps()
    steps_per_epoch = len(loader) // grad_accum
    if train_cfg.max_steps:
        total_steps = train_cfg.max_steps
    elif train_cfg.max_epochs:
        total_steps = train_cfg.max_epochs * steps_per_epoch
    else:
        total_steps = 50 * steps_per_epoch

    if is_main:
        effective_bs = model_engine.train_micro_batch_size_per_gpu() * world_size * grad_accum
        print(f"Training: {total_steps} steps ({steps_per_epoch} steps/epoch)")
        print(f"Effective batch size: {effective_bs}")
        print(f"DeepSpeed ZeRO stage: {model_engine.zero_optimization_stage()}")

    # W&B
    if is_main and not args.no_wandb:
        try:
            import wandb
            wandb.init(
                project=train_cfg.wandb_project,
                name=args.run_name or "exp_a",
                config={**vars(model_cfg), **vars(token_cfg), **vars(train_cfg),
                        "world_size": world_size, "deepspeed_zero": 2},
            )
        except ImportError:
            pass

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    step = start_step
    epoch = start_epoch
    t0 = time.time()
    accum_loss = 0.0

    while step < total_steps:
        sampler.set_epoch(epoch)

        for batch in loader:
            if step >= total_steps:
                break

            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward — DeepSpeed handles AMP via config (bf16.enabled=true)
            result = model_engine(
                token_ids=batch["token_ids"],
                segment_ids=batch["segment_ids"],
                group_pos_ids=batch["group_pos_ids"],
                labels=batch["token_ids"],
                loss_mask=batch["loss_mask"],
            )
            loss = result["loss"]

            # Backward — DeepSpeed handles gradient scaling, accumulation, allreduce
            model_engine.backward(loss)

            # Step — DeepSpeed handles grad clipping, optimizer step, zero_grad
            model_engine.step()

            accum_loss += loss.item()

            # LR schedule (manual — DeepSpeed scheduler is optional)
            lr_mult = cosine_warmup_schedule(
                step, train_cfg.warmup_steps, total_steps, train_cfg.lr_decay_ratio,
            )
            for pg in optimizer.param_groups:
                if "initial_lr" not in pg:
                    pg["initial_lr"] = pg["lr"]
                pg["lr"] = pg["initial_lr"] * lr_mult

            step += 1

            # Logging
            if is_main and step % train_cfg.log_every == 0:
                elapsed = time.time() - t0
                steps_per_sec = step / elapsed if elapsed > 0 else 0
                avg_loss = accum_loss / train_cfg.log_every
                accum_loss = 0.0

                tokens_per_sec = (
                    model_engine.train_micro_batch_size_per_gpu() * world_size
                    * batch["token_ids"].shape[1] * steps_per_sec
                )
                print(
                    f"step {step}/{total_steps} | "
                    f"loss={avg_loss:.4f} | lr={lr_mult:.4f} | "
                    f"{steps_per_sec:.1f} steps/s | "
                    f"{tokens_per_sec/1e3:.1f}K tok/s"
                )
                try:
                    import wandb
                    if wandb.run:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/lr_mult": lr_mult,
                            "train/tokens_per_sec": tokens_per_sec,
                            "train/step": step,
                        }, step=step)
                except (ImportError, Exception):
                    pass

            # DeepSpeed checkpoint (handles ZeRO state sharding automatically)
            if is_main and step % train_cfg.save_every == 0:
                model_engine.save_checkpoint(
                    str(output_dir),
                    tag=f"step_{step}",
                    client_state={"step": step, "epoch": epoch},
                )
                print(f"Saved DeepSpeed checkpoint: {output_dir}/step_{step}")

                # Keep only last N checkpoints
                ckpt_dirs = sorted(
                    [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
                    key=lambda p: p.stat().st_mtime,
                )
                for old in ckpt_dirs[:-train_cfg.keep_last_n]:
                    import shutil
                    shutil.rmtree(old)

        epoch += 1

    # Save final model weights (consolidated, not sharded — for inference)
    if is_main:
        raw_model = model_engine.module
        final_path = output_dir / "final.pt"
        torch.save({
            "step": step,
            "epoch": epoch,
            "model": raw_model.state_dict(),
            "config": {
                "model": vars(model_cfg),
                "token": vars(token_cfg),
            },
        }, final_path)
        print(f"Training complete. Final consolidated weights: {final_path}")

    deepspeed.comm.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train KoeTTS AR model")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to encoded manifest JSONL (from pre_encode.py)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="DeepSpeed checkpoint dir to resume from")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max training steps")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Override max training epochs")
    parser.add_argument("--run_name", type=str, default=None,
                        help="W&B run name")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")

    # DeepSpeed adds its own args (--deepspeed, --local_rank, etc.)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
