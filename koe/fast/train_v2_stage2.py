"""V2 Stage 2 decoder training — 25 Hz, code_dim=32, single GPU.

Trains the v2 codec pipeline:
  25Hz encoder (frozen/FT) → projection (128→code_dim) → Q2D2 → projection (code_dim→512)
  → HiFi-GAN or Vocos decoder → waveform

Key differences from train_stage2.py:
  - Learnable projection layers for code_dim reduction (128→32)
  - Default 25 Hz strides [4,4,4,5,3]
  - Support for Vocos iSTFT decoder
  - Default code_dim=32 with Q2D2 K=4

Usage:
    python -m koe.fast.train_v2_stage2 --data_dir /data/librilight \
        --output_dir ./checkpoints --stage1_ckpt /data/25hz_stage1.pt

    # With Vocos decoder:
    python -m koe.fast.train_v2_stage2 --data_dir /data/librilight \
        --output_dir ./checkpoints --stage1_ckpt /data/25hz_stage1.pt \
        --decoder vocos
"""

import argparse
import math
import os
import signal
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from koe.codec_impl import (
    JEPAEncoder,
    HiFiDecoderBlock,
    AntiAliasedActivation,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    MRSTFTLoss,
    discriminator_loss,
    generator_loss,
    feature_loss,
    set_requires_grad,
)

SAMPLE_RATE = 24000


# ═══════════════════════════════════════════════════════════
# Complex STFT Loss (phase supervision via real+imaginary L1)
# ═══════════════════════════════════════════════════════════

class MultiResolutionComplexSTFTLoss(nn.Module):
    """L1 loss on real+imaginary STFT components at multiple resolutions.

    Unlike magnitude-only STFT loss, this implicitly supervises phase
    by penalizing errors in both real and imaginary parts.
    """

    def __init__(self, n_ffts=(512, 1024, 2048), hop_div=4):
        super().__init__()
        self.n_ffts = n_ffts
        self.hop_div = hop_div

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_1d = pred.squeeze(1).float() if pred.dim() == 3 else pred.float()
        tgt_1d = target.squeeze(1).float() if target.dim() == 3 else target.float()
        loss = torch.tensor(0.0, device=pred.device, dtype=torch.float32)
        for n_fft in self.n_ffts:
            hop = n_fft // self.hop_div
            window = torch.hann_window(n_fft, device=pred.device)
            p_stft = torch.stft(pred_1d, n_fft, hop, window=window, return_complex=True)
            t_stft = torch.stft(tgt_1d, n_fft, hop, window=window, return_complex=True)
            # L1 on real and imaginary separately
            loss = loss + F.l1_loss(p_stft.real, t_stft.real) + F.l1_loss(p_stft.imag, t_stft.imag)
        return loss / len(self.n_ffts)


# ═══════════════════════════════════════════════════════════
# V2 Codec Model
# ═══════════════════════════════════════════════════════════

class V2Codec(nn.Module):
    """V2 codec: encoder → proj_down → quantizer → proj_up → decoder.

    The projection layers allow using a smaller code_dim (e.g., 32) for the
    quantizer while the encoder outputs 128 dims and the decoder expects 512.
    """

    def __init__(
        self,
        encoder: JEPAEncoder,
        code_dim: int = 32,
        decoder_type: str = "hifigan",
        channels: List[int] = (64, 128, 256, 384, 512, 512),
        strides: List[int] = (4, 4, 4, 5, 3),
        hifi_kernels: List[int] = (3, 7, 11, 15, 23, 32),
        use_weight_norm: bool = False,
        anti_alias: bool = False,
        vocos_hidden: int = 512,
        vocos_n_fft: int = 1024,
        vocos_hop: int = 256,
        vocos_blocks: int = 8,
    ):
        super().__init__()
        channels = list(channels)
        strides = list(strides)
        self.encoder = encoder
        self.enc_dim = encoder.code_dim  # 128
        self.code_dim = code_dim
        self.hop_length = encoder.hop_length
        self.decoder_type = decoder_type
        self._use_weight_norm = use_weight_norm

        _wn = nn.utils.weight_norm if use_weight_norm else (lambda x: x)

        # Projection: encoder dim → quantizer dim
        self.proj_down = nn.Conv1d(self.enc_dim, code_dim, 1)

        # Quantizer placeholder (set externally via swap)
        self.quantizer = None

        if decoder_type == "hifigan":
            # Projection: quantizer dim → decoder input
            self.proj_up = _wn(nn.Conv1d(code_dim, channels[-1], 1))

            # HiFi-GAN decoder (mirrors encoder strides in reverse)
            self.decoder = nn.ModuleList([
                HiFiDecoderBlock(
                    channels[i + 1], channels[i], strides[i],
                    list(hifi_kernels), use_gaatn=False,
                    use_weight_norm=use_weight_norm,
                    anti_alias=anti_alias,
                )
                for i in range(len(strides) - 1, -1, -1)
            ])
            self.output_conv = _wn(nn.Conv1d(channels[0], 1, kernel_size=7, padding=3))
            self.final_activation = nn.Tanh()
        elif decoder_type == "vocos":
            from koe.vocos_decoder import VocosDecoder, VocosDecoderConfig
            jepa_hop = math.prod(strides)
            self.vocos = VocosDecoder(VocosDecoderConfig(
                jepa_dim=code_dim,
                hidden_dim=vocos_hidden,
                n_fft=vocos_n_fft,
                hop_length=vocos_hop,
                sample_rate=SAMPLE_RATE,
                n_convnext_blocks=vocos_blocks,
                jepa_hop=jepa_hop,
            ))
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

        self._init_new_weights()

    def _init_new_weights(self):
        """Initialize only the new trainable layers (not encoder)."""
        for m in [self.proj_down, self.proj_up if hasattr(self, 'proj_up') else None]:
            if m is not None:
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if self.decoder_type == "hifigan":
            for m in [self.output_conv, *self.decoder]:
                self._init_module(m)

    @staticmethod
    def _init_module(m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Module):
            for subm in m.modules():
                if isinstance(subm, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                    nn.init.trunc_normal_(subm.weight, std=0.02)
                    if subm.bias is not None:
                        nn.init.zeros_(subm.bias)

    def encode(self, wav: torch.Tensor):
        """Encode waveform → projected features → quantized.

        wav: [B, 1, T_wav]
        Returns: z_q, z_proj, indices, aux_loss
        """
        z_e = self.encoder.encode(wav)          # [B, 128, T]
        z_proj = self.proj_down(z_e)            # [B, code_dim, T]
        z_q, indices, aux_loss = self.quantizer(z_proj)
        return z_q, z_proj, indices, aux_loss

    def decode(self, z_q: torch.Tensor, target_len: Optional[int] = None):
        """Decode quantized features → waveform.

        z_q: [B, code_dim, T]
        """
        if self.decoder_type == "hifigan":
            x = self.proj_up(z_q)               # [B, 512, T]
            for dec in self.decoder:
                x, _ = dec(x)
            wav = self.output_conv(x)            # [B, 1, T_wav]
            wav = self.final_activation(wav)
            return wav
        elif self.decoder_type == "vocos":
            wav = self.vocos(z_q, target_len=target_len)  # [B, T_wav] float32
            return wav.unsqueeze(1).to(z_q.dtype)  # [B, 1, T_wav] match model dtype

    def forward(self, wav: torch.Tensor):
        """Full forward pass.

        Returns: (wav_rec, indices, aux_loss, z_proj)
        """
        original_length = wav.shape[-1]
        z_q, z_proj, indices, aux_loss = self.encode(wav)
        rec = self.decode(z_q, target_len=original_length)

        if rec.shape[-1] > original_length:
            rec = rec[..., :original_length]
        elif rec.shape[-1] < original_length:
            rec = F.pad(rec, (0, original_length - rec.shape[-1]))

        return rec, indices, aux_loss, z_proj


# ═══════════════════════════════════════════════════════════
# LR Schedule
# ═══════════════════════════════════════════════════════════

def cosine_warmup(step: int, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress))


# ═══════════════════════════════════════════════════════════
# Audio Dataset (same as train_stage2.py)
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
                    wav = torch.from_numpy(data).unsqueeze(0)
                else:
                    wav = torch.from_numpy(data.T)
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
# Main training function
# ═══════════════════════════════════════════════════════════

def train(args):
    device = torch.device(f"cuda:0")
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # ── Build encoder ──
    strides = [int(s) for s in args.strides.split(",")]
    hop = math.prod(strides)
    channels = [64, 128, 256, 384, 512, 512]  # len = len(strides) + 1

    encoder = JEPAEncoder(
        sample_rate=SAMPLE_RATE,
        code_dim=128,  # encoder always outputs 128
        channels=channels,
        strides=strides,
        n_res_blocks=args.n_res_blocks,
        n_conformer=args.n_conformer,
        conformer_heads=args.conformer_heads,
        use_gaatn=True,
    )

    # Load Stage 1 encoder weights
    if args.stage1_ckpt and Path(args.stage1_ckpt).exists():
        ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
        enc_sd = ckpt.get("state_dict", ckpt.get("encoder", {}))
        if enc_sd:
            try:
                encoder.load_state_dict(enc_sd, strict=True)
                print(f"[train] Loaded Stage 1 encoder from {args.stage1_ckpt}")
            except RuntimeError:
                encoder.load_state_dict(enc_sd, strict=False)
                print(f"[train] Loaded Stage 1 encoder (partial) from {args.stage1_ckpt}")

    # ── Build V2 model ──
    model = V2Codec(
        encoder=encoder,
        code_dim=args.code_dim,
        decoder_type=args.decoder,
        channels=channels,
        strides=strides,
        use_weight_norm=args.weight_norm,
        anti_alias=args.anti_alias,
    )

    # ── Set quantizer ──
    if args.quantizer == "q2d2":
        from koe.fast.q2d2 import Q2D2Quantizer
        model.quantizer = Q2D2Quantizer(
            dim=args.code_dim,
            num_levels=args.q2d2_levels,
            grid_type=args.q2d2_grid,
            commitment_weight=args.commitment_weight,
        )
        print(f"[train] Q2D2 quantizer (dim={args.code_dim}, K={args.q2d2_levels}, grid={args.q2d2_grid}, commit={args.commitment_weight})")
    elif args.quantizer == "d4":
        from paper.scripts.lattice_quantizer import D4LatticeQuantizer
        model.quantizer = D4LatticeQuantizer(
            dim=args.code_dim,
            n_codes=args.d4_n_codes,
            commitment_weight=args.commitment_weight,
            use_tanh=args.d4_tanh,
        )
        print(f"[train] D4 lattice quantizer (dim={args.code_dim}, n_codes={args.d4_n_codes}, "
              f"commit={args.commitment_weight}, tanh={args.d4_tanh})")
    else:
        from koe.codec_impl import FiniteScalarQuantizer
        fsq_levels = [int(l) for l in args.fsq_levels.split(",")]
        model.quantizer = FiniteScalarQuantizer(
            levels=fsq_levels, dim=args.code_dim, normalized=True,
        )
        print(f"[train] FSQ quantizer (dim={args.code_dim}, levels={fsq_levels})")

    print(f"[train] V2 config: strides={strides}, hop={hop}, frame_rate={SAMPLE_RATE/hop:.1f} Hz, "
          f"code_dim={args.code_dim}, decoder={args.decoder}")

    # ── Discriminators ──
    mpd = MultiPeriodDiscriminator()
    msd = MultiScaleDiscriminator()

    # ── Resume ──
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
        start_step = ckpt.get("step", 0)
        wandb_run_id = ckpt.get("wandb_run_id", wandb_run_id)
        best_pesq = ckpt.get("metrics", {}).get("pesq", 0.0)
        print(f"[train] Resuming from step {start_step}")

    # ── Move to GPU ──
    dtype = torch.float32 if args.fp32 else torch.bfloat16
    model = model.to(device, dtype=dtype)
    mpd = mpd.to(device, dtype=dtype)
    msd = msd.to(device, dtype=dtype)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"[train] Model: {n_params:.1f}M total, {n_trainable:.1f}M trainable")

    # ── Encoder fine-tuning with gradient scaling ──
    enc_hooks = []
    if args.encoder_lr_scale != 1.0:
        for p in model.encoder.parameters():
            if p.requires_grad:
                enc_hooks.append(
                    p.register_hook(lambda g, s=args.encoder_lr_scale: g * s)
                )
        print(f"[train] Encoder LR scale: {args.encoder_lr_scale}x")

    if args.freeze_encoder:
        model.encoder.requires_grad_(False)
        model.encoder.eval()
        print("[train] Encoder FROZEN")

    # ── Teacher encoder for distillation ──
    teacher_encoder = None
    if args.teacher_ckpt:
        import copy
        t_ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
        t_sd = t_ckpt.get("state_dict", t_ckpt)
        enc_sd = {k.replace("encoder.", ""): v for k, v in t_sd.items() if k.startswith("encoder.")}
        teacher_encoder = copy.deepcopy(model.encoder)
        teacher_encoder.load_state_dict(enc_sd, strict=False)
        teacher_encoder = teacher_encoder.to(device, dtype=dtype)
        teacher_encoder.eval()
        teacher_encoder.requires_grad_(False)
        print(f"[train] Teacher distillation: λ={args.lambda_distill}, start={args.distill_start_step}, "
              f"teacher keys={len(enc_sd)}")

    # ── Optimizers ──
    # Separate quantizer params into own group when lr_quantizer is set,
    # so they get their own LR and gradient clip (critical for D4 affine).
    quantizer_param_ids = {id(p) for p in model.quantizer.parameters()}
    main_params = [p for p in model.parameters() if p.requires_grad and id(p) not in quantizer_param_ids]
    quantizer_params = [p for p in model.quantizer.parameters() if p.requires_grad]

    if args.lr_quantizer is not None and quantizer_params:
        gen_opt = torch.optim.AdamW([
            {"params": main_params, "lr": args.lr_gen},
            {"params": quantizer_params, "lr": args.lr_quantizer},
        ], betas=(0.8, 0.99), weight_decay=1e-3, fused=True)
        print(f"[train] Separate quantizer LR: {args.lr_quantizer} (main: {args.lr_gen})")
    else:
        gen_opt = torch.optim.AdamW(
            main_params + quantizer_params, lr=args.lr_gen,
            betas=(0.8, 0.99), weight_decay=1e-3, fused=True,
        )
    gen_params = main_params + quantizer_params
    disc_params = list(mpd.parameters()) + list(msd.parameters())
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

    # Complex STFT loss (supervises phase via real+imaginary L1)
    complex_stft_fn = None
    if args.lambda_complex_stft > 0:
        complex_stft_fn = MultiResolutionComplexSTFTLoss().to(device)
        print(f"[train] Complex STFT loss (lambda={args.lambda_complex_stft})")

    perceptual_loss_fn = None
    if args.lambda_perceptual > 0:
        from koe.fast.losses import WavLMPerceptualLoss
        perceptual_loss_fn = WavLMPerceptualLoss(device=str(device))
        print(f"[train] WavLM perceptual loss (lambda={args.lambda_perceptual})")

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

    # ── Output dir ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── W&B ──
    if not args.no_wandb:
        try:
            import wandb
            wandb_kwargs = dict(
                project="koe-v2",
                name=args.run_name or "v2_stage2",
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
    step = start_step

    def _on_sigterm(signum, frame):
        print(f"\n[train] SIGTERM at step {step}. Saving...")
        sigterm_received[0] = True

    signal.signal(signal.SIGTERM, _on_sigterm)

    # ── Collapse detection ──
    collapse_window = 500
    disc_loss_history = deque(maxlen=collapse_window)
    gen_loss_history = deque(maxlen=collapse_window)
    lambda_gan = args.lambda_gan

    # ── Training loop ──
    t0 = time.time()
    t_last_log = t0
    accum = {
        "g_loss": 0.0, "d_loss": 0.0, "stft": 0.0, "l1": 0.0,
        "adv": 0.0, "feat": 0.0, "aux": 0.0, "perceptual": 0.0,
        "complex_stft": 0.0,
    }
    nan_count = 0

    print(f"[train] Starting from step {step}, target {args.total_steps} steps")

    while step < args.total_steps:
        for batch in loader:
            if step >= args.total_steps or sigterm_received[0]:
                break

            wav = batch.to(device, dtype=dtype, non_blocking=True)

            # ── Generator forward ──
            wav_rec, indices, aux_loss, z_proj = model(wav)

            # Cast to float32 for numerically stable loss computation
            wav_f32 = wav.float()
            rec_f32 = wav_rec.float()

            stft = stft_loss_fn(rec_f32, wav_f32)
            l1 = F.l1_loss(rec_f32, wav_f32)

            # GAN losses — skip disc forward entirely when disc will never
            # activate (disc_warmup > total_steps). An untrained disc produces
            # random feature maps whose gradients can destabilize the generator.
            use_disc = args.disc_warmup <= args.total_steps
            if use_disc:
                set_requires_grad(mpd, False)
                set_requires_grad(msd, False)
                mpd_out = mpd(wav, wav_rec)
                msd_out = msd(wav, wav_rec)
                g_adv = generator_loss(mpd_out[1]) + generator_loss(msd_out[1])
                g_feat = feature_loss(mpd_out[2], mpd_out[3]) + feature_loss(msd_out[2], msd_out[3])
                g_loss = (
                    args.lambda_stft * stft + l1
                    + lambda_gan * (g_adv + g_feat)
                    + aux_loss.float()
                )
            else:
                g_adv = torch.tensor(0.0)
                g_feat = torch.tensor(0.0)
                g_loss = args.lambda_stft * stft + l1 + aux_loss.float()

            # Optional complex STFT loss (phase supervision)
            complex_stft_val = 0.0
            if complex_stft_fn is not None:
                c_loss = complex_stft_fn(rec_f32, wav_f32)
                g_loss = g_loss + args.lambda_complex_stft * c_loss
                complex_stft_val = c_loss.item()

            # Optional perceptual loss
            perceptual_val = 0.0
            if perceptual_loss_fn is not None:
                p_loss = perceptual_loss_fn(rec_f32, wav_f32)
                g_loss = g_loss + args.lambda_perceptual * p_loss
                perceptual_val = p_loss.item()

            # Optional teacher distillation (encoder-level cosine similarity)
            distill_val = 0.0
            if teacher_encoder is not None and step >= args.distill_start_step:
                with torch.no_grad():
                    teacher_z = teacher_encoder.encode(wav)
                student_z = model.encoder.encode(wav)
                min_t = min(student_z.shape[-1], teacher_z.shape[-1])
                dl = 1.0 - F.cosine_similarity(
                    student_z[:, :, :min_t], teacher_z[:, :, :min_t].detach(), dim=1
                ).mean()
                g_loss = g_loss + args.lambda_distill * dl
                distill_val = dl.item()

            # NaN check — skip bad batches, don't die
            g_loss_val = g_loss.item()
            if math.isnan(g_loss_val) or math.isinf(g_loss_val) or g_loss_val > 1e6:
                nan_count += 1
                if nan_count % 10 == 1:
                    print(f"[train] Bad gen loss {g_loss_val} at step {step} (total NaN={nan_count})")
                if nan_count >= 50:
                    print(f"[train] {nan_count} NaN losses, stopping")
                    sigterm_received[0] = True
                    break
                continue

            gen_opt.zero_grad()
            g_loss.backward()

            # Check for NaN in gradients (forward loss can be valid but
            # backward can still produce NaN, which silently corrupts all
            # parameters on the next optimizer step).
            grad_has_nan = False
            for p in gen_params:
                if p.grad is not None and p.grad.isnan().any():
                    grad_has_nan = True
                    break
            if grad_has_nan:
                nan_count += 1
                if nan_count % 10 == 1:
                    print(f"[train] NaN gradient at step {step} (total NaN={nan_count})")
                if nan_count >= 50:
                    print(f"[train] {nan_count} NaN gradient events, stopping")
                    sigterm_received[0] = True
                    break
                gen_opt.zero_grad(set_to_none=True)
                continue

            # Decay NaN counter only after a fully clean iteration
            # (both forward loss and backward gradients were valid)
            nan_count = max(0, nan_count - 1)

            if args.lr_quantizer is not None and quantizer_params:
                torch.nn.utils.clip_grad_norm_(main_params, 1.0)
                torch.nn.utils.clip_grad_norm_(
                    quantizer_params, args.quantizer_clip)
            else:
                torch.nn.utils.clip_grad_norm_(gen_params, 1.0)
            gen_opt.step()

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
                disc_opt.zero_grad()
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(disc_params, 1.0)
                disc_opt.step()
                set_requires_grad(mpd, False)
                set_requires_grad(msd, False)
                d_loss_val = d_loss.item()
            else:
                d_loss_val = 0.0

            step += 1

            # Accumulate
            accum["g_loss"] += g_loss_val
            accum["d_loss"] += d_loss_val
            accum["stft"] += stft.item()
            accum["l1"] += l1.item()
            accum["adv"] += g_adv.item()
            accum["feat"] += g_feat.item()
            accum["aux"] += aux_loss.item()
            accum["perceptual"] += perceptual_val
            accum["complex_stft"] += complex_stft_val

            # Collapse detection
            disc_loss_history.append(d_loss_val)
            gen_loss_history.append(g_loss_val)
            if len(disc_loss_history) == collapse_window and step > args.disc_warmup + collapse_window:
                avg_d = sum(disc_loss_history) / collapse_window
                avg_g = sum(gen_loss_history) / collapse_window
                if avg_d < 0.1:
                    print(f"[train] Disc collapse (avg_d={avg_d:.3f}). Increasing disc_lr 2x.")
                    for pg in disc_opt.param_groups:
                        pg["lr"] *= 2.0
                    disc_loss_history.clear()
                if avg_g > 10.0:
                    print(f"[train] Gen instability (avg_g={avg_g:.3f}). Reducing lambda_gan 50%.")
                    lambda_gan *= 0.5
                    gen_loss_history.clear()

            # LR schedule
            lr_mult = cosine_warmup(step, args.warmup_steps, args.total_steps)
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
                    "train/gpu_mem_gb": torch.cuda.memory_allocated() / 1e9,
                }
                if perceptual_loss_fn is not None:
                    log_dict["train/perceptual_loss"] = accum["perceptual"] / n

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
                metrics = _run_eval(model, eval_wavs, device, step, dtype=dtype)
                if metrics and metrics.get("pesq", 0) > best_pesq:
                    best_pesq = metrics["pesq"]
                    print(f"[train] New best PESQ: {best_pesq:.4f}")
                torch.cuda.empty_cache()

            # ── Checkpoint ──
            if step % args.save_every == 0:
                _save_checkpoint(
                    model, mpd, msd, gen_opt, disc_opt,
                    step, wandb_run_id, best_pesq, args,
                    output_dir,
                )

        if sigterm_received[0] or step >= args.total_steps:
            break

    # Cleanup
    for h in enc_hooks:
        h.remove()

    _save_checkpoint(
        model, mpd, msd, gen_opt, disc_opt,
        step, wandb_run_id, best_pesq, args,
        output_dir,
    )

    peak = torch.cuda.max_memory_allocated() / 1e9
    elapsed = time.time() - t0
    print(f"\n[train] Done. step={step}, elapsed={elapsed:.0f}s, peak_vram={peak:.1f}GB")


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

def _find_resume_checkpoint(args) -> Optional[str]:
    if args.resume_from and Path(args.resume_from).exists():
        return args.resume_from
    latest = Path(args.output_dir) / "v2_latest.pt"
    if latest.exists():
        return str(latest)
    return None


def _compute_mel(wav_np: "np.ndarray", sr: int = 24000, n_mels: int = 80) -> "np.ndarray":
    """Compute log-mel spectrogram for visualization."""
    import numpy as np
    n_fft = 1024
    hop = 256
    # STFT
    from numpy.lib.stride_tricks import sliding_window_view
    pad = n_fft // 2
    x = np.pad(wav_np, (pad, pad), mode="reflect")
    window = np.hanning(n_fft + 1)[:n_fft]
    frames = sliding_window_view(x, n_fft)[::hop] * window
    S = np.abs(np.fft.rfft(frames, n_fft))
    # Mel filterbank
    fmin, fmax = 0, sr // 2
    mel_lo = 2595 * np.log10(1 + fmin / 700)
    mel_hi = 2595 * np.log10(1 + fmax / 700)
    mel_pts = 700 * (10 ** (np.linspace(mel_lo, mel_hi, n_mels + 2) / 2595) - 1)
    bins = np.floor((n_fft + 1) * mel_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m, k] = (k - bins[m]) / max(bins[m + 1] - bins[m], 1)
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m, k] = (bins[m + 2] - k) / max(bins[m + 2] - bins[m + 1], 1)
    mel = fb @ S.T
    return np.log(np.clip(mel, 1e-5, None))


def _per_freq_snr(orig: "np.ndarray", rec: "np.ndarray", sr: int, n_fft: int = 1024) -> "np.ndarray":
    """Per-frequency-bin SNR in dB."""
    import numpy as np
    hop = n_fft // 4
    window = np.hanning(n_fft + 1)[:n_fft]
    pad = n_fft // 2
    o = np.pad(orig, (pad, pad), mode="reflect")
    r = np.pad(rec, (pad, pad), mode="reflect")
    from numpy.lib.stride_tricks import sliding_window_view
    So = np.fft.rfft(sliding_window_view(o, n_fft)[::hop] * window, n_fft)
    Sr = np.fft.rfft(sliding_window_view(r, n_fft)[::hop] * window, n_fft)
    sig_power = (np.abs(So) ** 2).mean(axis=0)
    noise_power = (np.abs(So - Sr) ** 2).mean(axis=0)
    return 10 * np.log10(sig_power / np.clip(noise_power, 1e-10, None))


def _phase_coherence(orig: "np.ndarray", rec: "np.ndarray", n_fft: int = 1024) -> float:
    """Mean phase coherence (cosine similarity of complex STFT)."""
    import numpy as np
    hop = n_fft // 4
    window = np.hanning(n_fft + 1)[:n_fft]
    pad = n_fft // 2
    o = np.pad(orig, (pad, pad), mode="reflect")
    r = np.pad(rec, (pad, pad), mode="reflect")
    from numpy.lib.stride_tricks import sliding_window_view
    So = np.fft.rfft(sliding_window_view(o, n_fft)[::hop] * window, n_fft)
    Sr = np.fft.rfft(sliding_window_view(r, n_fft)[::hop] * window, n_fft)
    # Phase coherence: |<So, Sr>| / (|So| * |Sr|)
    num = np.abs((So.conj() * Sr).sum(axis=0))
    den = np.sqrt((np.abs(So) ** 2).sum(axis=0) * (np.abs(Sr) ** 2).sum(axis=0))
    coherence = num / np.clip(den, 1e-10, None)
    return float(coherence.mean())


def _run_eval(model, eval_wavs, device, step, dtype=torch.bfloat16) -> Optional[Dict[str, float]]:
    import numpy as np
    from koe.eval_codec import compute_pesq, compute_stoi, mel_spectral_distance

    model.eval()
    stois, pesqs, mels = [], [], []
    all_indices = []
    all_z_proj = []
    all_snr_curves = []
    phase_coherences = []

    for wav_t in eval_wavs:
        wav_in = wav_t.to(device, dtype=dtype)
        with torch.no_grad():
            rec, indices, _, z_proj = model(wav_in)
        orig = wav_t[0, 0].numpy()
        rec_np = rec[0, 0].float().cpu().numpy()
        n = min(len(orig), len(rec_np))
        orig, rec_np = orig[:n], rec_np[:n]
        stois.append(compute_stoi(orig, rec_np, SAMPLE_RATE))
        pesqs.append(compute_pesq(orig, rec_np, SAMPLE_RATE))
        mels.append(mel_spectral_distance(orig, rec_np, SAMPLE_RATE))
        all_indices.append(indices.cpu())
        all_z_proj.append(z_proj.float().cpu())
        all_snr_curves.append(_per_freq_snr(orig, rec_np, SAMPLE_RATE))
        phase_coherences.append(_phase_coherence(orig, rec_np))

    pesqs_clean = [p for p in pesqs if not math.isnan(p)]
    metrics = {
        "stoi": float(np.mean(stois)),
        "pesq": float(np.mean(pesqs_clean)) if pesqs_clean else 0.0,
        "mel_distance": float(np.mean(mels)),
        "phase_coherence": float(np.mean(phase_coherences)),
    }

    # STE displacement: measure quantization error in quantizer's internal space
    ste_disp = 0.0
    try:
        with torch.no_grad():
            wav_in = eval_wavs[0].to(device, dtype=dtype)
            z_e = model.encoder.encode(wav_in)
            z_p = model.proj_down(z_e)
            z_q, _, _ = model.quantizer(z_p)
            ste_disp = (z_p - z_q).pow(2).mean().sqrt().item()
        metrics["ste_displacement"] = ste_disp
    except Exception:
        pass

    # Feature distribution stats
    z_cat = torch.cat(all_z_proj, dim=2)  # [1, D, T_total]
    z_flat = z_cat.squeeze(0).T  # [T, D]
    feat_mean = z_flat.mean().item()
    feat_std = z_flat.std().item()
    feat_min = z_flat.min().item()
    feat_max = z_flat.max().item()
    # Kurtosis per dim
    z_centered = z_flat - z_flat.mean(0)
    m4 = (z_centered ** 4).mean(0)
    m2 = (z_centered ** 2).mean(0)
    kurtosis = (m4 / m2.clamp(min=1e-8) ** 2 - 3.0).mean().item()
    metrics["feat_mean"] = feat_mean
    metrics["feat_std"] = feat_std
    metrics["feat_kurtosis"] = kurtosis

    print(f"[eval] step {step} | PESQ={metrics['pesq']:.4f} "
          f"STOI={metrics['stoi']:.4f} mel={metrics['mel_distance']:.4f} "
          f"phase={metrics['phase_coherence']:.3f} ste={ste_disp:.4f}")

    try:
        import wandb
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if wandb.run:
            log_dict = {f"eval/{k}": v for k, v in metrics.items()}

            if step % 5000 == 0:
                for i in range(min(3, len(eval_wavs))):
                    orig_np = eval_wavs[i][0, 0].numpy()
                    wav_in = eval_wavs[i].to(device, dtype=dtype)
                    with torch.no_grad():
                        rec_i, _, _, _ = model(wav_in)
                    rec_np = rec_i[0, 0].float().cpu().numpy()
                    n = min(len(orig_np), len(rec_np))
                    orig_clip, rec_clip = orig_np[:n], rec_np[:n]

                    log_dict[f"audio/orig_{i}"] = wandb.Audio(
                        orig_clip, sample_rate=SAMPLE_RATE, caption=f"original_{i}")
                    log_dict[f"audio/rec_{i}"] = wandb.Audio(
                        rec_clip, sample_rate=SAMPLE_RATE, caption=f"reconstructed_{i}")

                    mel_orig = _compute_mel(orig_clip, SAMPLE_RATE)
                    mel_rec = _compute_mel(rec_clip, SAMPLE_RATE)
                    fig, axes = plt.subplots(2, 1, figsize=(10, 4))
                    axes[0].imshow(mel_orig, aspect="auto", origin="lower")
                    axes[0].set_title("Original")
                    axes[0].set_ylabel("Mel bin")
                    axes[1].imshow(mel_rec, aspect="auto", origin="lower")
                    axes[1].set_title("Reconstructed")
                    axes[1].set_ylabel("Mel bin")
                    axes[1].set_xlabel("Frame")
                    plt.tight_layout()
                    log_dict[f"mel/sample_{i}"] = wandb.Image(fig, caption=f"mel_{i}")
                    plt.close(fig)

                # Per-frequency SNR
                mean_snr = np.stack(all_snr_curves).mean(axis=0)
                freqs = np.linspace(0, SAMPLE_RATE / 2, len(mean_snr))
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(freqs, mean_snr)
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("SNR (dB)")
                ax.set_title("Per-frequency SNR")
                ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)
                # Band averages
                bands = [(0, 300, "sub-bass"), (300, 2000, "mid"),
                         (2000, 6000, "presence"), (6000, 12000, "brilliance")]
                for flo, fhi, name in bands:
                    mask = (freqs >= flo) & (freqs < fhi)
                    if mask.any():
                        band_snr = float(mean_snr[mask].mean())
                        log_dict[f"snr/{name}"] = band_snr
                plt.tight_layout()
                log_dict["snr/per_freq"] = wandb.Image(fig, caption="per_freq_snr")
                plt.close(fig)

                # Codebook utilization histogram
                idx_all = torch.cat(all_indices, dim=1).reshape(-1)
                codebook_size = int(idx_all.max().item()) + 1
                counts = torch.bincount(idx_all, minlength=codebook_size).float()
                n_used = (counts > 0).sum().item()
                probs = counts / counts.sum()
                probs_nz = probs[probs > 0]
                entropy = -(probs_nz * probs_nz.log()).sum().item()
                max_entropy = math.log(codebook_size) if codebook_size > 1 else 1.0
                utilization = entropy / max_entropy

                fig, ax = plt.subplots(figsize=(8, 3))
                ax.bar(range(codebook_size), counts.numpy(), width=1.0)
                ax.set_xlabel("Code index")
                ax.set_ylabel("Count")
                ax.set_title(f"Codebook: {n_used}/{codebook_size} used, "
                             f"ent={entropy:.2f}/{max_entropy:.2f} ({utilization*100:.0f}%)")
                plt.tight_layout()
                log_dict["codebook/histogram"] = wandb.Image(fig, caption="codebook_usage")
                plt.close(fig)
                log_dict["codebook/n_used"] = n_used
                log_dict["codebook/utilization"] = utilization

                # Feature distribution histogram
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.hist(z_flat.reshape(-1).numpy(), bins=100, density=True, alpha=0.7)
                ax.set_xlabel("Value")
                ax.set_ylabel("Density")
                ax.set_title(f"Feature dist: mean={feat_mean:.3f} std={feat_std:.3f} "
                             f"kurt={kurtosis:.2f} range=[{feat_min:.1f},{feat_max:.1f}]")
                plt.tight_layout()
                log_dict["features/histogram"] = wandb.Image(fig, caption="feat_dist")
                plt.close(fig)

                # Affine decomposition (Q2D2: rotation angle + scale per pair)
                if hasattr(model.quantizer, "affine"):
                    aff = model.quantizer.affine.detach().float().cpu()
                    if aff.shape[-1] == 2:  # Q2D2: [P, 2, 2]
                        angles = []
                        scales = []
                        for p in range(aff.shape[0]):
                            U, S, Vh = torch.linalg.svd(aff[p])
                            angles.append(math.atan2(U[1, 0].item(), U[0, 0].item()))
                            scales.append(S.tolist())
                        angles_deg = [a * 180 / math.pi for a in angles]
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
                        ax1.bar(range(len(angles_deg)), angles_deg)
                        ax1.set_xlabel("Pair index")
                        ax1.set_ylabel("Rotation (deg)")
                        ax1.set_title("Learned rotation per pair")
                        s_arr = np.array(scales)
                        ax2.bar(range(len(scales)), s_arr[:, 0], alpha=0.7, label="s1")
                        ax2.bar(range(len(scales)), s_arr[:, 1], alpha=0.7, label="s2")
                        ax2.set_xlabel("Pair index")
                        ax2.set_ylabel("Singular value")
                        ax2.set_title("Learned scale per pair")
                        ax2.legend()
                        plt.tight_layout()
                        log_dict["affine/decomposition"] = wandb.Image(fig, caption="affine")
                        plt.close(fig)
                        log_dict["affine/mean_rotation_deg"] = float(np.mean(np.abs(angles_deg)))
                        log_dict["affine/mean_scale"] = float(s_arr.mean())
                        log_dict["affine/cond_max"] = float(s_arr[:, 0].max() / s_arr[:, 1].min())
                    elif aff.shape[-1] == 4:  # D4: [G, 4, 4]
                        conds = [torch.linalg.cond(aff[g]).item() for g in range(aff.shape[0])]
                        dets = [torch.linalg.det(aff[g]).item() for g in range(aff.shape[0])]
                        drift = (aff - torch.eye(4).unsqueeze(0)).norm().item()
                        log_dict["affine/cond_max"] = max(conds)
                        log_dict["affine/cond_mean"] = float(np.mean(conds))
                        log_dict["affine/det_mean"] = float(np.mean(dets))
                        log_dict["affine/drift_from_identity"] = drift

            wandb.log(log_dict, step=step)
    except (ImportError, Exception):
        pass

    model.train()
    if hasattr(model, '_freeze_encoder') or (hasattr(model, 'encoder') and not any(p.requires_grad for p in model.encoder.parameters())):
        model.encoder.eval()
    return metrics


def _save_checkpoint(
    model, mpd, msd, gen_opt, disc_opt,
    step, wandb_run_id, best_pesq, args,
    output_dir,
):
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
        "config": {
            "strides": [int(s) for s in args.strides.split(",")],
            "code_dim": args.code_dim,
            "decoder": args.decoder,
            "quantizer": args.quantizer,
        },
        "wandb_run_id": wandb_run_id,
        "metrics": {"pesq": best_pesq},
    }

    for fname in [f"v2_step{step:07d}.pt", "v2_latest.pt"]:
        torch.save(ckpt, output_dir / fname)

    # Keep last 3 step checkpoints
    step_ckpts = sorted(output_dir.glob("v2_step*.pt"))
    for old in step_ckpts[:-3]:
        old.unlink()

    print(f"[train] Saved checkpoint at step {step}")

    try:
        import wandb
        if wandb.run:
            wandb.log({"checkpoint/step": step, "checkpoint/best_pesq": best_pesq}, step=step)
    except (ImportError, Exception):
        pass


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="V2 Stage 2 decoder training (25 Hz, code_dim=32)")

    # Required
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--stage1_ckpt", type=str, default=None)

    # Resume
    parser.add_argument("--resume_from", type=str, default=None)

    # V2 architecture
    parser.add_argument("--code_dim", type=int, default=32,
                        help="Quantizer code dimension (default 32)")
    parser.add_argument("--decoder", type=str, default="hifigan",
                        choices=["hifigan", "vocos"])
    parser.add_argument("--strides", type=str, default="4,4,4,5,3",
                        help="Encoder strides (default 25 Hz)")
    parser.add_argument("--n_res_blocks", type=int, default=8)
    parser.add_argument("--n_conformer", type=int, default=8)
    parser.add_argument("--conformer_heads", type=int, default=16)

    # Quantizer
    parser.add_argument("--quantizer", type=str, default="q2d2",
                        choices=["fsq", "q2d2", "d4"])
    parser.add_argument("--q2d2_levels", type=int, default=4)
    parser.add_argument("--q2d2_grid", type=str, default="rhombic",
                        choices=["square", "rhombic"])
    parser.add_argument("--fsq_levels", type=str, default="4,4,4,4")
    parser.add_argument("--d4_n_codes", type=int, default=256,
                        help="D4 lattice codes per group (256=8bits, 64=6bits)")
    parser.add_argument("--d4_tanh", action="store_true",
                        help="Use tanh bounding for D4 input (like Q2D2)")

    # Training
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr_gen", type=float, default=1.5e-4)
    parser.add_argument("--lr_disc", type=float, default=7.5e-5)
    parser.add_argument("--lr_quantizer", type=float, default=None,
                        help="Separate LR for quantizer params (None=use lr_gen)")
    parser.add_argument("--quantizer_clip", type=float, default=1.0,
                        help="Gradient clip norm for quantizer params")
    parser.add_argument("--encoder_lr_scale", type=float, default=0.1)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_seconds", type=float, default=15.0)
    parser.add_argument("--num_workers", type=int, default=2)

    # Loss weights
    parser.add_argument("--lambda_stft", type=float, default=2.0)
    parser.add_argument("--lambda_gan", type=float, default=1.0)
    parser.add_argument("--lambda_perceptual", type=float, default=0.0)
    parser.add_argument("--lambda_complex_stft", type=float, default=0.0,
                        help="Complex STFT loss weight (phase supervision, 0=disabled)")
    parser.add_argument("--commitment_weight", type=float, default=0.25,
                        help="Q2D2 commitment loss weight (0=disabled)")
    parser.add_argument("--weight_norm", action="store_true",
                        help="Apply weight normalization to decoder convolutions")
    parser.add_argument("--anti_alias", action="store_true",
                        help="Anti-aliased SnakeBeta activation (BigVGAN-style)")

    # Teacher distillation
    parser.add_argument("--teacher_ckpt", type=str, default=None,
                        help="v9 stage2 checkpoint for encoder distillation")
    parser.add_argument("--lambda_distill", type=float, default=0.1,
                        help="Distillation loss weight")
    parser.add_argument("--distill_start_step", type=int, default=5000,
                        help="Step to begin distillation")

    # GAN schedule
    parser.add_argument("--disc_warmup", type=int, default=5000)

    # Eval / logging
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--eval_samples", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=10)

    # W&B
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--fp32", action="store_true",
                        help="Use float32 instead of bfloat16 (slower, more stable)")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
