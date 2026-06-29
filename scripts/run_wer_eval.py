"""Clean WER eval across all presented codec models.

Verified against source 2026-05-14:
- V2 models (cd64 Q2D2, cd64 FSQ, cd32 Q2D2): koe.fast.train_v2_stage2.V2Codec
  + quantizer chosen from checkpoint config ('q2d2' or 'fsq'). NOT hardcoded.
- OG v9 models: koe.codec_impl.WaveformJEPAFSQVAE.
- Eval set: 50 LibriLight files, seed 42, <=10s -- identical to run_evals.py Protocol B.
- WER: Whisper base.en (matches the existing 17.85% data point so numbers are comparable),
  jiwer word error rate over all 50 samples.

Usage:
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_wer_eval.py \
      --data_dir /local_data/data/librilight/librilight_9k \
      --output /local_data/eval_results/wer_clean_0514.json
"""
import argparse, glob, json, math, os, random, sys, tempfile
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.expanduser("~/koe"))
SAMPLE_RATE = 24000


def load_eval_samples(data_dir, n_samples=50, max_seconds=10.0, seed=42):
    """EXACT copy of the Protocol-B sampler from run_evals.py / run_v9_evals.py."""
    files = []
    for ext in ["*.flac", "*.wav"]:
        files.extend(sorted(glob.glob(os.path.join(data_dir, "**", ext), recursive=True)))
    rng = random.Random(seed)
    rng.shuffle(files)
    wavs = []
    max_samples = int(max_seconds * SAMPLE_RATE)
    for f in files:
        if len(wavs) >= n_samples:
            break
        try:
            data, sr = sf.read(f, dtype="float32")
            wav = data if data.ndim == 1 else data.mean(axis=1)
            if sr != SAMPLE_RATE:
                import torchaudio
                wav = torchaudio.functional.resample(
                    torch.from_numpy(wav).unsqueeze(0), sr, SAMPLE_RATE
                ).squeeze(0).numpy()
            if len(wav) < SAMPLE_RATE:
                continue
            wavs.append(wav[:max_samples])
        except Exception:
            continue
    print(f"Loaded {len(wavs)} eval samples")
    return wavs


def load_v2_model(stage2_ckpt, device):
    """V2Codec. Quantizer type read FROM the checkpoint config -- not hardcoded."""
    from koe.codec_impl import JEPAEncoder, FiniteScalarQuantizer
    from koe.fast.train_v2_stage2 import V2Codec
    from koe.fast.q2d2 import Q2D2Quantizer

    ckpt = torch.load(stage2_ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    strides = cfg["strides"]
    code_dim = cfg["code_dim"]
    qtype = cfg["quantizer"]

    encoder = JEPAEncoder(
        sample_rate=SAMPLE_RATE, code_dim=128,
        channels=[64, 128, 256, 384, 512, 512],
        strides=strides, n_res_blocks=8,
        n_conformer=8, conformer_heads=16, use_gaatn=True,
    )
    model = V2Codec(encoder=encoder, code_dim=code_dim, strides=strides)

    if qtype == "q2d2":
        model.quantizer = Q2D2Quantizer(
            dim=code_dim, num_levels=4, grid_type="rhombic", commitment_weight=0.25,
        )
    elif qtype == "fsq":
        # cd64 FSQ checkpoint verified: 4 boundary modules, bounds shape (4,) -> levels=[4,4,4,4]
        model.quantizer = FiniteScalarQuantizer(
            levels=[4, 4, 4, 4], dim=code_dim, normalized=True,
        )
    else:
        raise ValueError(f"unknown quantizer in config: {qtype}")

    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    # sanity: a correct load should have very few missing/unexpected keys
    print(f"  V2Codec loaded: cd={code_dim}, q={qtype}, strides={strides}, "
          f"step={ckpt.get('step','?')}, missing={len(missing)}, unexpected={len(unexpected)}")
    if len(missing) > 5 or len(unexpected) > 5:
        print(f"    WARNING: missing[:5]={missing[:5]} unexpected[:5]={unexpected[:5]}")
    model = model.to(device, dtype=torch.bfloat16).eval()
    return model, math.prod(strides)


def load_v9_model(stage2_ckpt, device):
    """WaveformJEPAFSQVAE (OG v9). Matches run_v9_evals.load_v9_model exactly."""
    from koe.codec_impl import WaveformJEPAFSQVAE
    ckpt = torch.load(stage2_ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    strides = cfg.get("strides", [4, 4, 4, 5, 6])
    fsq_levels = cfg.get("fsq_levels", [8, 8, 8, 8])
    n_res_blocks = cfg.get("n_res_blocks", 8)
    model = WaveformJEPAFSQVAE(
        sample_rate=SAMPLE_RATE, code_dim=128,
        channels=[64, 128, 256, 384, 512, 512],
        strides=strides, n_res_blocks=n_res_blocks,
        n_conformer=8, conformer_heads=16,
        fsq_levels=fsq_levels, hifi_kernels=[3, 7, 11, 15, 23, 32],
    )
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"  WaveformJEPAFSQVAE loaded: strides={strides}, fsq={fsq_levels}, "
          f"step={ckpt.get('step','?')}, missing={len(missing)}, unexpected={len(unexpected)}")
    if len(missing) > 5 or len(unexpected) > 5:
        print(f"    WARNING: missing[:5]={missing[:5]} unexpected[:5]={unexpected[:5]}")
    model = model.to(device, dtype=torch.bfloat16).eval()
    return model, math.prod(strides)


def reconstruct(model, hop, wavs, device):
    recs = []
    with torch.no_grad():
        for wav_np in wavs:
            wav = torch.from_numpy(wav_np)
            orig_len = wav.shape[0]
            rem = orig_len % hop
            if rem:
                wav = torch.nn.functional.pad(wav, (0, hop - rem))
            wav_in = wav.unsqueeze(0).unsqueeze(0).to(device, dtype=torch.bfloat16)
            out = model(wav_in)
            rec = out[0] if isinstance(out, (tuple, list)) else out
            recs.append(rec[0, 0, :orig_len].float().cpu().numpy())
    return recs


def compute_wer(orig_list, rec_list, device):
    """Whisper base.en transcription + jiwer WER. Same ASR as the existing 17.85% number."""
    import whisper
    from jiwer import wer as jiwer_wer
    wh = whisper.load_model("base.en", device=device)
    orig_texts, rec_texts = [], []
    for i, (orig, rec) in enumerate(zip(orig_list, rec_list)):
        for audio, bucket in [(orig, orig_texts), (rec, rec_texts)]:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, SAMPLE_RATE)
                txt = wh.transcribe(f.name, language="en")["text"].strip().lower()
                bucket.append(txt)
                os.unlink(f.name)
        if (i + 1) % 10 == 0:
            print(f"    transcribed {i+1}/{len(orig_list)}")
    del wh
    torch.cuda.empty_cache()
    # WER of reconstruction vs the ASR transcript of the ORIGINAL (resynthesis WER)
    w = float(jiwer_wer(orig_texts, rec_texts))
    return w, orig_texts, rec_texts


MODELS = {
    # OG v9 reference: step 400000, the checkpoint run_v9_evals.py uses by default,
    # source of v9_results.json (PESQ 3.140). cd128 FSQ [8,8,8,8], 237.5 tok/s.
    "OG v9 (cd128 FSQ, 237.5 tok/s, 400K)": {
        "ckpt": "/local_data/data/checkpoints/v9_ll_stage2_400k.pt", "loader": "v9"},
    "cd64 Q2D2 (100 tok/s)": {
        "ckpt": "/local_data/checkpoints/v9_v2_cd64_nodisc/v2_latest.pt", "loader": "v2"},
    "cd64 FSQ (100 tok/s)": {
        "ckpt": "/local_data/checkpoints/v9_v2_cd64_fsq_nodisc/v2_latest.pt", "loader": "v2"},
    "cd32 Q2D2 (50 tok/s)": {
        "ckpt": "/local_data/checkpoints/v9_v2_cd32_nodisc/v2_latest.pt", "loader": "v2"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/local_data/data/librilight/librilight_9k")
    ap.add_argument("--output", default="/local_data/eval_results/wer_clean_0514.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_samples", type=int, default=50)
    args = ap.parse_args()

    device = torch.device(args.device)
    orig_wavs = load_eval_samples(args.data_dir, args.n_samples)

    results = {}
    for name, m in MODELS.items():
        print(f"\n{'='*60}\n{name}\n{'='*60}")
        if not os.path.exists(m["ckpt"]):
            print(f"  MISSING checkpoint: {m['ckpt']} -- skipping")
            results[name] = {"error": "checkpoint missing"}
            continue
        try:
            if m["loader"] == "v2":
                model, hop = load_v2_model(m["ckpt"], device)
            else:
                model, hop = load_v9_model(m["ckpt"], device)
            recs = reconstruct(model, hop, orig_wavs, device)
            del model
            torch.cuda.empty_cache()
            w, ot, rt = compute_wer(orig_wavs, recs, str(device))
            print(f"  WER = {w*100:.2f}%  (Whisper base.en, {len(orig_wavs)} samples)")
            results[name] = {
                "wer": w, "wer_pct": w * 100, "n_samples": len(orig_wavs),
                "asr": "whisper-base.en", "ckpt": m["ckpt"],
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[name] = {"error": str(e)}
        # write incrementally
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n{'='*60}\nSUMMARY (Whisper base.en, {len(orig_wavs)} samples)\n{'='*60}")
    for name, r in results.items():
        if "wer_pct" in r:
            print(f"  {name:<45} WER {r['wer_pct']:.2f}%")
        else:
            print(f"  {name:<45} {r.get('error','?')}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
