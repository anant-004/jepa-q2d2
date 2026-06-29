"""Pre-encode audio files into packed FSQ tokens using a trained KoeCodec.

Usage:
    python scripts/pre_encode.py \
        --codec_path checkpoints/tokenizer.pt \
        --input_manifest data/libritts_r/manifest.jsonl \
        --output_dir data/libritts_r/tokens/ \
        --output_manifest data/libritts_r/encoded_manifest.jsonl

Input manifest (JSONL):
    {"text": "Hello world", "audio_path": "/path/to/audio.wav"}
    {"text": "Good morning", "audio_path": "/path/to/audio.wav", "prompt_path": "/path/to/prompt.wav"}

Output manifest (JSONL):
    {"text": "Hello world", "audio_tokens": "/output/tokens/0000001.npy"}
    {"text": "Good morning", "audio_tokens": "/output/tokens/0000002.npy", "prompt_tokens": "/output/tokens/0000002_prompt.npy"}

Each .npy file contains packed FSQ tokens of shape [T_z, 19],
dtype int64, values in [0, 16383].
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from koe.codec import KoeCodec


def main():
    parser = argparse.ArgumentParser(description="Pre-encode audio to FSQ tokens")
    parser.add_argument("--codec_path", type=str, required=True,
                        help="Path to trained KoeCodec checkpoint (.pt)")
    parser.add_argument("--input_manifest", type=str, required=True,
                        help="JSONL manifest with text + audio_path")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save .npy token files")
    parser.add_argument("--output_manifest", type=str, required=True,
                        help="Path to write encoded JSONL manifest")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load codec
    print(f"Loading codec from {args.codec_path}")
    codec = KoeCodec.from_pretrained(args.codec_path, device=args.device)
    print(f"Codec loaded. Sample rate: {codec.config.sample_rate}, "
          f"hop: {codec.config.hop_length}, frame rate: {codec.config.frame_rate} Hz")

    # Load input manifest
    samples = []
    with open(args.input_manifest) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"Found {len(samples)} samples to encode")

    # Encode each sample
    encoded = []
    for i, sample in enumerate(samples):
        idx_str = f"{i:07d}"
        out_entry = {"text": sample["text"]}

        # Encode main audio
        audio_path = sample.get("audio_path") or sample.get("path")
        tokens = codec.encode_file(audio_path, device=torch.device(args.device))
        # tokens: [1, T_z, 19] → squeeze batch → [T_z, 19]
        tokens_np = tokens[0].cpu().numpy()
        token_path = output_dir / f"{idx_str}.npy"
        np.save(token_path, tokens_np)
        out_entry["audio_tokens"] = str(token_path)

        # Encode prompt audio if present
        prompt_path = sample.get("prompt_path")
        if prompt_path:
            prompt_tokens = codec.encode_file(prompt_path, device=torch.device(args.device))
            prompt_np = prompt_tokens[0].cpu().numpy()
            prompt_token_path = output_dir / f"{idx_str}_prompt.npy"
            np.save(prompt_token_path, prompt_np)
            out_entry["prompt_tokens"] = str(prompt_token_path)

        encoded.append(out_entry)

        if (i + 1) % 100 == 0:
            print(f"Encoded {i + 1}/{len(samples)} "
                  f"(last: {tokens_np.shape[0]} frames, "
                  f"{tokens_np.shape[0] * 19} tokens)")

    # Write output manifest
    with open(args.output_manifest, "w") as f:
        for entry in encoded:
            f.write(json.dumps(entry) + "\n")

    print(f"Done. Wrote {len(encoded)} entries to {args.output_manifest}")
    print(f"Token files saved to {output_dir}/")


if __name__ == "__main__":
    main()
