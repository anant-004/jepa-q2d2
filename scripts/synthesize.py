"""CLI script for text-to-speech synthesis with a trained KoeTTS model.

Usage:
    python scripts/synthesize.py \
        --model checkpoints/ar_model/final.pt \
        --codec checkpoints/tokenizer/tokenizer_final.pt \
        --text "Hello, world!" \
        --output output.wav

    # With voice prompt (voice cloning)
    python scripts/synthesize.py \
        --model checkpoints/ar_model/final.pt \
        --codec checkpoints/tokenizer/tokenizer_final.pt \
        --text "Hello, world!" \
        --prompt prompt.wav \
        --output output.wav

    # Batch mode from text file
    python scripts/synthesize.py \
        --model checkpoints/ar_model/final.pt \
        --codec checkpoints/tokenizer/tokenizer_final.pt \
        --text_file sentences.txt \
        --prompt prompt.wav \
        --output_dir outputs/
"""

import argparse
import time
from pathlib import Path

import torch

from koe.generate import KoeTTSGenerator


def main():
    parser = argparse.ArgumentParser(description="KoeTTS text-to-speech synthesis")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained AR model checkpoint")
    parser.add_argument("--codec", type=str, required=True,
                        help="Path to trained tokenizer/codec checkpoint")
    parser.add_argument("--text", type=str, default=None,
                        help="Text to synthesize (single utterance)")
    parser.add_argument("--text_file", type=str, default=None,
                        help="File with one sentence per line (batch mode)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Path to voice prompt audio (for voice cloning)")
    parser.add_argument("--output", type=str, default="output.wav",
                        help="Output path for single utterance")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for batch mode")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_seconds", type=float, default=30.0,
                        help="Maximum audio duration in seconds")
    args = parser.parse_args()

    if not args.text and not args.text_file:
        parser.error("Must provide either --text or --text_file")

    # Load generator
    print(f"Loading model from {args.model}")
    print(f"Loading codec from {args.codec}")
    gen = KoeTTSGenerator.from_checkpoints(
        model_path=args.model,
        codec_path=args.codec,
        device=args.device,
    )
    print("Models loaded.")

    # Single utterance
    if args.text:
        print(f"Synthesizing: \"{args.text}\"")
        t0 = time.time()
        gen.generate_to_file(
            text=args.text,
            output_path=args.output,
            prompt_path=args.prompt,
            temperature=args.temperature,
            top_k=args.top_k,
            max_audio_seconds=args.max_seconds,
        )
        elapsed = time.time() - t0
        print(f"Saved to {args.output} ({elapsed:.1f}s)")

    # Batch mode
    if args.text_file:
        assert args.output_dir, "Must provide --output_dir for batch mode"
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(args.text_file) as f:
            sentences = [line.strip() for line in f if line.strip()]

        print(f"Synthesizing {len(sentences)} sentences...")
        for i, sentence in enumerate(sentences):
            out_path = output_dir / f"{i:04d}.wav"
            print(f"  [{i+1}/{len(sentences)}] \"{sentence[:50]}{'...' if len(sentence) > 50 else ''}\"")
            t0 = time.time()
            gen.generate_to_file(
                text=sentence,
                output_path=str(out_path),
                prompt_path=args.prompt,
                temperature=args.temperature,
                top_k=args.top_k,
                max_audio_seconds=args.max_seconds,
            )
            elapsed = time.time() - t0
            print(f"    → {out_path} ({elapsed:.1f}s)")

        print(f"Done. {len(sentences)} files saved to {output_dir}/")


if __name__ == "__main__":
    main()
