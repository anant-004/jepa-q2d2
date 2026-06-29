"""Run reconstruct_audio on Modal and save the result locally.

Usage:
    modal run scripts/run_reconstruct.py --input fish-audio-maya_test.mp3
"""
import modal
import sys
import os

# Import the app and function from modal_app
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from modal_app import app, reconstruct_audio

@app.local_entrypoint()
def main(
    input: str = "fish-audio-maya_test.mp3",
    output: str = "reconstructed_v9_304k.wav",
    ckpt: str = "/checkpoints/tokenizer_v9_ll/stage2_step304000.pt",
):
    with open(input, "rb") as f:
        audio_bytes = f.read()

    print(f"Sending {len(audio_bytes)} bytes to Modal...")
    print(f"Checkpoint: {ckpt}")

    result_bytes = reconstruct_audio.remote(
        audio_bytes=audio_bytes,
        ckpt_path=ckpt,
    )

    with open(output, "wb") as f:
        f.write(result_bytes)

    print(f"Saved: {output} ({len(result_bytes)} bytes)")
