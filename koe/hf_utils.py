"""HuggingFace Hub utilities for checkpoint management.

Push/pull model checkpoints to a private HuggingFace repo.
Used by train_tokenizer.py and modal_app.py for durable storage
that survives Modal preemption.

Usage:
    from koe.hf_utils import push_checkpoint_to_hf, pull_latest_checkpoint

    # Upload after training step
    push_checkpoint_to_hf(state_dict, config, step=1000, stage="stage1")

    # Resume from latest
    ckpt = pull_latest_checkpoint(stage="stage1")
"""

import io
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _get_hf_api(hf_token: Optional[str] = None):
    """Get HuggingFace API client."""
    from huggingface_hub import HfApi
    token = hf_token or os.environ.get("HF_TOKEN")
    return HfApi(token=token)


def push_checkpoint_to_hf(
    state_dict: Dict[str, Any],
    config: Dict[str, Any],
    step: int,
    stage: str,
    repo_id: str = "Andy004/koe-tokenizer",
    hf_token: Optional[str] = None,
) -> str:
    """Push a checkpoint to a private HuggingFace repo.

    Args:
        state_dict: model state dict to save
        config: config dict to include in checkpoint
        step: training step number
        stage: "stage1" or "stage2"
        repo_id: HuggingFace repo ID
        hf_token: HF API token (falls back to HF_TOKEN env var)

    Returns:
        URL of the uploaded file
    """
    from huggingface_hub import HfApi

    api = _get_hf_api(hf_token)

    # Ensure repo exists (don't force private — repo may already be public)
    api.create_repo(repo_id, exist_ok=True)

    # Save checkpoint to a temporary file
    filename = f"{stage}_step{step}.pt"
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save({"step": step, "state_dict": state_dict, "config": config}, f)
        tmp_path = f.name

    try:
        url = api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=f"checkpoints/{filename}",
            repo_id=repo_id,
        )
    finally:
        os.unlink(tmp_path)

    # Also save as "latest" for easy resume
    latest_filename = f"{stage}_latest.pt"
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save({"step": step, "state_dict": state_dict, "config": config}, f)
        tmp_path = f.name

    try:
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=f"checkpoints/{latest_filename}",
            repo_id=repo_id,
        )
    finally:
        os.unlink(tmp_path)

    return url


def pull_checkpoint_from_hf(
    filename: str,
    repo_id: str = "Andy004/koe-tokenizer",
    hf_token: Optional[str] = None,
    device: str = "cpu",
) -> Optional[Dict[str, Any]]:
    """Download a specific checkpoint from HuggingFace.

    Args:
        filename: e.g. "stage1_step1000.pt" or "stage1_latest.pt"
        repo_id: HuggingFace repo ID
        hf_token: HF API token
        device: device to load checkpoint onto

    Returns:
        Checkpoint dict or None if not found
    """
    from huggingface_hub import hf_hub_download

    token = hf_token or os.environ.get("HF_TOKEN")

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=f"checkpoints/{filename}",
            token=token,
        )
        return torch.load(path, map_location=device, weights_only=False)
    except Exception:
        return None


def pull_latest_checkpoint(
    stage: str,
    repo_id: str = "Andy004/koe-tokenizer",
    hf_token: Optional[str] = None,
    device: str = "cpu",
) -> Optional[Dict[str, Any]]:
    """Pull the latest checkpoint for a given stage.

    Args:
        stage: "stage1" or "stage2"
        repo_id: HuggingFace repo ID
        hf_token: HF API token
        device: device to load onto

    Returns:
        Checkpoint dict or None if no checkpoint exists
    """
    return pull_checkpoint_from_hf(
        f"{stage}_latest.pt", repo_id=repo_id, hf_token=hf_token, device=device,
    )


def push_final_model(
    state_dict: Dict[str, Any],
    config: Dict[str, Any],
    stage: str,
    repo_id: str = "Andy004/koe-tokenizer",
    hf_token: Optional[str] = None,
) -> str:
    """Push final consolidated model weights with a model card.

    Args:
        state_dict: final model state dict
        config: codec/model config
        stage: "stage1", "stage2", or "full"
        repo_id: HuggingFace repo ID
        hf_token: HF API token

    Returns:
        URL of uploaded file
    """
    api = _get_hf_api(hf_token)
    api.create_repo(repo_id, private=True, exist_ok=True)

    filename = f"{stage}_final.pt"
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save({"state_dict": state_dict, "config": config}, f)
        tmp_path = f.name

    try:
        url = api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=filename,
            repo_id=repo_id,
        )
    finally:
        os.unlink(tmp_path)

    return url


def update_model_card(
    repo_id: str = "Andy004/koe-tokenizer",
    eval_metrics: Optional[Dict[str, float]] = None,
    step: Optional[int] = None,
    hf_token: Optional[str] = None,
):
    """Update the model card with evaluation results.

    Args:
        repo_id: HuggingFace repo ID
        eval_metrics: dict of metric name → value
        step: training step
        hf_token: HF API token
    """
    from huggingface_hub import ModelCard

    api = _get_hf_api(hf_token)

    card_text = "---\nlicense: apache-2.0\ntags:\n- audio\n- codec\n- tts\n---\n\n"
    card_text += "# KoeTTS JEPA Tokenizer\n\n"
    card_text += "Audio codec based on Density-Adaptive JEPA with FSQ quantization.\n"
    card_text += "Compresses audio to 47.5 tokens/second (19 groups × 2.5 Hz).\n\n"

    if eval_metrics and step:
        card_text += f"## Evaluation at Step {step}\n\n"
        card_text += "| Metric | Value |\n|--------|-------|\n"
        for name, val in sorted(eval_metrics.items()):
            card_text += f"| {name} | {val:.4f} |\n"
        card_text += "\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(card_text)
        tmp_path = f.name

    try:
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            repo_id=repo_id,
        )
    finally:
        os.unlink(tmp_path)
