"""Experiment configurations for 4-GPU parallel research.

Each experiment runs on a separate GPU with different hyperparameters.
All share the same base architecture (2.5 Hz, strides [8,8,5,5,6]).

Usage:
    from experiment_configs import EXPERIMENTS, get_experiment_args
    args = get_experiment_args("baseline_fsq", data_dir="/data", output_dir="./ckpts")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    gpu_id: int
    description: str

    # Overrides (None = use train.py defaults)
    quantizer: str = "fsq"
    lambda_perceptual: float = 0.0
    lambda_phase: float = 0.0
    lambda_gan: float = 0.1
    lambda_stft: float = 2.0
    lr_gen: float = 1.5e-4
    lr_disc: float = 7.5e-5
    batch_size: int = 8
    total_steps: int = 100000
    q2d2_levels: int = 4
    q2d2_grid: str = "rhombic"

    def to_args(
        self,
        data_dir: str,
        output_dir: str,
        stage1_ckpt: str,
        gcs_bucket: Optional[str] = None,
    ) -> List[str]:
        """Convert to command-line arguments for train.py."""
        base_output = f"{output_dir}/{self.name}"
        args = [
            "--data_dir", data_dir,
            "--output_dir", base_output,
            "--stage1_ckpt", stage1_ckpt,
            "--run_name", self.name,
            "--quantizer", self.quantizer,
            "--lambda_perceptual", str(self.lambda_perceptual),
            "--lambda_phase", str(self.lambda_phase),
            "--lambda_gan", str(self.lambda_gan),
            "--lambda_stft", str(self.lambda_stft),
            "--lr_gen", str(self.lr_gen),
            "--lr_disc", str(self.lr_disc),
            "--batch_size", str(self.batch_size),
            "--total_steps", str(self.total_steps),
        ]
        if self.quantizer == "q2d2":
            args.extend([
                "--q2d2_levels", str(self.q2d2_levels),
                "--q2d2_grid", self.q2d2_grid,
            ])
        if gcs_bucket:
            args.extend([
                "--gcs_bucket", gcs_bucket,
                "--gcs_prefix", self.name,
            ])
        return args


# ═══════════════════════════════════════════════════════════
# 4 initial research directions
# ═══════════════════════════════════════════════════════════

EXPERIMENTS: Dict[str, ExperimentConfig] = {
    "baseline_fsq": ExperimentConfig(
        name="baseline_fsq",
        gpu_id=0,
        description="Vanilla 2.5 Hz FSQ decoder training with encoder FT",
    ),
    "fsq_wavlm": ExperimentConfig(
        name="fsq_wavlm",
        gpu_id=1,
        description="FSQ + WavLM perceptual loss (StableCodec recipe)",
        lambda_perceptual=0.1,
    ),
    "fsq_phase_gan": ExperimentConfig(
        name="fsq_phase_gan",
        gpu_id=2,
        description="FSQ + phase-aware STFT + stronger GAN",
        lambda_phase=0.1,
        lambda_gan=0.2,
    ),
    "q2d2_baseline": ExperimentConfig(
        name="q2d2_baseline",
        gpu_id=3,
        description="Q2D2 quantizer replacing FSQ (geometry-aware 2D grids)",
        quantizer="q2d2",
        q2d2_levels=4,
        q2d2_grid="rhombic",
    ),
}


def get_experiment_args(
    name: str,
    data_dir: str,
    output_dir: str,
    stage1_ckpt: str,
    gcs_bucket: Optional[str] = None,
) -> List[str]:
    """Get CLI args for a named experiment."""
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {name}. Available: {list(EXPERIMENTS.keys())}")
    return EXPERIMENTS[name].to_args(data_dir, output_dir, stage1_ckpt, gcs_bucket)
