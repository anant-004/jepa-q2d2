"""4-GPU parallel experiment launcher.

Runs 4 independent training processes, one per GPU, with SIGTERM cascading
for graceful spot instance preemption.

Usage:
    python -m koe.fast.launcher \
        --data_dir /data/librilight \
        --output_dir ./checkpoints \
        --stage1_ckpt /data/stage1_final.pt \
        --gcs_bucket koe-checkpoints

    # Run specific experiments only:
    python -m koe.fast.launcher \
        --data_dir /data/librilight \
        --output_dir ./checkpoints \
        --stage1_ckpt /data/stage1_final.pt \
        --experiments baseline_fsq fsq_wavlm
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import Dict

from koe.fast.experiment_configs import EXPERIMENTS, get_experiment_args


def launch_experiment(
    name: str,
    gpu_id: int,
    args: argparse.Namespace,
) -> subprocess.Popen:
    """Launch a single experiment as a subprocess on a specific GPU."""
    train_args = get_experiment_args(
        name=name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        stage1_ckpt=args.stage1_ckpt,
        gcs_bucket=args.gcs_bucket,
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cmd = [sys.executable, "-m", "koe.fast.train_stage2"] + train_args

    log_dir = os.path.join(args.output_dir, name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")

    print(f"[launcher] Starting {name} on GPU {gpu_id} → {log_file}")

    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,  # New process group for clean SIGTERM
        )

    return proc


def main():
    parser = argparse.ArgumentParser(description="4-GPU parallel experiment launcher")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--gcs_bucket", type=str, default=None)
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Experiment names to run (default: all)")

    args = parser.parse_args()

    # Select experiments
    if args.experiments:
        exp_names = args.experiments
        for name in exp_names:
            if name not in EXPERIMENTS:
                print(f"[launcher] Unknown experiment: {name}")
                print(f"[launcher] Available: {list(EXPERIMENTS.keys())}")
                sys.exit(1)
    else:
        exp_names = list(EXPERIMENTS.keys())

    print(f"[launcher] Launching {len(exp_names)} experiments: {exp_names}")

    # Launch all experiments
    processes: Dict[str, subprocess.Popen] = {}
    for name in exp_names:
        gpu_id = EXPERIMENTS[name].gpu_id
        proc = launch_experiment(name, gpu_id, args)
        processes[name] = proc
        time.sleep(2)  # Stagger launches slightly

    print(f"[launcher] All {len(processes)} experiments running")

    # SIGTERM cascading: forward to all children
    def _on_sigterm(signum, frame):
        print(f"\n[launcher] SIGTERM received. Forwarding to {len(processes)} children...")
        for name, proc in processes.items():
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    print(f"[launcher] Sent SIGTERM to {name} (pid={proc.pid})")
                except ProcessLookupError:
                    pass

        # Wait for children to save (up to 25s within 30s grace period)
        deadline = time.time() + 25
        for name, proc in processes.items():
            remaining = max(0, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
                print(f"[launcher] {name} exited (rc={proc.returncode})")
            except subprocess.TimeoutExpired:
                print(f"[launcher] {name} did not exit in time, killing")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_sigterm)
    signal.signal(signal.SIGINT, _on_sigterm)

    # Monitor processes
    try:
        while True:
            all_done = True
            for name, proc in processes.items():
                rc = proc.poll()
                if rc is not None:
                    if rc != 0:
                        print(f"[launcher] {name} exited with code {rc}")
                else:
                    all_done = False

            if all_done:
                print("[launcher] All experiments completed")
                break

            time.sleep(10)

    except KeyboardInterrupt:
        _on_sigterm(signal.SIGINT, None)


if __name__ == "__main__":
    main()
