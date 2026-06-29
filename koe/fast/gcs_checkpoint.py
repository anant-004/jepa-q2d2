"""GCS checkpoint upload/download management for spot instance training.

Uses gsutil CLI (available on all GCS instances) for reliable checkpoint
persistence. Designed to survive spot preemptions with minimal checkpoint loss.
"""

import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional


class GCSCheckpointManager:
    """Manages checkpoint uploads/downloads to Google Cloud Storage.

    Designed for spot instance training where preemption can happen at any time.
    Uses gsutil CLI as primary (always available on GCS instances).
    Falls back gracefully if gsutil is not available.

    Features:
    - Background async uploads (non-blocking during training)
    - Blocking upload on SIGTERM (for spot preemption)
    - Auto-resume: download latest checkpoint from GCS
    - Keep last N checkpoints (configurable)
    """

    def __init__(
        self,
        bucket: str,
        prefix: str,
        local_dir: str = "./checkpoints",
        keep_last_n: int = 3,
    ):
        """
        Args:
            bucket: GCS bucket name (without gs://)
            prefix: Path prefix within bucket (e.g., "experiments/exp0")
            local_dir: Local checkpoint directory
            keep_last_n: Number of checkpoints to keep on GCS
        """
        self.bucket = bucket.strip("/")
        self.prefix = prefix.strip("/")
        self.local_dir = Path(local_dir)
        self.keep_last_n = keep_last_n

        self.local_dir.mkdir(parents=True, exist_ok=True)

        self._upload_thread: Optional[threading.Thread] = None
        self._upload_lock = threading.Lock()

        self._gsutil_available: Optional[bool] = None

    @property
    def gcs_prefix(self) -> str:
        return f"gs://{self.bucket}/{self.prefix}"

    def _check_gsutil(self) -> bool:
        """Check if gsutil is available. Cached after first call."""
        if self._gsutil_available is not None:
            return self._gsutil_available
        try:
            subprocess.run(
                ["gsutil", "version"],
                capture_output=True,
                timeout=10,
            )
            self._gsutil_available = True
            _log("gsutil available")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._gsutil_available = False
            _log("WARNING: gsutil not available, GCS operations disabled")
        return self._gsutil_available

    def _run_gsutil(
        self, args: List[str], timeout: int = 300
    ) -> subprocess.CompletedProcess:
        """Run a gsutil command with timeout.

        Args:
            args: Arguments to pass to gsutil (e.g., ["cp", src, dst])
            timeout: Timeout in seconds

        Returns:
            CompletedProcess result

        Raises:
            subprocess.TimeoutExpired: If command exceeds timeout
            subprocess.CalledProcessError: If command fails
        """
        cmd = ["gsutil"] + args
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )

    def _gcs_checkpoint_name(self, step: int) -> str:
        return f"checkpoint_step{step:07d}.pt"

    def _gcs_checkpoint_path(self, step: int) -> str:
        return f"{self.gcs_prefix}/{self._gcs_checkpoint_name(step)}"

    def upload_checkpoint(
        self,
        local_path: str,
        step: int,
        blocking: bool = False,
    ) -> None:
        """Upload a checkpoint to GCS.

        Args:
            local_path: Path to local checkpoint file
            step: Training step (for naming)
            blocking: If True, wait for upload to complete (use for SIGTERM handler).
                      If False, upload in background thread.
        """
        if not self._check_gsutil():
            _log(f"skipping upload for step {step} (gsutil unavailable)")
            return

        if not os.path.isfile(local_path):
            _log(f"WARNING: local checkpoint not found: {local_path}")
            return

        if blocking:
            self._do_upload(local_path, step)
        else:
            # Wait for any in-flight upload before starting a new one
            self._wait_for_upload()
            thread = threading.Thread(
                target=self._do_upload,
                args=(local_path, step),
                daemon=True,
            )
            thread.start()
            with self._upload_lock:
                self._upload_thread = thread

    def _do_upload(self, local_path: str, step: int) -> None:
        """Perform the actual upload. Runs in background thread or inline."""
        gcs_dest = self._gcs_checkpoint_path(step)
        filename = self._gcs_checkpoint_name(step)
        t0 = time.monotonic()

        try:
            # Upload checkpoint
            _log(f"uploading step {step} -> {gcs_dest}")
            self._run_gsutil(["cp", local_path, gcs_dest], timeout=300)
            elapsed = time.monotonic() - t0
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            _log(f"uploaded step {step} ({size_mb:.1f} MB in {elapsed:.1f}s)")

            # Update latest pointer
            self._update_latest_pointer(step, filename)

            # Cleanup old checkpoints
            self.cleanup_old_checkpoints()

        except subprocess.TimeoutExpired:
            _log(f"WARNING: upload timed out for step {step}")
        except subprocess.CalledProcessError as e:
            _log(f"WARNING: upload failed for step {step}: {e.stderr.strip()}")
        except Exception as e:
            _log(f"WARNING: unexpected upload error for step {step}: {e}")

    def _update_latest_pointer(self, step: int, filename: str) -> None:
        """Write and upload the latest.txt pointer file."""
        pointer_content = f"{step}\n{filename}\n"
        pointer_local = self.local_dir / "latest.txt"

        try:
            pointer_local.write_text(pointer_content)
            self._run_gsutil(
                ["cp", str(pointer_local), f"{self.gcs_prefix}/latest.txt"],
                timeout=30,
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            _log(f"WARNING: failed to update latest pointer: {e}")

    def _wait_for_upload(self, timeout: float = 300) -> None:
        """Wait for any in-flight background upload to finish."""
        with self._upload_lock:
            thread = self._upload_thread
        if thread is not None and thread.is_alive():
            _log("waiting for in-flight upload to finish...")
            thread.join(timeout=timeout)
            if thread.is_alive():
                _log("WARNING: in-flight upload did not finish within timeout")

    def download_latest(self) -> Optional[str]:
        """Download the latest checkpoint from GCS.

        Returns local path to downloaded checkpoint, or None if not found.
        Checks for a 'latest.txt' pointer file in the GCS prefix.
        """
        if not self._check_gsutil():
            return None

        # Try to read the latest pointer
        pointer_gcs = f"{self.gcs_prefix}/latest.txt"
        pointer_local = self.local_dir / "latest.txt"

        try:
            self._run_gsutil(["cp", pointer_gcs, str(pointer_local)], timeout=30)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            _log("no latest.txt found on GCS, no checkpoint to resume from")
            return None

        try:
            lines = pointer_local.read_text().strip().split("\n")
            if len(lines) < 2:
                _log(f"WARNING: malformed latest.txt: {lines}")
                return None
            step = int(lines[0])
            filename = lines[1]
        except (ValueError, IndexError) as e:
            _log(f"WARNING: failed to parse latest.txt: {e}")
            return None

        # Download the checkpoint
        gcs_path = f"{self.gcs_prefix}/{filename}"
        local_path = self.local_dir / filename

        # Skip download if we already have it locally
        if local_path.is_file():
            _log(f"checkpoint already exists locally: {local_path} (step {step})")
            return str(local_path)

        _log(f"downloading checkpoint step {step} from {gcs_path}")
        t0 = time.monotonic()

        try:
            self._run_gsutil(["cp", gcs_path, str(local_path)], timeout=600)
            elapsed = time.monotonic() - t0
            size_mb = local_path.stat().st_size / (1024 * 1024)
            _log(f"downloaded step {step} ({size_mb:.1f} MB in {elapsed:.1f}s)")
            return str(local_path)
        except subprocess.TimeoutExpired:
            _log("WARNING: checkpoint download timed out")
            local_path.unlink(missing_ok=True)
            return None
        except subprocess.CalledProcessError as e:
            _log(f"WARNING: checkpoint download failed: {e.stderr.strip()}")
            local_path.unlink(missing_ok=True)
            return None

    def cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints from GCS, keeping last N."""
        if self.keep_last_n <= 0:
            return

        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= self.keep_last_n:
            return

        # Sort by name (which sorts by step due to zero-padded naming)
        checkpoints.sort()
        to_delete = checkpoints[: len(checkpoints) - self.keep_last_n]

        for ckpt_path in to_delete:
            try:
                self._run_gsutil(["rm", ckpt_path], timeout=30)
                _log(f"deleted old checkpoint: {ckpt_path}")
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                _log(f"WARNING: failed to delete {ckpt_path}: {e}")

    def list_checkpoints(self) -> List[str]:
        """List all checkpoint files in the GCS prefix.

        Returns list of full GCS paths (gs://bucket/prefix/checkpoint_*.pt).
        """
        if not self._check_gsutil():
            return []

        try:
            result = self._run_gsutil(
                ["ls", f"{self.gcs_prefix}/checkpoint_step*.pt"],
                timeout=30,
            )
            paths = [
                line.strip()
                for line in result.stdout.strip().split("\n")
                if line.strip() and line.strip().endswith(".pt")
            ]
            return paths
        except subprocess.CalledProcessError:
            # No files found or prefix doesn't exist — not an error
            return []
        except subprocess.TimeoutExpired:
            _log("WARNING: listing checkpoints timed out")
            return []

    def wait_for_pending_upload(self, timeout: float = 300) -> None:
        """Public method to wait for any in-flight upload. Call before exit."""
        self._wait_for_upload(timeout=timeout)


def make_sigterm_handler(
    save_fn: Callable[[], str],
    gcs_manager: Optional[GCSCheckpointManager],
    step_fn: Callable[[], int],
) -> Callable:
    """Create a SIGTERM signal handler that saves checkpoint + uploads to GCS.

    Budget: 30 seconds (GCS spot preemption grace period).
    - Save locally: ~3 seconds
    - Upload to GCS: ~15 seconds
    - Buffer: ~12 seconds

    Args:
        save_fn: Callable that saves checkpoint locally, returns path to saved file.
        gcs_manager: GCSCheckpointManager instance. If None, only saves locally.
        step_fn: Callable that returns the current training step.

    Returns:
        A signal handler function to register with signal.signal().
    """

    def handler(signum, frame):
        step = step_fn()
        _log(f"SIGTERM received at step {step}, saving checkpoint...")
        t0 = time.monotonic()

        try:
            local_path = save_fn()
            save_elapsed = time.monotonic() - t0
            _log(f"checkpoint saved locally in {save_elapsed:.1f}s: {local_path}")
        except Exception as e:
            _log(f"CRITICAL: failed to save checkpoint on SIGTERM: {e}")
            sys.exit(1)

        if gcs_manager is not None:
            try:
                # Blocking upload — we must finish before the VM is killed
                gcs_manager.upload_checkpoint(
                    local_path, step=step, blocking=True
                )
                total_elapsed = time.monotonic() - t0
                _log(f"SIGTERM handling complete in {total_elapsed:.1f}s")
            except Exception as e:
                _log(f"WARNING: GCS upload failed during SIGTERM: {e}")

        sys.exit(0)

    return handler


def _log(msg: str) -> None:
    print(f"[gcs] {msg}", flush=True)
