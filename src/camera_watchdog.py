"""
CameraWatchdog – monitor indi-allsky camera health and auto-restart via SSH.

The watchdog runs as a daemon thread.  On each tick it checks whether a
new image has appeared in the configured directory within the last
``stale_threshold_seconds`` seconds.  If the camera appears to have stopped
producing images it connects to the indi-allsky host over SSH and restarts
the systemd service.

Status is exposed through :attr:`CameraWatchdog.status` and consumed by the
web UI watchdog page.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import WatchdogConfig
from .settings_store import SettingsStore, SSHSettings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status data class
# ---------------------------------------------------------------------------


@dataclass
class WatchdogStatus:
    """Current state of the camera watchdog."""

    is_healthy: bool = True
    last_check_time: Optional[float] = None   # Unix timestamp
    last_image_time: Optional[float] = None   # Unix timestamp of newest image
    last_restart_time: Optional[float] = None # Unix timestamp of last SSH restart
    last_restart_result: str = ""             # Success / error message
    consecutive_failures: int = 0
    running: bool = False

    def as_dict(self) -> dict:
        import time as _time

        def _fmt(ts: Optional[float]) -> Optional[str]:
            return _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime(ts)) if ts else None

        return {
            "is_healthy": self.is_healthy,
            "last_check": _fmt(self.last_check_time),
            "last_image": _fmt(self.last_image_time),
            "last_restart": _fmt(self.last_restart_time),
            "last_restart_result": self.last_restart_result,
            "consecutive_failures": self.consecutive_failures,
            "running": self.running,
        }


# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------


class CameraWatchdog:
    """Background thread that monitors indi-allsky camera health.

    Parameters
    ----------
    config:
        Watchdog section of the application config.
    settings_store:
        Provides the SSH credentials at check time (so settings changes are
        picked up without restarting the watchdog).
    image_dir:
        Directory to watch for new images.
    """

    def __init__(
        self,
        config: WatchdogConfig,
        settings_store: SettingsStore,
        image_dir: str | Path,
    ) -> None:
        self.config = config
        self.settings_store = settings_store
        self.image_dir = Path(image_dir)
        self.status = WatchdogStatus()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="camera-watchdog", daemon=True
        )
        self._thread.start()
        self.status.running = True
        logger.info("Camera watchdog started (check every %ds).", self.config.check_interval_seconds)

    def stop(self) -> None:
        """Signal the watchdog thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        self.status.running = False
        logger.info("Camera watchdog stopped.")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.wait(timeout=self.config.check_interval_seconds):
            try:
                self._tick()
            except Exception as exc:
                logger.exception("Watchdog tick raised an unexpected error: %s", exc)

    def _tick(self) -> None:
        """Single health-check iteration."""
        now = time.time()
        self.status.last_check_time = now

        latest_mtime = self._newest_image_mtime()
        self.status.last_image_time = latest_mtime

        if latest_mtime is None:
            # No images at all yet – treat as healthy (system may be starting).
            logger.debug("Watchdog: no images found; skipping health check.")
            self.status.is_healthy = True
            self.status.consecutive_failures = 0
            return

        age_seconds = now - latest_mtime
        if age_seconds <= self.config.stale_threshold_seconds:
            self.status.is_healthy = True
            self.status.consecutive_failures = 0
            logger.debug("Watchdog: camera healthy (newest image %.0fs ago).", age_seconds)
        else:
            self.status.consecutive_failures += 1
            self.status.is_healthy = False
            logger.warning(
                "Watchdog: camera STALE – newest image %.0fs ago (threshold %ds). "
                "Failure #%d.",
                age_seconds,
                self.config.stale_threshold_seconds,
                self.status.consecutive_failures,
            )
            self._attempt_restart()

    # ------------------------------------------------------------------
    # Image directory helpers
    # ------------------------------------------------------------------

    def _newest_image_mtime(self) -> Optional[float]:
        """Return the modification time of the most-recently-changed image."""
        if not self.image_dir.exists():
            return None
        mtimes = [
            p.stat().st_mtime
            for p in self.image_dir.glob("**/*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".fits"}
        ]
        return max(mtimes) if mtimes else None

    # ------------------------------------------------------------------
    # SSH restart
    # ------------------------------------------------------------------

    def _attempt_restart(self) -> None:
        """SSH to the indi-allsky host and restart the service."""
        settings = self.settings_store.load()
        ssh = settings.ssh

        if not ssh.host or not ssh.username:
            logger.error(
                "Watchdog: cannot restart – SSH host/username not configured."
            )
            self.status.last_restart_time = time.time()
            self.status.last_restart_result = (
                "ERROR: SSH host or username not configured in settings."
            )
            return

        logger.info(
            "Watchdog: attempting SSH restart of %s on %s@%s:%d",
            self.config.service_name, ssh.username, ssh.host, ssh.port,
        )
        result = self._ssh_restart(ssh)
        self.status.last_restart_time = time.time()
        self.status.last_restart_result = result
        logger.info("Watchdog restart result: %s", result)

    def _ssh_restart(self, ssh: SSHSettings) -> str:
        """
        Open an SSH connection and restart the indi-allsky systemd service.

        The SSH client loads the user's known_hosts file (``~/.ssh/known_hosts``)
        and rejects connections to unknown hosts.  If the target host is not yet
        in known_hosts, run ``ssh-keyscan <host> >> ~/.ssh/known_hosts`` on the
        AllSkyAnalyzer server before the watchdog will be able to connect.

        Returns a human-readable result string.
        """
        try:
            import paramiko  # type: ignore[import]
        except ImportError:
            return "ERROR: paramiko not installed – cannot perform SSH restart."

        client = paramiko.SSHClient()
        # Load the system-wide and per-user known_hosts files so that only
        # previously verified host keys are accepted (prevents MITM attacks).
        client.load_system_host_keys()
        try:
            client.load_host_keys(str(Path.home() / ".ssh" / "known_hosts"))
        except (OSError, IOError):
            pass  # File may not exist yet; system keys are still loaded.
        # Reject connections to hosts whose key is not in known_hosts.
        client.set_missing_host_key_policy(paramiko.RejectPolicy())

        try:
            connect_kwargs: dict = {
                "hostname": ssh.host,
                "port": ssh.port,
                "username": ssh.username,
                "timeout": 30,
            }
            if ssh.password:
                connect_kwargs["password"] = ssh.password

            client.connect(**connect_kwargs)

            command = f"sudo systemctl restart {self.config.service_name}"
            _stdin, stdout, stderr = client.exec_command(command, timeout=60)
            exit_status = stdout.channel.recv_exit_status()
            err_output = stderr.read().decode(errors="replace").strip()

            if exit_status == 0:
                return f"OK – '{command}' succeeded."
            else:
                return f"ERROR (exit {exit_status}): {err_output or 'no stderr'}"

        except paramiko.ssh_exception.NoValidConnectionsError as exc:
            return f"ERROR: Cannot connect to {ssh.host}:{ssh.port} – {exc}"
        except paramiko.ssh_exception.SSHException as exc:
            # Catches unknown-host-key rejection with a helpful hint.
            return (
                f"ERROR: SSH key error – {exc}. "
                f"Run: ssh-keyscan {ssh.host} >> ~/.ssh/known_hosts"
            )
        except Exception as exc:
            return f"ERROR: {exc}"
        finally:
            client.close()
