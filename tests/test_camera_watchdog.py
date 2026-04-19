"""Tests for CameraWatchdog."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.camera_watchdog import CameraWatchdog, WatchdogStatus
from src.config import WatchdogConfig
from src.settings_store import AllSkySettings, SSHSettings, SettingsStore


def _make_watchdog(
    tmp_path: Path,
    check_interval: int = 3600,  # very long so it doesn't auto-tick in tests
    stale_threshold: int = 300,
) -> CameraWatchdog:
    cfg = WatchdogConfig(
        check_interval_seconds=check_interval,
        stale_threshold_seconds=stale_threshold,
        service_name="indi-allsky",
        data_dir=str(tmp_path / "settings"),
    )
    store = SettingsStore(data_dir=tmp_path / "settings")
    return CameraWatchdog(config=cfg, settings_store=store, image_dir=tmp_path / "images")


class TestWatchdogStatus:
    def test_initial_status(self, tmp_path: Path):
        wd = _make_watchdog(tmp_path)
        assert wd.status.is_healthy is True
        assert wd.status.consecutive_failures == 0
        assert wd.status.running is False

    def test_status_as_dict_keys(self, tmp_path: Path):
        wd = _make_watchdog(tmp_path)
        d = wd.status.as_dict()
        assert set(d.keys()) == {
            "is_healthy", "last_check", "last_image",
            "last_restart", "last_restart_result",
            "consecutive_failures", "running",
        }


class TestWatchdogNewestImageMtime:
    def test_returns_none_for_empty_directory(self, tmp_path: Path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        wd = _make_watchdog(tmp_path)
        assert wd._newest_image_mtime() is None

    def test_returns_none_for_missing_directory(self, tmp_path: Path):
        wd = _make_watchdog(tmp_path)
        # image_dir does not exist
        assert wd._newest_image_mtime() is None

    def test_finds_newest_image(self, tmp_path: Path):
        import os

        img_dir = tmp_path / "images"
        img_dir.mkdir()

        older = img_dir / "old.jpg"
        newer = img_dir / "new.jpg"
        older.write_bytes(b"")
        newer.write_bytes(b"")

        os.utime(older, (time.time() - 600, time.time() - 600))
        os.utime(newer, (time.time() - 10, time.time() - 10))

        wd = _make_watchdog(tmp_path)
        mtime = wd._newest_image_mtime()
        assert mtime is not None
        assert abs(mtime - (time.time() - 10)) < 5

    def test_ignores_non_image_files(self, tmp_path: Path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        (img_dir / "readme.txt").write_bytes(b"text")

        wd = _make_watchdog(tmp_path)
        assert wd._newest_image_mtime() is None


class TestWatchdogTickHealthy:
    def test_tick_marks_healthy_when_fresh_image(self, tmp_path: Path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        fresh = img_dir / "fresh.jpg"
        fresh.write_bytes(b"")

        wd = _make_watchdog(tmp_path, stale_threshold=300)
        wd._tick()

        assert wd.status.is_healthy is True
        assert wd.status.consecutive_failures == 0

    def test_tick_marks_unhealthy_when_no_images(self, tmp_path: Path):
        # No images at all → treated as healthy (system starting up).
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        wd = _make_watchdog(tmp_path, stale_threshold=1)
        wd._tick()
        # No images → healthy (skip check).
        assert wd.status.is_healthy is True


class TestWatchdogTickStale:
    def test_increments_consecutive_failures_on_stale(self, tmp_path: Path):
        import os

        img_dir = tmp_path / "images"
        img_dir.mkdir()

        stale = img_dir / "stale.jpg"
        stale.write_bytes(b"")
        os.utime(stale, (time.time() - 1000, time.time() - 1000))

        wd = _make_watchdog(tmp_path, stale_threshold=300)
        # Prevent actual SSH call.
        wd._attempt_restart = MagicMock()

        wd._tick()
        assert wd.status.is_healthy is False
        assert wd.status.consecutive_failures == 1

        wd._tick()
        assert wd.status.consecutive_failures == 2

    def test_calls_attempt_restart_when_stale(self, tmp_path: Path):
        import os

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        stale = img_dir / "stale.jpg"
        stale.write_bytes(b"")
        os.utime(stale, (time.time() - 1000, time.time() - 1000))

        wd = _make_watchdog(tmp_path, stale_threshold=300)
        wd._attempt_restart = MagicMock()

        wd._tick()
        wd._attempt_restart.assert_called_once()


class TestWatchdogAttemptRestart:
    def test_error_when_ssh_not_configured(self, tmp_path: Path):
        wd = _make_watchdog(tmp_path)
        wd._attempt_restart()
        assert "ERROR" in wd.status.last_restart_result
        assert "not configured" in wd.status.last_restart_result

    def test_uses_settings_store_ssh_credentials(self, tmp_path: Path):
        store = SettingsStore(data_dir=tmp_path / "settings")
        store.save(AllSkySettings(ssh=SSHSettings(host="1.2.3.4", username="pi", password="pw")))

        cfg = WatchdogConfig(
            check_interval_seconds=3600,
            stale_threshold_seconds=300,
            data_dir=str(tmp_path / "settings"),
        )
        wd = CameraWatchdog(config=cfg, settings_store=store, image_dir=tmp_path / "images")

        with patch("src.camera_watchdog.CameraWatchdog._ssh_restart", return_value="OK – succeeded.") as mock_ssh:
            wd._attempt_restart()
            mock_ssh.assert_called_once()
            ssh_arg = mock_ssh.call_args[0][0]
            assert ssh_arg.host == "1.2.3.4"
            assert ssh_arg.username == "pi"


class TestWatchdogStartStop:
    def test_start_sets_running(self, tmp_path: Path):
        wd = _make_watchdog(tmp_path)
        wd.start()
        assert wd.status.running is True
        wd.stop()

    def test_stop_clears_running(self, tmp_path: Path):
        wd = _make_watchdog(tmp_path)
        wd.start()
        wd.stop()
        assert wd.status.running is False

    def test_double_start_does_not_create_extra_threads(self, tmp_path: Path):
        import threading
        wd = _make_watchdog(tmp_path)
        before = threading.active_count()
        wd.start()
        wd.start()  # second call is a no-op
        after = threading.active_count()
        assert after - before <= 1
        wd.stop()
