"""Tests for the IndiAllSkyReader module."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.config import IndiAllSkyConfig
from src.indi_allsky_reader import IndiAllSkyReader


def _make_images(base: Path, names: list[str]) -> list[Path]:
    """Helper: write empty files in *base* and return their paths."""
    base.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        p = base / name
        p.write_bytes(b"")
        paths.append(p)
    return paths


class TestIndiAllSkyReaderIterImages:
    def test_yields_nothing_for_missing_directory(self, tmp_path):
        cfg = IndiAllSkyConfig(image_dir=str(tmp_path / "nonexistent"))
        reader = IndiAllSkyReader(cfg)
        assert list(reader.iter_images()) == []

    def test_yields_matching_images(self, tmp_path):
        _make_images(tmp_path, ["a.jpg", "b.jpeg", "c.png"])
        cfg = IndiAllSkyConfig(image_dir=str(tmp_path), file_pattern="*.jpg")
        reader = IndiAllSkyReader(cfg)
        found = [p.name for p in reader.iter_images()]
        assert "a.jpg" in found
        assert "b.jpeg" not in found  # pattern is *.jpg only

    def test_ignores_non_image_files(self, tmp_path):
        _make_images(tmp_path, ["image.jpg", "notes.txt", "data.csv"])
        cfg = IndiAllSkyConfig(image_dir=str(tmp_path), file_pattern="*.jpg")
        reader = IndiAllSkyReader(cfg)
        found = [p.name for p in reader.iter_images()]
        assert "notes.txt" not in found
        assert "data.csv" not in found

    def test_max_age_filters_old_images(self, tmp_path):
        old = tmp_path / "old.jpg"
        old.write_bytes(b"")
        # Back-date the file modification time.
        old_mtime = time.time() - 7200  # 2 hours ago
        import os
        os.utime(old, (old_mtime, old_mtime))

        new = tmp_path / "new.jpg"
        new.write_bytes(b"")

        cfg = IndiAllSkyConfig(
            image_dir=str(tmp_path),
            file_pattern="*.jpg",
            max_age_seconds=3600,  # only files within the last hour
        )
        reader = IndiAllSkyReader(cfg)
        found = [p.name for p in reader.iter_images()]
        assert "new.jpg" in found
        assert "old.jpg" not in found


class TestIndiAllSkyReaderLatestImage:
    def test_returns_none_when_no_images(self, tmp_path):
        cfg = IndiAllSkyConfig(image_dir=str(tmp_path / "empty"), file_pattern="*.jpg")
        reader = IndiAllSkyReader(cfg)
        assert reader.latest_image() is None

    def test_returns_most_recent_image(self, tmp_path):
        import os

        img1 = tmp_path / "first.jpg"
        img2 = tmp_path / "second.jpg"
        img1.write_bytes(b"")
        img2.write_bytes(b"")

        # Ensure img2 is strictly newer.
        os.utime(img1, (time.time() - 10, time.time() - 10))
        os.utime(img2, (time.time(), time.time()))

        cfg = IndiAllSkyConfig(image_dir=str(tmp_path), file_pattern="*.jpg")
        reader = IndiAllSkyReader(cfg)
        assert reader.latest_image().name == "second.jpg"
