"""Tests for the StarFinder module."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.config import AstrometryConfig
from src.star_finder import StarFinder, Star, StarFinderResult, _CATALOGUE, _CONSTELLATION_LINES


def _make_starfield(height: int = 400, width: int = 400) -> np.ndarray:
    """Create a synthetic dark-sky image with a handful of bright point sources."""
    bgr = np.zeros((height, width, 3), dtype=np.uint8)
    for cx, cy, brightness in [
        (100, 100, 255),
        (200, 150, 220),
        (300, 300, 200),
        (50, 350, 180),
    ]:
        cv2.circle(bgr, (cx, cy), 2, (brightness, brightness, brightness), -1)
    return bgr


class TestStarFinderBlobDetection:
    """Unit tests for the photometric detection path (no plate solving)."""

    def _finder(self) -> StarFinder:
        cfg = AstrometryConfig(index_dir="/nonexistent", fov_deg=0.0)
        return StarFinder(cfg)

    def _save_image(self, bgr: np.ndarray, tmp_path: Path) -> Path:
        p = tmp_path / "test_stars.jpg"
        cv2.imwrite(str(p), bgr)
        return p

    def test_finds_bright_stars(self, tmp_path):
        finder = self._finder()
        bgr = _make_starfield()
        img_path = self._save_image(bgr, tmp_path)
        result = finder.find(img_path)
        assert isinstance(result, StarFinderResult)
        # We injected 4 point sources; detector should find at least some.
        assert len(result.stars) > 0

    def test_returns_empty_for_nonexistent_file(self, tmp_path):
        finder = self._finder()
        result = finder.find(tmp_path / "nope.jpg")
        assert result.stars == []
        assert result.plate_solved is False

    def test_stars_sorted_brightest_first(self, tmp_path):
        finder = self._finder()
        bgr = _make_starfield()
        img_path = self._save_image(bgr, tmp_path)
        result = finder.find(img_path)
        if len(result.stars) > 1:
            brightnesses = [s.brightness for s in result.stars]
            assert brightnesses == sorted(brightnesses, reverse=True)

    def test_no_plate_solve_without_index_dir(self, tmp_path):
        """Plate solving should be skipped when index_dir does not exist."""
        finder = self._finder()
        bgr = _make_starfield()
        img_path = self._save_image(bgr, tmp_path)
        result = finder.find(img_path)
        assert result.plate_solved is False
        assert result.wcs_header is None


class TestSkyToPixel:
    """Unit tests for the WCS gnomonic projection helper."""

    def _wcs(self, crval1=0.0, crval2=0.0, crpix1=200.0, crpix2=200.0,
             cdelt1=-0.1, cdelt2=0.1) -> dict:
        return {
            "CRVAL1": crval1, "CRVAL2": crval2,
            "CRPIX1": crpix1, "CRPIX2": crpix2,
            "CDELT1": cdelt1, "CDELT2": cdelt2,
            "CD1_1": cdelt1, "CD2_2": cdelt2,
        }

    def test_centre_maps_to_crpix(self):
        wcs = self._wcs()
        px, py = StarFinder._sky_to_pixel(0.0, 0.0, wcs)
        assert abs(px - 200.0) < 1.0
        assert abs(py - 200.0) < 1.0

    def test_off_centre_ra(self):
        wcs = self._wcs()
        px_centre, _ = StarFinder._sky_to_pixel(0.0, 0.0, wcs)
        px_offset, _ = StarFinder._sky_to_pixel(1.0, 0.0, wcs)
        # Moving 1 deg in RA should shift x by ~10 pixels (1/0.1 deg/px).
        assert abs((px_offset - px_centre) - (-10.0)) < 2.0

    def test_behind_projection_plane_returns_sentinel(self):
        wcs = self._wcs(crval1=0.0, crval2=0.0)
        # RA 180 deg is on the opposite side of the celestial sphere.
        px, py = StarFinder._sky_to_pixel(180.0, 0.0, wcs)
        assert px == pytest.approx(-9999.0)
        assert py == pytest.approx(-9999.0)


class TestCatalogueIntegrity:
    def test_constellation_line_indices_in_range(self):
        n = len(_CATALOGUE)
        for abbr, a, b in _CONSTELLATION_LINES:
            assert 0 <= a < n, f"Index {a} out of range for {abbr}"
            assert 0 <= b < n, f"Index {b} out of range for {abbr}"

    def test_catalogue_magnitudes_are_numeric(self):
        for abbr, ra, dec, mag, name in _CATALOGUE:
            assert isinstance(mag, (int, float)), f"Bad magnitude for {name}"

    def test_catalogue_ra_in_range(self):
        for abbr, ra, dec, mag, name in _CATALOGUE:
            assert 0.0 <= ra < 360.0, f"RA out of range for {name}: {ra}"

    def test_catalogue_dec_in_range(self):
        for abbr, ra, dec, mag, name in _CATALOGUE:
            assert -90.0 <= dec <= 90.0, f"Dec out of range for {name}: {dec}"
