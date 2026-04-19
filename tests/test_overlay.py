"""Tests for the OverlayRenderer module."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.config import OverlayConfig
from src.overlay import OverlayRenderer
from src.star_finder import StarFinderResult, Star, ConstellationLine
from src.anomaly_detector import Anomaly


def _blank_image(h: int = 300, w: int = 300) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _sample_star_result(plate_solved: bool = False) -> StarFinderResult:
    result = StarFinderResult(plate_solved=plate_solved)
    result.stars = [
        Star(x=100, y=100, brightness=200, name="Sirius", constellation="Canis Major"),
        Star(x=200, y=150, brightness=180),
    ]
    if plate_solved:
        result.constellation_lines = [
            ConstellationLine(x1=100, y1=100, x2=200, y2=150, constellation="Canis Major"),
        ]
    return result


def _sample_anomalies() -> list[Anomaly]:
    return [
        Anomaly(x=50, y=50, width=20, height=20, cx=60, cy=60, confidence=7.5, label="point"),
    ]


class TestOverlayRendererFromArray:
    def _renderer(self) -> OverlayRenderer:
        return OverlayRenderer(OverlayConfig())

    def test_returns_ndarray_of_same_shape(self):
        renderer = self._renderer()
        bgr = _blank_image()
        star_result = _sample_star_result()
        anomalies = _sample_anomalies()
        out = renderer.render_from_array(bgr, star_result, anomalies)
        assert isinstance(out, np.ndarray)
        assert out.shape == bgr.shape

    def test_output_differs_from_input(self):
        """Rendering should modify the image (stars / anomaly boxes drawn)."""
        renderer = self._renderer()
        bgr = _blank_image()
        star_result = _sample_star_result()
        anomalies = _sample_anomalies()
        out = renderer.render_from_array(bgr, star_result, anomalies)
        assert not np.array_equal(out, bgr)

    def test_does_not_modify_input(self):
        """The original array should be untouched."""
        renderer = self._renderer()
        bgr = _blank_image()
        original = bgr.copy()
        renderer.render_from_array(bgr, _sample_star_result(), _sample_anomalies())
        assert np.array_equal(bgr, original)

    def test_plate_solved_draws_constellation_lines(self):
        renderer = self._renderer()
        bgr = _blank_image()
        star_result = _sample_star_result(plate_solved=True)
        out = renderer.render_from_array(bgr, star_result, [])
        # With constellation lines drawn the image should be modified.
        assert not np.array_equal(out, bgr)

    def test_empty_stars_and_anomalies(self):
        """Should handle empty lists gracefully."""
        renderer = self._renderer()
        bgr = _blank_image()
        star_result = StarFinderResult()
        out = renderer.render_from_array(bgr, star_result, [])
        assert isinstance(out, np.ndarray)


class TestOverlayRendererRenderToFile:
    def _renderer(self) -> OverlayRenderer:
        return OverlayRenderer(OverlayConfig())

    def _save_test_image(self, tmp_path: Path) -> Path:
        img_path = tmp_path / "input.jpg"
        cv2.imwrite(str(img_path), _blank_image())
        return img_path

    def test_saves_output_file(self, tmp_path):
        renderer = self._renderer()
        img_path = self._save_test_image(tmp_path)
        out_path = tmp_path / "overlay.jpg"
        result_path = renderer.render(
            img_path, _sample_star_result(), _sample_anomalies(), out_path
        )
        assert result_path.exists()

    def test_raises_for_nonexistent_input(self, tmp_path):
        renderer = self._renderer()
        with pytest.raises(FileNotFoundError):
            renderer.render(
                tmp_path / "nope.jpg",
                _sample_star_result(), [],
                tmp_path / "out.jpg",
            )

    def test_creates_parent_directories(self, tmp_path):
        renderer = self._renderer()
        img_path = self._save_test_image(tmp_path)
        nested_out = tmp_path / "deep" / "nested" / "overlay.jpg"
        renderer.render(img_path, _sample_star_result(), [], nested_out)
        assert nested_out.exists()

    def test_png_output_format(self, tmp_path):
        renderer = self._renderer()
        img_path = self._save_test_image(tmp_path)
        out_path = tmp_path / "overlay.png"
        renderer.render(img_path, _sample_star_result(), [], out_path)
        assert out_path.exists()
        # Verify it can be re-read as a valid PNG.
        loaded = cv2.imread(str(out_path))
        assert loaded is not None
