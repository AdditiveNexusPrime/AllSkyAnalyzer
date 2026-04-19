"""Tests for the AnomalyDetector module."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import AnomalyConfig
from src.anomaly_detector import AnomalyDetector, Anomaly


def _make_test_image(height: int = 200, width: int = 200) -> np.ndarray:
    """Create a dark BGR image with a bright spot injected (7x7 region)."""
    bgr = np.zeros((height, width, 3), dtype=np.uint8)
    # Inject a bright filled square – large enough to survive morphological ops
    # and register a contour area well above any min_area_px threshold.
    bgr[97:104, 97:104] = (255, 255, 255)
    return bgr


class TestAnomalyDetectorDetectFromArray:
    def _detector(self, **kwargs) -> AnomalyDetector:
        cfg = AnomalyConfig(use_cuda=False, **kwargs)
        return AnomalyDetector(cfg)

    def test_finds_bright_spot(self):
        det = self._detector(z_score_threshold=3.0, min_area_px=1, max_area_px=50)
        bgr = _make_test_image()
        anomalies = det.detect_from_array(bgr)
        assert len(anomalies) > 0

    def test_no_anomalies_on_uniform_image(self):
        det = self._detector(z_score_threshold=5.0, min_area_px=10)
        # Perfectly uniform image → no anomalies.
        bgr = np.full((200, 200, 3), 100, dtype=np.uint8)
        anomalies = det.detect_from_array(bgr)
        assert anomalies == []

    def test_anomaly_fields_are_populated(self):
        det = self._detector(z_score_threshold=3.0, min_area_px=1, max_area_px=50)
        bgr = _make_test_image()
        anomalies = det.detect_from_array(bgr)
        assert len(anomalies) > 0
        a = anomalies[0]
        assert a.width > 0
        assert a.height > 0
        assert a.confidence > 0
        assert a.label in {"streak", "point", "diffuse"}

    def test_sorted_by_confidence_descending(self):
        # Build an image with two bright spots of different intensity.
        bgr = np.zeros((200, 200, 3), dtype=np.uint8)
        # Bright spot 1 (brighter).
        bgr[45:52, 45:52] = (255, 255, 255)
        # Dimmer spot 2.
        bgr[145:152, 145:152] = (180, 180, 180)

        det = self._detector(z_score_threshold=3.0, min_area_px=1, max_area_px=50)
        anomalies = det.detect_from_array(bgr)
        confidences = [a.confidence for a in anomalies]
        assert confidences == sorted(confidences, reverse=True)

    def test_area_filter_excludes_large_region(self):
        # A large bright region (e.g., cloud / moon) should be excluded.
        bgr = np.zeros((300, 300, 3), dtype=np.uint8)
        bgr[50:250, 50:250] = 200  # 200×200 = 40 000 px² region
        det = self._detector(
            z_score_threshold=3.0, min_area_px=10, max_area_px=5000
        )
        anomalies = det.detect_from_array(bgr)
        assert all(a.width * a.height <= 5000 for a in anomalies)


class TestAnomalyClassification:
    def test_streak_classification(self):
        det = AnomalyDetector(AnomalyConfig(use_cuda=False))
        label = det._classify(w=100, h=5, aspect=20.0)
        assert label == "streak"

    def test_point_classification(self):
        det = AnomalyDetector(AnomalyConfig(use_cuda=False))
        label = det._classify(w=5, h=5, aspect=1.0)
        assert label == "point"

    def test_diffuse_classification(self):
        det = AnomalyDetector(AnomalyConfig(use_cuda=False))
        label = det._classify(w=50, h=50, aspect=1.0)
        assert label == "diffuse"


class TestAnomalyToDict:
    def test_to_dict_keys(self):
        a = Anomaly(x=10, y=20, width=5, height=5, confidence=8.5, label="point")
        d = a.to_dict()
        assert set(d.keys()) == {
            "x", "y", "width", "height", "cx", "cy",
            "confidence", "aspect_ratio", "label",
        }
        assert d["label"] == "point"
        assert d["confidence"] == 8.5
