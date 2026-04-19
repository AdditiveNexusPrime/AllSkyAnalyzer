"""
AnomalyDetector – detect unusual features (meteors, satellites, UFOs, …)
in allsky camera images.

The detector uses a statistical z-score approach on the luminance channel:
pixels that are dramatically brighter than their local neighbourhood are
candidate anomalies.  GPU acceleration is used when CUDA is available.

Pipeline
--------
1. Convert to grayscale / luminance.
2. Apply a Gaussian blur to produce a smooth "background" estimate.
3. Compute the per-pixel residual between the raw image and the background.
4. Threshold at *z_score_threshold* standard deviations above the mean.
5. Find contours; filter by area.
6. Return bounding boxes + confidence scores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import AnomalyConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Anomaly:
    """A single detected anomaly region."""

    # Bounding box in pixel coordinates.
    x: int
    y: int
    width: int
    height: int
    # Centroid.
    cx: float = 0.0
    cy: float = 0.0
    # How many sigma above the background the peak pixel is.
    confidence: float = 0.0
    # Aspect ratio of the bounding box (width / height).
    aspect_ratio: float = 1.0
    # Rough classification hint.
    label: str = "anomaly"

    def to_dict(self) -> dict:
        return {
            "x": self.x, "y": self.y,
            "width": self.width, "height": self.height,
            "cx": self.cx, "cy": self.cy,
            "confidence": round(self.confidence, 2),
            "aspect_ratio": round(self.aspect_ratio, 2),
            "label": self.label,
        }


# ---------------------------------------------------------------------------
# AnomalyDetector
# ---------------------------------------------------------------------------


class AnomalyDetector:
    """Detect anomalous bright regions in an allsky image.

    Parameters
    ----------
    config:
        The anomaly section of the application config.
    """

    # Kernel size for background estimation Gaussian blur (must be odd).
    _BG_KERNEL: int = 51

    def __init__(self, config: AnomalyConfig) -> None:
        self.config = config
        self._use_cuda = config.use_cuda and self._cuda_available()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image_path: str | Path) -> list[Anomaly]:
        """Detect anomalies in *image_path*.

        Returns a (possibly empty) list of :class:`Anomaly` objects sorted by
        descending confidence.
        """
        image_path = Path(image_path)
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            logger.error("Cannot read image: %s", image_path)
            return []

        return self.detect_from_array(bgr)

    def detect_from_array(self, bgr: np.ndarray) -> list[Anomaly]:
        """Run detection on a BGR NumPy array (height × width × 3)."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        residual = self._compute_residual(gray)
        mask = self._threshold_mask(residual)
        anomalies = self._find_anomalies(mask, residual)

        logger.info("Anomaly detector found %d candidates", len(anomalies))
        return anomalies

    # ------------------------------------------------------------------
    # Internal pipeline steps
    # ------------------------------------------------------------------

    def _compute_residual(self, gray: np.ndarray) -> np.ndarray:
        """Return the difference between the raw image and a smoothed background."""
        k = self._BG_KERNEL

        if self._use_cuda:
            residual = self._compute_residual_cuda(gray, k)
            if residual is not None:
                return residual
            logger.debug("CUDA residual failed; falling back to CPU.")

        # CPU path.
        background = cv2.GaussianBlur(gray, (k, k), 0)
        return np.clip(gray - background, 0, None)

    def _compute_residual_cuda(self, gray: np.ndarray, k: int) -> Optional[np.ndarray]:
        """GPU-accelerated background subtraction using OpenCV CUDA module."""
        try:
            gpu_gray = cv2.cuda_GpuMat()
            gpu_gray.upload(gray)

            gpu_filter = cv2.cuda.createGaussianFilter(
                cv2.CV_32F, cv2.CV_32F, (k, k), 0
            )
            gpu_bg = gpu_filter.apply(gpu_gray)
            gpu_residual = cv2.cuda.subtract(gpu_gray, gpu_bg)

            residual = gpu_residual.download()
            return np.clip(residual, 0, None)
        except cv2.error as exc:
            logger.debug("CUDA residual error: %s", exc)
            return None

    def _threshold_mask(self, residual: np.ndarray) -> np.ndarray:
        """Create a binary mask of pixels above the z-score threshold."""
        mean = float(np.mean(residual))
        std = float(np.std(residual))
        if std < 1e-6:
            return np.zeros(residual.shape, dtype=np.uint8)

        threshold = mean + self.config.z_score_threshold * std
        mask = (residual > threshold).astype(np.uint8) * 255

        # Morphological clean-up to merge nearby pixels.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _find_anomalies(
        self, mask: np.ndarray, residual: np.ndarray
    ) -> list[Anomaly]:
        """Extract anomaly bounding boxes from the binary mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        anomalies: list[Anomaly] = []

        mean_res = float(np.mean(residual))
        std_res = float(np.std(residual))

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_area_px or area > self.config.max_area_px:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w / 2
            cy = y + h / 2

            # Confidence = peak residual value in sigma.
            roi = residual[y : y + h, x : x + w]
            peak = float(np.max(roi))
            confidence = (peak - mean_res) / std_res if std_res > 1e-6 else 0.0

            aspect = w / h if h > 0 else 1.0
            label = self._classify(w, h, aspect)

            anomalies.append(
                Anomaly(
                    x=x, y=y, width=w, height=h,
                    cx=cx, cy=cy,
                    confidence=confidence,
                    aspect_ratio=aspect,
                    label=label,
                )
            )

        anomalies.sort(key=lambda a: a.confidence, reverse=True)
        return anomalies

    @staticmethod
    def _classify(w: int, h: int, aspect: float) -> str:
        """Heuristically classify an anomaly by its shape."""
        if aspect > 4 or aspect < 0.25:
            return "streak"  # likely a meteor or satellite trail
        if w < 20 and h < 20:
            return "point"   # satellite flash or bright star artefact
        return "diffuse"     # cloud patch or lens artefact

    # ------------------------------------------------------------------
    # CUDA availability check
    # ------------------------------------------------------------------

    @staticmethod
    def _cuda_available() -> bool:
        """Return True if OpenCV was compiled with CUDA support."""
        try:
            count = cv2.cuda.getCudaEnabledDeviceCount()
            available = count > 0
            if available:
                logger.debug("CUDA device count: %d", count)
            else:
                logger.debug("No CUDA devices found via OpenCV.")
            return available
        except (AttributeError, cv2.error):
            return False
