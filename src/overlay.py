"""
OverlayRenderer – draw detected stars, constellation lines, and anomaly
markers onto an allsky image and save the annotated result.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import OverlayConfig
from .star_finder import Star, ConstellationLine, StarFinderResult
from .anomaly_detector import Anomaly

logger = logging.getLogger(__name__)


class OverlayRenderer:
    """Render a fully-annotated overlay image.

    Parameters
    ----------
    config:
        The overlay section of the application config.
    """

    def __init__(self, config: OverlayConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        image_path: str | Path,
        star_result: StarFinderResult,
        anomalies: list[Anomaly],
        output_path: str | Path,
    ) -> Path:
        """
        Load *image_path*, draw all detections, and save to *output_path*.

        Returns the resolved output path.
        """
        image_path = Path(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        overlay = self._draw_overlay(bgr, star_result, anomalies)
        self._save(overlay, output_path)
        logger.info("Overlay saved: %s", output_path)
        return output_path

    def render_from_array(
        self,
        bgr: np.ndarray,
        star_result: StarFinderResult,
        anomalies: list[Anomaly],
    ) -> np.ndarray:
        """Draw all detections onto *bgr* (in-place copy) and return the result."""
        return self._draw_overlay(bgr.copy(), star_result, anomalies)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_overlay(
        self,
        bgr: np.ndarray,
        star_result: StarFinderResult,
        anomalies: list[Anomaly],
    ) -> np.ndarray:
        """Compose all annotation layers onto *bgr* and return the result."""
        # Work on a copy so we can alpha-blend layers.
        canvas = bgr.copy()

        if star_result.plate_solved:
            self._draw_constellation_lines(canvas, star_result.constellation_lines)

        self._draw_stars(canvas, star_result.stars)
        self._draw_anomalies(canvas, anomalies)
        self._draw_legend(canvas, star_result, anomalies)

        return canvas

    def _draw_constellation_lines(
        self, canvas: np.ndarray, lines: list[ConstellationLine]
    ) -> None:
        """Draw faint lines connecting constellation stars."""
        r, g, b, a = self.config.constellation_colour
        colour = (b, g, r)  # OpenCV uses BGR
        alpha = a / 255.0

        line_layer = canvas.copy()
        drawn_labels: set[str] = set()

        for line in lines:
            pt1 = (int(line.x1), int(line.y1))
            pt2 = (int(line.x2), int(line.y2))
            cv2.line(line_layer, pt1, pt2, colour, 1, cv2.LINE_AA)

            # Label the constellation once near the midpoint.
            if (
                self.config.show_constellation_names
                and line.constellation not in drawn_labels
            ):
                mx = int((line.x1 + line.x2) / 2)
                my = int((line.y1 + line.y2) / 2)
                cv2.putText(
                    line_layer, line.constellation, (mx + 4, my - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale,
                    colour, 1, cv2.LINE_AA,
                )
                drawn_labels.add(line.constellation)

        cv2.addWeighted(line_layer, alpha, canvas, 1 - alpha, 0, canvas)

    def _draw_stars(self, canvas: np.ndarray, stars: list[Star]) -> None:
        """Draw circles and optional name labels around detected stars."""
        r, g, b, a = self.config.star_colour
        colour = (b, g, r)
        alpha = a / 255.0
        radius = self.config.star_radius_px
        height, width = canvas.shape[:2]

        star_layer = canvas.copy()
        for star in stars:
            sx, sy = int(star.x), int(star.y)
            if sx < 0 or sy < 0 or sx >= width or sy >= height:
                continue

            # Larger circle for named (catalogue) stars.
            r_px = radius + (2 if star.name else 0)
            cv2.circle(star_layer, (sx, sy), r_px, colour, 1, cv2.LINE_AA)

            if self.config.show_star_names and star.name:
                cv2.putText(
                    star_layer, star.name, (sx + r_px + 2, sy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale,
                    colour, 1, cv2.LINE_AA,
                )

        cv2.addWeighted(star_layer, alpha, canvas, 1 - alpha, 0, canvas)

    def _draw_anomalies(self, canvas: np.ndarray, anomalies: list[Anomaly]) -> None:
        """Draw red bounding boxes around anomaly candidates."""
        r, g, b, a = self.config.anomaly_colour
        colour = (b, g, r)
        alpha = a / 255.0

        anomaly_layer = canvas.copy()
        for anom in anomalies:
            pt1 = (anom.x, anom.y)
            pt2 = (anom.x + anom.width, anom.y + anom.height)
            cv2.rectangle(anomaly_layer, pt1, pt2, colour, 2)

            label = f"{anom.label} {anom.confidence:.1f}σ"
            cv2.putText(
                anomaly_layer, label,
                (anom.x, max(anom.y - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale,
                colour, 1, cv2.LINE_AA,
            )

        cv2.addWeighted(anomaly_layer, alpha, canvas, 1 - alpha, 0, canvas)

    def _draw_legend(
        self,
        canvas: np.ndarray,
        star_result: StarFinderResult,
        anomalies: list[Anomaly],
    ) -> None:
        """Draw a small summary legend in the bottom-left corner."""
        height, width = canvas.shape[:2]
        lines = [
            f"Stars: {len(star_result.stars)}",
            f"Anomalies: {len(anomalies)}",
            "Plate solved: YES" if star_result.plate_solved else "Plate solved: NO",
        ]

        x, y = 10, height - 10 - len(lines) * 18
        bg_pt1 = (x - 4, y - 14)
        bg_pt2 = (x + 180, height - 6)
        cv2.rectangle(canvas, bg_pt1, bg_pt2, (0, 0, 0), -1)

        for i, text in enumerate(lines):
            cv2.putText(
                canvas, text, (x, y + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (220, 220, 220), 1, cv2.LINE_AA,
            )

    # ------------------------------------------------------------------
    # Save helper
    # ------------------------------------------------------------------

    def _save(self, image: np.ndarray, path: Path) -> None:
        """Write *image* to *path*, choosing format from the extension."""
        ext = path.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            params = [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
        elif ext == ".png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
        else:
            params = []

        success = cv2.imwrite(str(path), image, params)
        if not success:
            raise IOError(f"cv2.imwrite failed for path: {path}")
