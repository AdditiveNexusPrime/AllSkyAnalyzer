"""
AllSkyAnalyzer – main analysis orchestrator.

Ties together IndiAllSkyReader, StarFinder, AnomalyDetector, and
OverlayRenderer into a single easy-to-use facade.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from .config import Config
from .indi_allsky_reader import IndiAllSkyReader
from .star_finder import StarFinder, StarFinderResult, Star
from .anomaly_detector import AnomalyDetector, Anomaly
from .overlay import OverlayRenderer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------


@dataclass
class AnalysisResult:
    """All outputs from analysing a single allsky image."""

    image_path: str
    stars: list[Star] = field(default_factory=list)
    anomalies: list[Anomaly] = field(default_factory=list)
    plate_solved: bool = False
    overlay_path: Optional[str] = None
    json_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "plate_solved": self.plate_solved,
            "star_count": len(self.stars),
            "anomaly_count": len(self.anomalies),
            "stars": [
                {
                    "x": s.x, "y": s.y,
                    "brightness": s.brightness,
                    "ra": s.ra, "dec": s.dec,
                    "magnitude": s.magnitude,
                    "name": s.name,
                    "constellation": s.constellation,
                }
                for s in self.stars
            ],
            "anomalies": [a.to_dict() for a in self.anomalies],
            "overlay_path": self.overlay_path,
        }


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------


class AllSkyAnalyzer:
    """Orchestrate the full analysis pipeline for allsky images.

    Parameters
    ----------
    config:
        Application configuration object.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.reader = IndiAllSkyReader(config.indi_allsky)
        self.star_finder = StarFinder(config.astrometry)
        self.anomaly_detector = AnomalyDetector(config.anomaly)
        self.overlay_renderer = OverlayRenderer(config.overlay)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_image(self, image_path: str | Path) -> AnalysisResult:
        """Run the full pipeline on a single image.

        Parameters
        ----------
        image_path:
            Path to the allsky image file.

        Returns
        -------
        AnalysisResult
            Containing detected stars, anomalies, and paths to any saved files.
        """
        image_path = Path(image_path)
        logger.info("Analysing: %s", image_path)

        result = AnalysisResult(image_path=str(image_path))

        # 1. Find stars / plate-solve.
        star_result: StarFinderResult = self.star_finder.find(image_path)
        result.stars = star_result.stars
        result.plate_solved = star_result.plate_solved

        # 2. Detect anomalies.
        result.anomalies = self.anomaly_detector.detect(image_path)

        # 3. Render and save overlay.
        if self.config.output.save_overlay:
            overlay_path = self._build_output_path(image_path, suffix="_overlay")
            try:
                self.overlay_renderer.render(
                    image_path, star_result, result.anomalies, overlay_path
                )
                result.overlay_path = str(overlay_path)
            except Exception as exc:
                logger.error("Overlay rendering failed: %s", exc)

        # 4. Save JSON sidecar.
        if self.config.output.save_json:
            json_path = self._build_output_path(image_path, suffix="_results", ext=".json")
            self._save_json(result, json_path)
            result.json_path = str(json_path)

        logger.info(
            "Done – stars=%d  anomalies=%d  overlay=%s",
            len(result.stars), len(result.anomalies), result.overlay_path,
        )
        return result

    def analyze_all(self) -> list[AnalysisResult]:
        """Analyze all images found by the IndiAllSkyReader.

        Returns a list of results, one per image.
        """
        results: list[AnalysisResult] = []
        for image_path in self.reader.iter_images():
            try:
                results.append(self.analyze_image(image_path))
            except Exception as exc:
                logger.error("Failed to analyze %s: %s", image_path, exc)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_output_path(
        self,
        source: Path,
        suffix: str = "",
        ext: Optional[str] = None,
    ) -> Path:
        """Build an output file path in the configured output directory."""
        out_dir = Path(self.config.output.directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = source.stem + suffix
        extension = ext if ext is not None else source.suffix
        return out_dir / (stem + extension)

    @staticmethod
    def _save_json(result: AnalysisResult, path: Path) -> None:
        """Serialise *result* to a JSON sidecar file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(result.to_dict(), fh, indent=2)
        logger.debug("JSON saved: %s", path)
