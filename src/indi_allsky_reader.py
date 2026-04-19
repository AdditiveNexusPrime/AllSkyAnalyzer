"""
IndiAllSkyReader – discovers and reads images written by indi-allsky.

indi-allsky stores captured images in a configurable base directory, organised
by camera / date subdirectories.  This module provides an iterator over those
images so that the rest of the pipeline can remain independent of the specific
directory layout.

Reference: https://github.com/aaronwmorris/indi-allsky
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Generator

from .config import IndiAllSkyConfig

logger = logging.getLogger(__name__)


class IndiAllSkyReader:
    """Discover allsky images produced by an indi-allsky installation.

    Parameters
    ----------
    config:
        The indi-allsky section of the application config.
    """

    def __init__(self, config: IndiAllSkyConfig) -> None:
        self.config = config
        self.image_dir = Path(config.image_dir)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def iter_images(self) -> Generator[Path, None, None]:
        """Yield :class:`pathlib.Path` objects for every matching image.

        Images are yielded in ascending modification-time order.  If
        ``max_age_seconds`` is configured only images newer than that
        threshold are yielded.
        """
        if not self.image_dir.exists():
            logger.warning("indi-allsky image directory not found: %s", self.image_dir)
            return

        cutoff = (
            time.time() - self.config.max_age_seconds
            if self.config.max_age_seconds > 0
            else 0.0
        )

        candidates: list[Path] = sorted(
            self.image_dir.glob(self.config.file_pattern),
            key=lambda p: p.stat().st_mtime,
        )

        for path in candidates:
            if cutoff and path.stat().st_mtime < cutoff:
                continue
            if not self._is_valid_image(path):
                continue
            logger.debug("Found image: %s", path)
            yield path

    def latest_image(self) -> Path | None:
        """Return the most recently modified image, or *None* if none found."""
        images = list(self.iter_images())
        return images[-1] if images else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_valid_image(path: Path) -> bool:
        """Return True if the file looks like a readable image."""
        return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".fits"}
