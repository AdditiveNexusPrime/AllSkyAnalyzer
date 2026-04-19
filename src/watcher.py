"""
ImageWatcher – monitor the indi-allsky image directory and trigger analysis
whenever a new image file is written.

Uses the ``watchdog`` library for cross-platform filesystem event monitoring.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from watchdog.events import FileCreatedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .config import Config
from .analyzer import AllSkyAnalyzer

logger = logging.getLogger(__name__)

# Extensions that are considered valid allsky images.
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".fits"}


class _NewImageHandler(FileSystemEventHandler):
    """Watchdog event handler that analyses each newly created image."""

    def __init__(self, analyzer: AllSkyAnalyzer) -> None:
        super().__init__()
        self.analyzer = analyzer

    def on_created(self, event: FileCreatedEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            return
        logger.info("New image detected: %s", path)
        try:
            self.analyzer.analyze_image(path)
        except Exception as exc:
            logger.error("Analysis failed for %s: %s", path, exc)


class ImageWatcher:
    """Watch the configured indi-allsky image directory for new images.

    Parameters
    ----------
    config:
        Application configuration object.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.analyzer = AllSkyAnalyzer(config)
        self._observer: Observer | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Block and watch until interrupted (e.g. KeyboardInterrupt)."""
        watch_dir = Path(self.config.indi_allsky.image_dir)
        watch_dir.mkdir(parents=True, exist_ok=True)

        handler = _NewImageHandler(self.analyzer)
        self._observer = Observer()
        self._observer.schedule(handler, str(watch_dir), recursive=True)
        self._observer.start()

        logger.info("Watching %s …", watch_dir)
        try:
            while self._observer.is_alive():
                time.sleep(1)
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the watcher."""
        if self._observer and self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
            logger.info("Watcher stopped.")
