"""
REST API + Web UI for AllSkyAnalyzer, built with FastAPI.

JSON API endpoints
------------------
GET  /api/health           – liveness probe
GET  /api/config           – return current configuration as JSON
POST /api/analyze          – upload an image and receive analysis results
GET  /api/latest           – analyze the most-recent indi-allsky image
GET  /api/overlay/{file}   – serve a previously generated overlay image
GET  /api/watchdog/status  – watchdog status as JSON

Web UI pages (HTML)
-------------------
GET  /                     – dashboard
GET  /watchdog             – watchdog status page
GET  /settings             – settings form
POST /settings             – save settings
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Config
from .analyzer import AllSkyAnalyzer
from .settings_store import SettingsStore
from .camera_watchdog import CameraWatchdog
from .web_ui import create_web_router

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


def create_app(config: Config) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AllSkyAnalyzer",
        description="Analyze allsky camera images for stars, constellations, and anomalies.",
        version="1.0.0",
    )

    # ------------------------------------------------------------------
    # Shared services
    # ------------------------------------------------------------------
    analyzer = AllSkyAnalyzer(config)

    settings_store = SettingsStore(data_dir=config.watchdog.data_dir)

    watchdog = CameraWatchdog(
        config=config.watchdog,
        settings_store=settings_store,
        image_dir=config.indi_allsky.image_dir,
    )
    watchdog.start()

    # ------------------------------------------------------------------
    # Static files
    # ------------------------------------------------------------------
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # ------------------------------------------------------------------
    # Web UI routes (HTML pages)
    # ------------------------------------------------------------------
    web_router = create_web_router(config, settings_store, watchdog)
    app.include_router(web_router)

    # ------------------------------------------------------------------
    # JSON API routes
    # ------------------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/config")
    def get_config() -> dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(config)

    @app.post("/api/analyze")
    async def analyze_upload(file: UploadFile = File(...)) -> JSONResponse:
        """Upload an image file and receive analysis results as JSON."""
        suffix = Path(file.filename or "image.jpg").suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = Path(tmp.name)

        try:
            result = analyzer.analyze_image(tmp_path)
            return JSONResponse(content=result.to_dict())
        except Exception as exc:
            logger.exception("Analysis error for uploaded file.")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            tmp_path.unlink(missing_ok=True)

    @app.get("/api/latest")
    def analyze_latest() -> JSONResponse:
        """Analyze the most recently captured indi-allsky image."""
        latest = analyzer.reader.latest_image()
        if latest is None:
            raise HTTPException(
                status_code=404,
                detail="No images found in the configured indi-allsky directory.",
            )
        try:
            result = analyzer.analyze_image(latest)
            return JSONResponse(content=result.to_dict())
        except Exception as exc:
            logger.exception("Analysis error for latest image.")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/overlay/{filename}")
    def get_overlay(filename: str) -> FileResponse:
        """Return a previously generated overlay image by filename.

        The response path is always taken from a directory listing of the
        output directory, never constructed from user-supplied data, which
        prevents path-traversal attacks.
        """
        import re

        out_dir = Path(config.output.directory).resolve()

        if not re.fullmatch(r"[\w.\-]+", filename):
            raise HTTPException(status_code=400, detail="Invalid filename.")

        if out_dir.is_dir():
            for candidate in out_dir.iterdir():
                if candidate.name == filename and candidate.is_file():
                    return FileResponse(str(candidate))

        raise HTTPException(status_code=404, detail="Overlay not found.")

    @app.get("/api/watchdog/status")
    def watchdog_status() -> JSONResponse:
        """Return current watchdog status as JSON."""
        return JSONResponse(content=watchdog.status.as_dict())

    # ------------------------------------------------------------------
    # Shutdown hook
    # ------------------------------------------------------------------
    @app.on_event("shutdown")
    def _shutdown() -> None:
        watchdog.stop()

    return app
