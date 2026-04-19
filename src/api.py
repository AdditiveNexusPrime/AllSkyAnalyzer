"""
REST API for AllSkyAnalyzer, built with FastAPI.

Endpoints
---------
GET  /health           – liveness probe
GET  /config           – return current configuration as JSON
POST /analyze          – upload an image and receive analysis results
GET  /latest           – analyze the most-recent indi-allsky image
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .config import Config
from .analyzer import AllSkyAnalyzer

logger = logging.getLogger(__name__)


def create_app(config: Config) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AllSkyAnalyzer",
        description="Analyze allsky camera images for stars, constellations, and anomalies.",
        version="1.0.0",
    )
    analyzer = AllSkyAnalyzer(config)

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/config")
    def get_config() -> dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(config)

    @app.post("/analyze")
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

    @app.get("/latest")
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

    @app.get("/overlay/{filename}")
    def get_overlay(filename: str) -> FileResponse:
        """Return a previously generated overlay image by filename.

        The response path is always taken from a directory listing of the
        output directory, never constructed from user-supplied data, which
        prevents path-traversal attacks.
        """
        import re

        out_dir = Path(config.output.directory).resolve()

        # Allowlist: only plain filenames (letters, digits, dash, underscore,
        # dot) are accepted – no path separators or relative components.
        if not re.fullmatch(r"[\w.\-]+", filename):
            raise HTTPException(status_code=400, detail="Invalid filename.")

        # Iterate the output directory and return the first entry whose name
        # exactly matches the requested filename.  The path is derived from
        # the directory listing, not from the user-supplied value.
        if out_dir.is_dir():
            for candidate in out_dir.iterdir():
                if candidate.name == filename and candidate.is_file():
                    return FileResponse(str(candidate))

        raise HTTPException(status_code=404, detail="Overlay not found.")

    return app
