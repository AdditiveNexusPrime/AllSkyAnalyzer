"""
Web UI router for AllSkyAnalyzer.

Mounts Jinja2-rendered HTML pages into the existing FastAPI app:

  GET  /           → dashboard  (latest overlay + recent results)
  GET  /watchdog   → watchdog status page
  GET  /settings   → settings form
  POST /settings   → save settings (updates settings file + live config)
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .config import Config
from .settings_store import AllSkySettings, SSHSettings, SettingsStore
from .camera_watchdog import CameraWatchdog

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_web_router(
    config: Config,
    settings_store: SettingsStore,
    watchdog: CameraWatchdog,
) -> APIRouter:
    """Return an :class:`APIRouter` with all web UI routes attached."""

    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    router = APIRouter()

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _recent_overlays(n: int = 12) -> list[dict]:
        """Return metadata for the *n* most-recent overlay images."""
        out_dir = Path(config.output.directory)
        if not out_dir.is_dir():
            return []

        image_exts = {".jpg", ".jpeg", ".png"}
        candidates = [
            p for p in out_dir.iterdir()
            if p.is_file() and p.suffix.lower() in image_exts
            and "_overlay" in p.stem
        ]
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        results = []
        for p in candidates[:n]:
            json_path = p.with_name(p.stem.replace("_overlay", "_results") + ".json")
            meta: dict[str, Any] = {"filename": p.name}
            if json_path.exists():
                try:
                    with open(json_path) as fh:
                        meta.update(json.load(fh))
                except Exception:
                    pass
            meta["mtime"] = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)
            )
            results.append(meta)
        return results

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @router.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request) -> HTMLResponse:
        overlays = _recent_overlays()
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "overlays": overlays,
                "watchdog": watchdog.status.as_dict(),
            },
        )

    @router.get("/watchdog", response_class=HTMLResponse)
    async def watchdog_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            "watchdog.html",
            {
                "request": request,
                "status": watchdog.status.as_dict(),
                "config": dataclasses.asdict(config.watchdog),
            },
        )

    @router.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request, saved: str = "") -> HTMLResponse:
        current = settings_store.load()
        return templates.TemplateResponse(
            "settings.html",
            {
                "request": request,
                "settings": current,
                "cfg": dataclasses.asdict(config),
                "saved": saved == "1",
            },
        )

    @router.post("/settings")
    async def save_settings(
        request: Request,
        # Allsky path
        allsky_image_dir: str = Form(""),
        # SSH settings
        ssh_host: str = Form(""),
        ssh_username: str = Form(""),
        ssh_password: str = Form(""),
        ssh_port: int = Form(22),
        # Analysis config
        z_score_threshold: float = Form(5.0),
        magnitude_limit: float = Form(6.0),
        # Watchdog config
        check_interval_seconds: int = Form(120),
        stale_threshold_seconds: int = Form(300),
    ) -> RedirectResponse:
        # Persist SSH + allsky path (encrypted where appropriate).
        current = settings_store.load()
        new_settings = AllSkySettings(
            allsky_image_dir=allsky_image_dir.strip() or current.allsky_image_dir,
            ssh=SSHSettings(
                host=ssh_host.strip(),
                username=ssh_username.strip(),
                # Keep existing password when the form field is left blank.
                password=ssh_password if ssh_password else current.ssh.password,
                port=ssh_port,
            ),
        )
        settings_store.save(new_settings)

        # Apply live config changes (no restart needed for these).
        config.indi_allsky.image_dir = new_settings.allsky_image_dir
        config.anomaly.z_score_threshold = z_score_threshold
        config.astrometry.magnitude_limit = magnitude_limit
        config.watchdog.check_interval_seconds = check_interval_seconds
        config.watchdog.stale_threshold_seconds = stale_threshold_seconds

        # Persist changes back to the YAML config file.
        try:
            cfg_path = Path("config/config.yaml")
            config.to_yaml(cfg_path)
        except Exception as exc:
            logger.warning("Could not persist config to YAML: %s", exc)

        logger.info("Settings updated via web UI.")
        return RedirectResponse(url="/settings?saved=1", status_code=303)

    return router
