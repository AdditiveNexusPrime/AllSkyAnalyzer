"""
Configuration dataclasses and YAML loader for AllSkyAnalyzer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Sub-configuration sections
# ---------------------------------------------------------------------------


@dataclass
class IndiAllSkyConfig:
    """Settings for reading images from an indi-allsky installation."""

    image_dir: str = "/var/lib/indi-allsky/images"
    db_path: str = "/var/lib/indi-allsky/indi-allsky.db"
    file_pattern: str = "**/*.jpg"
    # Maximum age of images to process on startup (seconds).  0 = all.
    max_age_seconds: int = 0


@dataclass
class AstrometryConfig:
    """Settings for the astrometry / plate-solving step."""

    # Path to a local astrometry.net index directory (used by the built-in
    # solver).  Set to empty string to use the hosted API instead.
    index_dir: str = "/usr/share/astrometry"
    # Astrometry.net online API key (used when index_dir is empty).
    api_key: str = ""
    # Downsample factor applied before plate solving (speeds up solving).
    downsample: int = 2
    # Field-of-view hint in degrees (0 = auto-detect).
    fov_deg: float = 0.0
    # Magnitude limit for catalogue stars drawn on the overlay.
    magnitude_limit: float = 6.0


@dataclass
class AnomalyConfig:
    """Settings for the GPU-accelerated anomaly detection step."""

    # Sigma threshold for the z-score detector.
    z_score_threshold: float = 5.0
    # Minimum contour area (pixels²) for an anomaly candidate.
    min_area_px: int = 10
    # Maximum contour area (pixels²) – avoids flagging clouds.
    max_area_px: int = 5000
    # Enable CUDA acceleration (requires NVIDIA GPU).
    use_cuda: bool = True
    # CUDA device index (0-based).  Use -1 for CPU-only.
    cuda_device: int = 0


@dataclass
class OverlayConfig:
    """Settings for the output overlay image."""

    # Colour for catalogue stars (R, G, B, A) – values 0-255.
    star_colour: tuple = field(default_factory=lambda: (255, 255, 0, 200))
    # Colour for constellation lines.
    constellation_colour: tuple = field(default_factory=lambda: (0, 200, 255, 160))
    # Colour for anomaly markers.
    anomaly_colour: tuple = field(default_factory=lambda: (255, 50, 50, 220))
    # Circle radius drawn around each star (pixels).
    star_radius_px: int = 6
    # Font scale for labels (OpenCV convention).
    font_scale: float = 0.45
    # Draw star names?
    show_star_names: bool = True
    # Draw constellation names?
    show_constellation_names: bool = True
    # JPEG quality for saved overlay (1-100).
    jpeg_quality: int = 92


@dataclass
class OutputConfig:
    """Settings for result output."""

    directory: str = "output"
    # Save annotated overlay images.
    save_overlay: bool = True
    # Save a JSON file alongside each overlay with detected objects.
    save_json: bool = True
    # Write results to a SQLite database (useful for long-term trend analysis).
    save_db: bool = False
    db_path: str = "output/results.db"


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Top-level configuration for AllSkyAnalyzer."""

    indi_allsky: IndiAllSkyConfig = field(default_factory=IndiAllSkyConfig)
    astrometry: AstrometryConfig = field(default_factory=AstrometryConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load a Config from a YAML file, falling back to defaults for missing keys."""
        path = Path(path)
        if not path.exists():
            return cls()

        with open(path) as fh:
            raw: dict = yaml.safe_load(fh) or {}

        def _merge(dataclass_cls, raw_dict: dict):
            """Recursively build a dataclass instance from a raw dict."""
            import dataclasses
            import typing

            # Resolve string annotations (needed under `from __future__ import
            # annotations` and in Python 3.12+).
            try:
                type_hints = typing.get_type_hints(dataclass_cls)
            except Exception:
                type_hints = {}

            kwargs = {}
            for f in dataclasses.fields(dataclass_cls):
                value = raw_dict.get(f.name)
                if value is None:
                    continue
                # Recurse into nested dataclasses.
                resolved_type = type_hints.get(f.name, f.type)
                if dataclasses.is_dataclass(resolved_type):
                    kwargs[f.name] = _merge(resolved_type, value)
                else:
                    kwargs[f.name] = value
            return dataclass_cls(**kwargs)

        return _merge(cls, raw)

    def to_yaml(self, path: str | Path) -> None:
        """Serialise the configuration back to YAML."""
        import dataclasses

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(dataclasses.asdict(self), fh, default_flow_style=False)
