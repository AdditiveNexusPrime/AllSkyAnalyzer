"""Tests for the Config module."""

from __future__ import annotations

import dataclasses
import textwrap
from pathlib import Path

import pytest
import yaml

from src.config import (
    Config,
    IndiAllSkyConfig,
    AstrometryConfig,
    AnomalyConfig,
    OverlayConfig,
    OutputConfig,
)


class TestConfigDefaults:
    """Verify that default values are sensible."""

    def test_default_config_creates_without_error(self):
        cfg = Config()
        assert isinstance(cfg, Config)

    def test_indi_allsky_defaults(self):
        cfg = Config()
        assert cfg.indi_allsky.image_dir == "/var/lib/indi-allsky/images"
        assert cfg.indi_allsky.file_pattern == "**/*.jpg"

    def test_anomaly_defaults(self):
        cfg = Config()
        assert cfg.anomaly.z_score_threshold == 5.0
        assert cfg.anomaly.use_cuda is True

    def test_overlay_defaults(self):
        cfg = Config()
        assert cfg.overlay.star_radius_px == 6
        assert cfg.overlay.jpeg_quality == 92

    def test_output_defaults(self):
        cfg = Config()
        assert cfg.output.save_overlay is True
        assert cfg.output.save_json is True


class TestConfigFromYaml:
    """Test loading Config from a YAML file."""

    def test_from_nonexistent_yaml_returns_defaults(self, tmp_path):
        cfg = Config.from_yaml(tmp_path / "nonexistent.yaml")
        assert isinstance(cfg, Config)
        assert cfg.indi_allsky.image_dir == "/var/lib/indi-allsky/images"

    def test_from_yaml_overrides_values(self, tmp_path):
        content = textwrap.dedent("""\
            indi_allsky:
              image_dir: /my/custom/images
              max_age_seconds: 3600
            anomaly:
              z_score_threshold: 7.5
            output:
              directory: /my/output
        """)
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(content)

        cfg = Config.from_yaml(yaml_file)
        assert cfg.indi_allsky.image_dir == "/my/custom/images"
        assert cfg.indi_allsky.max_age_seconds == 3600
        assert cfg.anomaly.z_score_threshold == 7.5
        assert cfg.output.directory == "/my/output"

    def test_from_yaml_preserves_defaults_for_missing_keys(self, tmp_path):
        content = "indi_allsky:\n  image_dir: /custom\n"
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(content)

        cfg = Config.from_yaml(yaml_file)
        # Unspecified keys should still have defaults.
        assert cfg.anomaly.z_score_threshold == 5.0
        assert cfg.overlay.jpeg_quality == 92


class TestConfigToYaml:
    """Test round-tripping Config through YAML."""

    def test_to_yaml_creates_file(self, tmp_path):
        cfg = Config()
        out_path = tmp_path / "out" / "config.yaml"
        cfg.to_yaml(out_path)
        assert out_path.exists()

    def test_to_yaml_roundtrip(self, tmp_path):
        cfg = Config()
        cfg.indi_allsky.image_dir = "/roundtrip/test"
        cfg.anomaly.z_score_threshold = 9.9

        yaml_path = tmp_path / "config.yaml"
        cfg.to_yaml(yaml_path)

        loaded = Config.from_yaml(yaml_path)
        assert loaded.indi_allsky.image_dir == "/roundtrip/test"
        assert loaded.anomaly.z_score_threshold == 9.9
