"""
SettingsStore – encrypted persistence for sensitive run-time settings.

Sensitive values (SSH username, password) are encrypted at rest using
Fernet symmetric encryption (AES-128-CBC with PKCS7 padding + HMAC-SHA256).
The encryption key is auto-generated on first use and stored in
``<data_dir>/secret.key``.  Settings are stored as a JSON file at
``<data_dir>/settings.json``.

Non-sensitive settings (allsky image directory, SSH hostname) are stored
as plain text in the same JSON file for readability.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SETTINGS_FILE = "settings.json"
_KEY_FILE = "secret.key"

# Sentinel stored in the JSON when a field has never been set.
_UNSET = ""


@dataclass
class SSHSettings:
    """SSH connection details for the indi-allsky host."""

    host: str = ""
    username: str = ""
    # The password is stored as an empty string when unset.
    # It is always encrypted before being written to disk.
    password: str = ""
    port: int = 22


@dataclass
class AllSkySettings:
    """Editable run-time settings exposed through the web UI."""

    # Path on the *local* filesystem (or mounted volume) where indi-allsky
    # writes images.
    allsky_image_dir: str = "/var/lib/indi-allsky/images"
    ssh: SSHSettings = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.ssh is None:
            self.ssh = SSHSettings()


class SettingsStore:
    """Load and persist :class:`AllSkySettings` with encrypted SSH credentials.

    Parameters
    ----------
    data_dir:
        Directory where ``secret.key`` and ``settings.json`` are stored.
        Created automatically if it does not exist.
    """

    def __init__(self, data_dir: str | Path) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._key_path = self._data_dir / _KEY_FILE
        self._settings_path = self._data_dir / _SETTINGS_FILE
        self._fernet = self._load_or_create_fernet()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> AllSkySettings:
        """Return the current settings, decrypting sensitive fields."""
        if not self._settings_path.exists():
            return AllSkySettings()

        try:
            with open(self._settings_path) as fh:
                raw: dict = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read settings file: %s", exc)
            return AllSkySettings()

        ssh_raw = raw.get("ssh", {})
        encrypted_pw = ssh_raw.get("password_enc", "")
        plain_pw = self._decrypt(encrypted_pw) if encrypted_pw else ""

        return AllSkySettings(
            allsky_image_dir=raw.get("allsky_image_dir", "/var/lib/indi-allsky/images"),
            ssh=SSHSettings(
                host=ssh_raw.get("host", ""),
                username=ssh_raw.get("username", ""),
                password=plain_pw,
                port=int(ssh_raw.get("port", 22)),
            ),
        )

    def save(self, settings: AllSkySettings) -> None:
        """Persist *settings*, encrypting the SSH password before writing."""
        encrypted_pw = (
            self._encrypt(settings.ssh.password) if settings.ssh.password else ""
        )

        data = {
            "allsky_image_dir": settings.allsky_image_dir,
            "ssh": {
                "host": settings.ssh.host,
                "username": settings.ssh.username,
                "password_enc": encrypted_pw,
                "port": settings.ssh.port,
            },
        }

        tmp_path = self._settings_path.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w") as fh:
                json.dump(data, fh, indent=2)
            # Atomic replace.
            tmp_path.replace(self._settings_path)
        except OSError as exc:
            logger.error("Failed to save settings: %s", exc)
            tmp_path.unlink(missing_ok=True)
            raise

    # ------------------------------------------------------------------
    # Encryption helpers
    # ------------------------------------------------------------------

    def _load_or_create_fernet(self):
        """Return a Fernet instance, generating a new key if needed."""
        from cryptography.fernet import Fernet

        if self._key_path.exists():
            key = self._key_path.read_bytes()
        else:
            key = Fernet.generate_key()
            # Write with restrictive permissions (owner read-only).
            self._key_path.write_bytes(key)
            try:
                os.chmod(self._key_path, 0o600)
            except OSError:
                pass
            logger.info("Generated new encryption key at %s", self._key_path)

        return Fernet(key)

    def _encrypt(self, plaintext: str) -> str:
        """Return the URL-safe base64 ciphertext string."""
        return self._fernet.encrypt(plaintext.encode()).decode()

    def _decrypt(self, ciphertext: str) -> str:
        """Return the decrypted plaintext, or empty string on failure."""
        try:
            return self._fernet.decrypt(ciphertext.encode()).decode()
        except Exception as exc:
            logger.warning("Decryption failed (key mismatch?): %s", exc)
            return ""
