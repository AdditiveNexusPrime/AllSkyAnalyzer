"""Tests for SettingsStore – encrypted persistence of SSH credentials."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.settings_store import AllSkySettings, SSHSettings, SettingsStore


@pytest.fixture()
def store(tmp_path: Path) -> SettingsStore:
    """Return a SettingsStore backed by a temporary directory."""
    return SettingsStore(data_dir=tmp_path)


class TestSettingsStoreDefaults:
    def test_load_returns_defaults_when_no_file(self, store: SettingsStore):
        s = store.load()
        assert isinstance(s, AllSkySettings)
        assert s.allsky_image_dir == "/var/lib/indi-allsky/images"
        assert s.ssh.host == ""
        assert s.ssh.password == ""

    def test_data_dir_created(self, tmp_path: Path):
        nested = tmp_path / "deep" / "nested"
        SettingsStore(data_dir=nested)
        assert nested.exists()

    def test_key_file_created(self, store: SettingsStore, tmp_path: Path):
        key_path = tmp_path / "secret.key"
        assert key_path.exists()

    def test_key_file_is_32_bytes_base64(self, tmp_path: Path):
        import base64
        store = SettingsStore(data_dir=tmp_path)
        raw = (tmp_path / "secret.key").read_bytes()
        # Fernet key is a 32-byte value URL-safe base64 encoded (44 chars).
        decoded = base64.urlsafe_b64decode(raw)
        assert len(decoded) == 32


class TestSettingsStoreSaveLoad:
    def test_roundtrip_allsky_dir(self, store: SettingsStore):
        settings = AllSkySettings(allsky_image_dir="/my/allsky")
        store.save(settings)
        loaded = store.load()
        assert loaded.allsky_image_dir == "/my/allsky"

    def test_roundtrip_ssh_host_and_user(self, store: SettingsStore):
        settings = AllSkySettings(
            ssh=SSHSettings(host="192.168.1.50", username="pi", port=2222)
        )
        store.save(settings)
        loaded = store.load()
        assert loaded.ssh.host == "192.168.1.50"
        assert loaded.ssh.username == "pi"
        assert loaded.ssh.port == 2222

    def test_password_is_encrypted_on_disk(self, store: SettingsStore, tmp_path: Path):
        settings = AllSkySettings(ssh=SSHSettings(password="s3cr3t!"))
        store.save(settings)
        raw = json.loads((tmp_path / "settings.json").read_text())
        # The stored value must NOT be the plaintext password.
        assert raw["ssh"]["password_enc"] != "s3cr3t!"
        assert len(raw["ssh"]["password_enc"]) > 20

    def test_password_decrypted_on_load(self, store: SettingsStore):
        settings = AllSkySettings(ssh=SSHSettings(password="s3cr3t!"))
        store.save(settings)
        loaded = store.load()
        assert loaded.ssh.password == "s3cr3t!"

    def test_empty_password_not_stored_encrypted(self, store: SettingsStore, tmp_path: Path):
        settings = AllSkySettings(ssh=SSHSettings(password=""))
        store.save(settings)
        raw = json.loads((tmp_path / "settings.json").read_text())
        assert raw["ssh"]["password_enc"] == ""

    def test_multiple_saves_overwrite(self, store: SettingsStore):
        store.save(AllSkySettings(allsky_image_dir="/first"))
        store.save(AllSkySettings(allsky_image_dir="/second"))
        assert store.load().allsky_image_dir == "/second"


class TestSettingsStoreKeyPersistence:
    def test_same_key_used_across_instances(self, tmp_path: Path):
        """Two SettingsStore instances pointing at the same dir share the key."""
        store1 = SettingsStore(data_dir=tmp_path)
        store1.save(AllSkySettings(ssh=SSHSettings(password="hello")))

        store2 = SettingsStore(data_dir=tmp_path)
        assert store2.load().ssh.password == "hello"

    def test_wrong_key_returns_empty_password(self, tmp_path: Path):
        """If the key file is replaced, decryption fails gracefully."""
        from cryptography.fernet import Fernet

        store1 = SettingsStore(data_dir=tmp_path)
        store1.save(AllSkySettings(ssh=SSHSettings(password="secret")))

        # Overwrite the key file with a fresh key.
        (tmp_path / "secret.key").write_bytes(Fernet.generate_key())

        store2 = SettingsStore(data_dir=tmp_path)
        loaded = store2.load()
        # Should not raise; password should be empty string on decrypt failure.
        assert loaded.ssh.password == ""


class TestSettingsStoreCorruptFile:
    def test_corrupt_json_returns_defaults(self, store: SettingsStore, tmp_path: Path):
        (tmp_path / "settings.json").write_text("not-valid-json{{{")
        loaded = store.load()
        assert loaded.allsky_image_dir == "/var/lib/indi-allsky/images"
