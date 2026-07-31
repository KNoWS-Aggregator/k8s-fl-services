from pathlib import Path

import pytest

from data_preparation.settings import Settings


def test_settings_are_loaded_from_environment(monkeypatch):
    values = {
        "SOURCES": "https://kvasir.example/hospital-a/slices/case-a/",
        "AUTHN": "https://auth.example/realms/test/",
        "DATASET": "dataset-a",
        "DATA_DIR": "/mnt/data",
        "MAX_CONCURRENT_PARTICIPANTS": "3",
        "POL_ENABLED": "true",
        "POLL_INTERVAL": "@daily",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    settings = Settings.from_env()

    assert settings.sources == "https://kvasir.example/hospital-a/slices/case-a"
    assert settings.kvasir_server == "https://kvasir.example"
    assert settings.source_pod_id == "hospital-a"
    assert settings.data_dir == Path("/mnt/data")
    assert settings.max_concurrent_participants == 3
    assert settings.poll_enabled is True
    assert settings.poll_interval == "@daily"


def test_required_environment_variable(monkeypatch):
    monkeypatch.delenv("SOURCES", raising=False)
    with pytest.raises(ValueError, match="SOURCES"):
        Settings.from_env()


def test_invalid_poll_schedule(monkeypatch):
    monkeypatch.setenv("SOURCES", "https://kvasir.example/hospital-a/slices/case-a")
    monkeypatch.setenv("AUTHN", "https://auth.example/realms/test")
    monkeypatch.setenv("DATASET", "dataset-a")
    monkeypatch.setenv("POLL_INTERVAL", "every now and then")
    with pytest.raises(ValueError, match="POLL_INTERVAL"):
        Settings.from_env()
