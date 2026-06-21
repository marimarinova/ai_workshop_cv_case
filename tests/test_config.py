"""Tests for configuration loading and environment overrides."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from pickup_putdown.config import AppConfig, load_config


@pytest.fixture
def temp_config(tmp_path: Path) -> Path:
    config = {
        "storage": {
            "bucket_uri": "s3://test-bucket",
            "region": "us-east-1",
            "anonymous": False,
        },
        "triage": {
            "target_fps": 2.0,
            "minimum_visible_duration_s": 1.0,
        },
        "proposals": {
            "target_fps": 4,
            "minimum_interaction_duration_s": 0.5,
        },
        "data_dir": "custom_data",
    }
    path = tmp_path / "test_config.yaml"
    with open(path, "w") as fh:
        yaml.dump(config, fh)
    return path


class TestConfigLoading:
    def test_load_from_file(self, temp_config):
        cfg = load_config(temp_config)
        assert isinstance(cfg, AppConfig)
        assert cfg.storage.bucket_uri == "s3://test-bucket"
        assert cfg.triage.target_fps == 2.0
        assert cfg.data_dir == "custom_data"

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_none_returns_defaults(self):
        cfg = load_config(None)
        assert cfg.storage.bucket_uri == ""
        assert cfg.triage.target_fps == 1.0
        assert cfg.data_dir == "data"

    def test_env_override_storage_bucket(self, temp_config):
        os.environ["PICKUP_PUTDOWN_STORAGE_BUCKET_URI"] = "s3://override-bucket"
        try:
            cfg = load_config(temp_config)
            assert cfg.storage.bucket_uri == "s3://override-bucket"
        finally:
            del os.environ["PICKUP_PUTDOWN_STORAGE_BUCKET_URI"]

    def test_env_override_triage_fps(self, temp_config):
        os.environ["PICKUP_PUTDOWN_TRIAGE_TARGET_FPS"] = "5.0"
        try:
            cfg = load_config(temp_config)
            assert cfg.triage.target_fps == 5.0
        finally:
            del os.environ["PICKUP_PUTDOWN_TRIAGE_TARGET_FPS"]

    def test_env_override_boolean(self, temp_config):
        os.environ["PICKUP_PUTDOWN_STORAGE_ANONYMOUS"] = "true"
        try:
            cfg = load_config(temp_config)
            assert cfg.storage.anonymous is True
        finally:
            del os.environ["PICKUP_PUTDOWN_STORAGE_ANONYMOUS"]

    def test_deterministic_resolution(self, temp_config):
        cfg1 = load_config(temp_config)
        cfg2 = load_config(temp_config)
        assert cfg1.model_dump() == cfg2.model_dump()

    def test_empty_config_returns_defaults(self, tmp_path: Path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        cfg = load_config(path)
        assert isinstance(cfg, AppConfig)
