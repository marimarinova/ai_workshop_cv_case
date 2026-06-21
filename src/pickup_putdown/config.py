"""Configuration loader with YAML support and environment-variable overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class StorageConfig(BaseModel):
    bucket_uri: str = ""
    region: str | None = None
    endpoint_url: str | None = None
    anonymous: bool = False


class ByteTrackTriageConfig(BaseModel):
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 30
    match_iou_threshold: float = 0.5
    max_age: int = 30
    min_hits: int = 3
    object_threshold: float = 0.10


class TriageConfig(BaseModel):
    model_path: str = "models/person_detector.pt"
    target_fps: float = 1.0
    image_size: int = 640
    device: str = "auto"
    half: bool = False
    detector_confidence: float = 0.10
    detector_iou_threshold: float = 0.70
    max_detections: int = 100
    minimum_track_confidence: float = 0.35
    minimum_visible_duration_s: float = 0.75
    minimum_observations: int = 2
    max_track_observation_gap_s: float = 1.5
    merge_gap_s: float = 1.0
    preview_sample_rate: float = 0.10
    sampling_seed: int = 42
    tracker_config: str = "configs/bytetrack_triage.yaml"


class ProposalsConfig(BaseModel):
    target_fps: float = 8
    minimum_wrist_confidence: float = 0.30
    minimum_interaction_duration_s: float = 0.25
    merge_gap_s: float = 1.0
    context_before_s: float = 2.0
    context_after_s: float = 2.0
    maximum_candidate_duration_s: float = 10.0


class AppConfig(BaseModel):
    storage: StorageConfig = Field(default_factory=StorageConfig)
    triage: TriageConfig = Field(default_factory=TriageConfig)
    tracker: ByteTrackTriageConfig = Field(default_factory=ByteTrackTriageConfig)
    proposals: ProposalsConfig = Field(default_factory=ProposalsConfig)
    data_dir: str = "data"
    output_dir: str = "outputs"
    cache_dir: str = "cache"
    results_dir: str = "results"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _build_env_overrides() -> dict[str, Any]:
    """Build a dict of overrides from environment variables.

    Supports the pattern PICKUP_PUTDOWN_<SECTION>_<KEY> for each config section.
    Values are cast to the appropriate Python type when possible.
    """
    overrides: dict[str, Any] = {}
    prefix = "PICKUP_PUTDOWN_"
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        rest = env_key[len(prefix) :].lower()
        parts = rest.split("_", 1)
        if len(parts) != 2:
            continue
        section, key = parts
        if section not in (
            "storage",
            "triage",
            "tracker",
            "proposals",
            "data",
            "output",
            "cache",
            "results",
        ):
            continue
        if section not in overrides:
            overrides[section] = {}
        overrides[section][key] = _cast_value(env_value)
    return overrides


def _cast_value(value: str) -> Any:
    """Attempt to cast a string environment variable to a Python type."""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    if value.lower() in ("null", "none", ""):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load configuration from a YAML file, applying environment overrides.

    Parameters
    ----------
    path : str or Path, optional
        Path to the YAML configuration file. If *None*, an empty config
        (all defaults) is returned.

    Returns
    -------
    AppConfig
        The resolved configuration with environment overrides applied.
    """
    config_dict: dict[str, Any] = {}
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path) as fh:
            config_dict = yaml.safe_load(fh) or {}

    env_overrides = _build_env_overrides()
    config_dict = _deep_merge(config_dict, env_overrides)

    return AppConfig(**config_dict)
