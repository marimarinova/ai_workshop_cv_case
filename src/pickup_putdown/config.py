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

    # Pipeline configuration for multiprocessed frame decoding
    pipeline_enabled: bool = True
    pipeline_queue_depth: int = 8
    pipeline_n_decoders: int = 2
    pipeline_resize_frames: bool = False  # Keep original resolution for identical results
    pipeline_frame_size: tuple[int, int] = (640, 640)  # Only used if resize_frames=True
    pipeline_frame_timeout_s: float = 30.0


class PoseConfig(BaseModel):
    """Configuration for YOLO pose inference on person-active spans."""

    model_path: str = "models/pose_detector.pt"
    target_fps: float = 8.0
    image_size: int = 640
    device: str = "auto"
    half: bool = False
    pose_confidence: float = 0.30
    max_detections: int = 100
    process_active_spans_only: bool = True


class ActorAssociationConfig(BaseModel):
    """Configuration for associating pose detections with actor tracks."""

    min_actor_confidence: float = 0.30
    match_iou_threshold: float = 0.15
    max_gap_s: float = 0.5
    allow_unmatched: bool = True


class RegionMeasurementConfig(BaseModel):
    """Configuration for region-based wrist measurements."""

    expanded_margin_override: float | None = None
    min_wrist_confidence: float = 0.30
    min_dwell_duration_s: float = 0.25
    velocity_window_frames: int = 5
    reversal_threshold: float = 0.30
    gap_tolerance_s: float = 0.5


class ProposalsConfig(BaseModel):
    """Configuration for raw interaction detection and candidate generation."""

    target_fps: float = 8
    minimum_wrist_confidence: float = 0.30
    minimum_interaction_duration_s: float = 0.25
    merge_gap_s: float = 1.0
    context_before_s: float = 2.0
    context_after_s: float = 2.0
    maximum_candidate_duration_s: float = 10.0
    trajectory_smoothing: bool = False
    smoothing_window: int = 3


class TrackAFeaturesConfig(BaseModel):
    """Configuration for Track A feature extraction (Task 9)."""

    # Sampling configuration
    min_samples: int = 3
    max_interval_s: float = (
        99999.0  # Large default = no intermediate splits (just pre/contact/post)
    )

    # Crop configuration
    hand_crop_size: int = 224
    shelf_patch_size: int = 224
    crop_scale_method: str = "bbox"  # "bbox" or "limb_length"

    # Encoder configuration (supported: mobilenet_v3_small, mobilenet_v3_large, resnet18, resnet50, efficientnet_b0)
    encoder_name: str = "mobilenet_v3_small"
    encoder_version: str = "v1.0"
    encoder_device: str = "auto"
    encoder_batch_size: int = 32

    # Cache configuration
    cache_dir: str = ".local/track_a_features"
    save_crops: bool = True

    # QA configuration
    qa_samples_per_category: int = 20


class TrackB1Config(BaseModel):
    """Configuration for Track B1 VideoMAE window classifier (Task 12)."""

    # Window parameters
    window_duration_s: float = 2.5
    window_stride_s: float = 0.5
    min_window_duration_s: float = 1.0

    # Frame sampling
    num_frames: int = 16
    image_size: tuple[int, int] = (224, 224)

    # Actor-conditioned crop
    crop_margin: float = 0.15

    # Label weights by confidence
    weight_high: float = 1.0
    weight_med: float = 1.0
    weight_low: float = 0.5

    # Training parameters
    batch_size: int = 8
    num_workers: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 20
    warmup_epochs: int = 2

    # Model configuration
    model_name: str = "MCG-NJU/videomae-small"
    freeze_backbone: bool = True
    unfreeze_last_n_blocks: int = 0

    # Inference parameters
    score_threshold: float = 0.5
    smoothing_window: int = 3
    same_type_merge_gap_s: float = 0.75
    min_event_duration_s: float = 0.3

    # Output paths
    checkpoint_dir: str = ".local/track_b1_checkpoints"
    results_dir: str = "results/layer1_track_b1"


class TrackAStageConfig(BaseModel):
    """Wiring for the Track A detector stage in the batch pipeline (Task 16).

    Defaults mirror the ``infer-track-a`` CLI so activating Track A is a
    checkpoint drop-in (Task 7) with no code change. ``config_path`` points at
    the Track A YAML — its own ``classifiers``/``state_machine``/``inference``
    schema, distinct from the resolved :class:`AppConfig`.
    """

    config_path: str = "configs/track_a.yaml"
    shelves_config: str = "configs/shelves.yaml"
    camera_id: str = "store_camera_01"
    artifact_dir: str = ".local/track_a_artifacts"


class PreviewConfig(BaseModel):
    """Configuration for candidate preview rendering."""

    preview_fps: float = 4.0
    max_output_width: int = 1280
    max_output_height: int = 720
    draw_actor_box: bool = True
    draw_wrist_positions: bool = True
    draw_region_polygons: bool = True
    draw_region_labels: bool = True
    draw_candidate_intervals: bool = True
    text_scale: float = 0.5
    line_thickness: int = 2


class AppConfig(BaseModel):
    storage: StorageConfig = Field(default_factory=StorageConfig)
    triage: TriageConfig = Field(default_factory=TriageConfig)
    tracker: ByteTrackTriageConfig = Field(default_factory=ByteTrackTriageConfig)
    pose: PoseConfig = Field(default_factory=PoseConfig)
    actor_association: ActorAssociationConfig = Field(default_factory=ActorAssociationConfig)
    region_measurements: RegionMeasurementConfig = Field(default_factory=RegionMeasurementConfig)
    proposals: ProposalsConfig = Field(default_factory=ProposalsConfig)
    preview: PreviewConfig = Field(default_factory=PreviewConfig)
    track_a_features: TrackAFeaturesConfig = Field(default_factory=TrackAFeaturesConfig)
    track_b1: TrackB1Config = Field(default_factory=TrackB1Config)
    track_a_stage: TrackAStageConfig = Field(default_factory=TrackAStageConfig)
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
            "pose",
            "actor_association",
            "region_measurements",
            "proposals",
            "preview",
            "track_a_features",
            "track_b1",
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
