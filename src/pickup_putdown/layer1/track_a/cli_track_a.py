"""Standalone ``infer-track-a`` CLI for the Track A detector (task_10).

Reads ``candidates.parquet`` + ``tracks_pose.parquet`` (Stage B / propose
outputs) and a shelves config, runs Track A, and writes a Task 8-consumable
``events.csv``. The real encoder/crop path is gated behind classifier
availability: without trained checkpoints (task_7 pending) the command reports
*blocked-unavailable* and never decodes video, so CI exercises parsing + gating
on the placeholder path.

Exit codes:
    0  ok                    — predictions written
    2  bad-input             — missing/unreadable inputs or misconfiguration
    3  blocked-unavailable   — Track A has no trained checkpoints yet (task_7)
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

import typer

from pickup_putdown.common.exceptions import ConfigError
from pickup_putdown.common.schemas import Candidate, PoseObservation
from pickup_putdown.config import load_config
from pickup_putdown.layer1.track_a import hand_state, shelf_state
from pickup_putdown.layer1.track_a.hand_state import load_hand_state_classifier
from pickup_putdown.layer1.track_a.inference import (
    CandidateInput,
    build_feature_fn,
    infer_track_a,
    write_events_csv,
)
from pickup_putdown.layer1.track_a.shelf_state import load_shelf_state_classifier

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_BAD_INPUT = 2
EXIT_BLOCKED_UNAVAILABLE = 3


def _read_rows(path: Path) -> list[dict[str, Any]]:
    pq = importlib.import_module("pyarrow.parquet")  # lazy: pyarrow is heavy
    return list(pq.read_table(str(path)).to_pylist())


def _sole_camera(camera_ids: list[str]) -> str:
    if len(camera_ids) != 1:
        raise ConfigError(
            f"shelves config has {len(camera_ids)} cameras {sorted(camera_ids)}; "
            "pass --camera-id to disambiguate"
        )
    return camera_ids[0]


def load_candidate_inputs(
    candidates_path: Path,
    pose_path: Path,
    shelves_path: Path,
    *,
    camera_id: str | None = None,
) -> list[CandidateInput]:
    """Deserialise propose outputs + shelves into Track A candidate inputs."""
    # Lazy: importing perception triggers its package init (heavy, and pulls the
    # platform-specific ingestion chain), so keep it off the module import path.
    from pickup_putdown.perception.shelf_regions import (
        get_expanded_regions,
        get_regions_for_camera,
        load_shelf_config,
    )

    candidates = [Candidate(**row) for row in _read_rows(candidates_path)]
    poses = tuple(PoseObservation(**row) for row in _read_rows(pose_path))

    shelf_config = load_shelf_config(shelves_path)
    camera = camera_id or _sole_camera(list(shelf_config.cameras))
    regions = get_expanded_regions(get_regions_for_camera(shelf_config, camera))

    inputs: list[CandidateInput] = []
    for candidate in candidates:
        region = regions.get(candidate.region_id or "")
        if region is None:
            logger.warning(
                "candidate %s region %r not in shelves config; skipping",
                candidate.candidate_id,
                candidate.region_id,
            )
            continue
        inputs.append(
            CandidateInput(candidate=candidate, pose_observations=poses, shelf_region=region)
        )
    return inputs


def infer_track_a_command(
    candidates: str = typer.Option(..., "--candidates", help="Path to candidates.parquet."),
    pose: str = typer.Option(..., "--pose", help="Path to tracks_pose.parquet."),
    shelves: str = typer.Option(..., "--shelves", help="Path to the shelves config YAML."),
    output: str = typer.Option("events.csv", "--output", "-o", help="Output events.csv path."),
    video: str | None = typer.Option(None, "--video", help="Source video (real encoder path)."),
    config: str | None = typer.Option(None, "--config", "-c", help="Optional config YAML."),
    camera_id: str | None = typer.Option(
        None, "--camera-id", help="Camera id in the shelves config."
    ),
    clip_id: str | None = typer.Option(
        None, "--clip-id", help="Clip id (default: from candidates)."
    ),
) -> None:
    """Run the Track A interpretable detector over one clip's candidates."""
    for label, value in (("candidates", candidates), ("pose", pose), ("shelves", shelves)):
        if not Path(value).is_file():
            typer.echo(f"{label} not found: {value}", err=True)
            raise typer.Exit(code=EXIT_BAD_INPUT)

    app_config = load_config(config)
    try:
        candidate_inputs = load_candidate_inputs(
            Path(candidates), Path(pose), Path(shelves), camera_id=camera_id
        )
    except (ConfigError, ValueError, KeyError, OSError) as exc:
        typer.echo(f"Failed to parse inputs: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=EXIT_BAD_INPUT) from exc

    # Real encoder/crop path is gated on trained checkpoints (task_7).
    if not (
        hand_state.is_available(app_config.track_a)
        and shelf_state.is_available(app_config.track_a)
    ):
        typer.echo(
            "Track A is unavailable: no trained classifier checkpoints "
            "(task_7 pending). Not running inference.",
            err=True,
        )
        raise typer.Exit(code=EXIT_BLOCKED_UNAVAILABLE)

    if video is None or not Path(video).is_file():
        typer.echo("--video is required and must exist for the trained path.", err=True)
        raise typer.Exit(code=EXIT_BAD_INPUT)

    resolved_clip = clip_id or (
        candidate_inputs[0].candidate.clip_id if candidate_inputs else "clip"
    )
    predictions = infer_track_a(
        resolved_clip,
        candidate_inputs,
        app_config.track_a,
        hand_classifier=load_hand_state_classifier(app_config.track_a),
        shelf_classifier=load_shelf_state_classifier(app_config.track_a),
        feature_fn=build_feature_fn(Path(video), app_config.track_a_features),
    )
    events_path = write_events_csv(Path(output), predictions)
    typer.echo(f"Wrote {len(predictions)} Track A prediction(s) to {events_path}")
