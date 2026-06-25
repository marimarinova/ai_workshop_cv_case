"""Tests for the infer-track-a CLI (task_10).

Drive the command through the real root app on synthetic parquet + shelves, with
no models on disk — so the real encoder path stays gated behind is_available and
CI exercises only parsing + the blocked-unavailable / bad-input exit codes.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from pickup_putdown.cli import app
from pickup_putdown.common.schemas import Candidate, PoseObservation
from pickup_putdown.layer1.track_a.cli_track_a import (
    EXIT_BAD_INPUT,
    EXIT_BLOCKED_UNAVAILABLE,
    load_candidate_inputs,
)

runner = CliRunner()

_SHELVES_YAML = """\
cameras:
  cam1:
    source_width: 100
    source_height: 100
    regions:
      - region_id: shelf1
        type: shelf
        polygon: [[10, 10], [40, 10], [40, 40], [10, 40]]
"""


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    pa = importlib.import_module("pyarrow")
    pq = importlib.import_module("pyarrow.parquet")
    pq.write_table(pa.Table.from_pylist(rows), str(path))


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    candidate = Candidate(
        candidate_id="cand1",
        clip_id="clip_x",
        actor_id="a1",
        hand_side="left",
        region_id="shelf1",
        raw_start_s=0.0,
        raw_end_s=4.0,
        window_start_s=0.0,
        window_end_s=4.0,
    )
    pose = PoseObservation(
        clip_id="clip_x",
        timestamp_s=1.0,
        sample_index=0,
        actor_id="a1",
        hand_side="left",
        wrist_x=20.0,
        wrist_y=20.0,
        wrist_confidence=0.9,
    )
    candidates_path = tmp_path / "candidates.parquet"
    pose_path = tmp_path / "tracks_pose.parquet"
    shelves_path = tmp_path / "shelves.yaml"
    _write_parquet(candidates_path, [candidate.model_dump()])
    _write_parquet(pose_path, [pose.model_dump()])
    shelves_path.write_text(_SHELVES_YAML, encoding="utf-8")
    return candidates_path, pose_path, shelves_path


def test_load_candidate_inputs_roundtrips_parquet_and_shelves(tmp_path: Path) -> None:
    candidates_path, pose_path, shelves_path = _fixture(tmp_path)

    inputs = load_candidate_inputs(candidates_path, pose_path, shelves_path)

    assert len(inputs) == 1
    assert inputs[0].candidate.candidate_id == "cand1"
    assert inputs[0].pose_observations[0].actor_id == "a1"
    # Region resolved to an expanded polygon (list of points).
    assert len(inputs[0].shelf_region) >= 3


def test_cli_blocked_unavailable_without_checkpoints(tmp_path: Path) -> None:
    candidates_path, pose_path, shelves_path = _fixture(tmp_path)

    result = runner.invoke(
        app,
        [
            "infer-track-a",
            "--candidates",
            str(candidates_path),
            "--pose",
            str(pose_path),
            "--shelves",
            str(shelves_path),
            "--output",
            str(tmp_path / "events.csv"),
        ],
    )

    # Inputs parsed fine, but no trained checkpoints -> blocked, not ok.
    assert result.exit_code == EXIT_BLOCKED_UNAVAILABLE
    assert not (tmp_path / "events.csv").exists()


def test_cli_bad_input_when_file_missing(tmp_path: Path) -> None:
    _, pose_path, shelves_path = _fixture(tmp_path)

    result = runner.invoke(
        app,
        [
            "infer-track-a",
            "--candidates",
            str(tmp_path / "missing.parquet"),
            "--pose",
            str(pose_path),
            "--shelves",
            str(shelves_path),
        ],
    )

    assert result.exit_code == EXIT_BAD_INPUT
