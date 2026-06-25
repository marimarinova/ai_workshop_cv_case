"""Integration tests for the task_16 ``infer`` CLI through the real registry.

Unlike :mod:`test_cli_infer` (which monkeypatches ``run_pipeline``), these
drive the command end-to-end through ``build_default_registry`` on a tiny
fixture video. Without model checkpoints on disk the triage/propose stages
report *unavailable*, so the run completes with a valid, empty ``events.csv``
and exercises the real wiring: config loading, ``resolved_config.yaml``
materialisation, the structured output directory, and schema validation.

Known limitation: because no checkpoints are present, the *executed*
triage/propose subprocess path (``_CliStage._invoke``) is not covered here.
Covering it requires real models and is out of scope for CI smoke testing.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from pickup_putdown.cli_infer import infer_app
from pickup_putdown.pipeline import validate_events_csv

runner = CliRunner()


def _tiny_video(path: Path) -> Path:
    path.write_bytes(b"SYNTHETIC-TINY-CLIP")
    return path


def test_infer_single_file_real_registry_no_models(tmp_path: Path) -> None:
    video = _tiny_video(tmp_path / "clip_tiny.mp4")
    out = tmp_path / "out"

    result = runner.invoke(infer_app, ["infer", "-i", str(video), "-o", str(out)])

    assert result.exit_code == 0
    clip_dir = out / "clip_tiny"
    # Structured output directory with the reproducibility artifacts.
    assert (clip_dir / "resolved_config.yaml").is_file()
    assert (clip_dir / "run_metadata.json").is_file()
    assert (clip_dir / "summary.json").is_file()

    events = clip_dir / "events.csv"
    assert validate_events_csv(events)  # header-only but schema-valid

    summary = json.loads((clip_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert summary["n_events"] == 0
    assert summary["events_valid"] is True
    # Without checkpoints the model-backed stages are unavailable.
    assert summary["stages"]["triage"]["status"] == "unavailable"
    assert summary["stages"]["propose"]["status"] == "unavailable"


def test_infer_directory_real_registry_writes_batch_summary(tmp_path: Path) -> None:
    videos = tmp_path / "vids"
    videos.mkdir()
    _tiny_video(videos / "a.mp4")
    _tiny_video(videos / "b.mp4")
    out = tmp_path / "out"

    result = runner.invoke(infer_app, ["infer", "-i", str(videos), "-o", str(out)])

    assert result.exit_code == 0
    batch = json.loads((out / "batch_summary.json").read_text(encoding="utf-8"))
    assert batch["n_total"] == 2
    assert batch["n_failed"] == 0
    assert batch["n_ok"] == 2
    for stem in ("a", "b"):
        assert validate_events_csv(out / stem / "events.csv")
