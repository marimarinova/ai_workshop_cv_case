"""CLI-level tests for the task_16 ``infer`` command (directory/batch mode).

These drive the Typer command through ``CliRunner`` with ``run_pipeline``
monkeypatched, so they exercise the batch orchestration (per-file isolation,
``batch_summary.json`` aggregation, exit-code semantics) without models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from pickup_putdown.cli_infer import infer_app

runner = CliRunner()


def _make_videos(directory: Path, names: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / f"{name}.mp4").write_bytes(b"SYNTHETIC")


def _patch_pipeline(
    monkeypatch: pytest.MonkeyPatch, *, fail_stems: set[str], status: str = "ok"
) -> None:
    def _fake_run_pipeline(
        video: Path, output_root: Path, config: Any, *, resume: bool = True, **_: Any
    ) -> dict[str, Any]:
        if video.stem in fail_stems:
            raise RuntimeError("boom")
        return {"clip_id": f"clip_{video.stem}", "status": status}

    monkeypatch.setattr("pickup_putdown.cli_infer.run_pipeline", _fake_run_pipeline)


def test_directory_mode_isolates_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    videos = tmp_path / "vids"
    _make_videos(videos, ["good_a", "bad", "good_b"])
    _patch_pipeline(monkeypatch, fail_stems={"bad"})
    out = tmp_path / "out"

    result = runner.invoke(infer_app, ["infer", "-i", str(videos), "-o", str(out)])

    # One failing clip must not stop the others, but must fail the batch overall.
    assert result.exit_code == 1
    batch = json.loads((out / "batch_summary.json").read_text(encoding="utf-8"))
    assert batch["n_total"] == 3
    assert batch["n_failed"] == 1
    assert batch["n_ok"] == 2
    statuses = {r["clip_id"]: r["status"] for r in batch["results"]}
    assert statuses == {"clip_good_a": "ok", "clip_bad": "failed", "clip_good_b": "ok"}
    assert any("boom" in (r.get("error") or "") for r in batch["results"])


def test_directory_mode_all_ok_exits_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    videos = tmp_path / "vids"
    _make_videos(videos, ["a", "b"])
    _patch_pipeline(monkeypatch, fail_stems=set())
    out = tmp_path / "out"

    result = runner.invoke(infer_app, ["infer", "-i", str(videos), "-o", str(out)])

    assert result.exit_code == 0
    batch = json.loads((out / "batch_summary.json").read_text(encoding="utf-8"))
    assert batch["n_failed"] == 0
    assert batch["n_ok"] == 2


def test_empty_directory_exits_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    _patch_pipeline(monkeypatch, fail_stems=set())

    result = runner.invoke(infer_app, ["infer", "-i", str(empty), "-o", str(tmp_path / "o")])

    assert result.exit_code == 2
    assert not (tmp_path / "o" / "batch_summary.json").exists()


def test_single_file_mode_prints_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"SYNTHETIC")
    _patch_pipeline(monkeypatch, fail_stems=set())
    out = tmp_path / "out"

    result = runner.invoke(infer_app, ["infer", "-i", str(video), "-o", str(out)])

    assert result.exit_code == 0
    # Single-file mode emits the per-clip summary, not a batch summary.
    assert '"clip_id": "clip_clip"' in result.stdout
    assert not (out / "batch_summary.json").exists()


def test_single_file_crash_exits_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"SYNTHETIC")
    _patch_pipeline(monkeypatch, fail_stems={"clip"})

    result = runner.invoke(infer_app, ["infer", "-i", str(video), "-o", str(tmp_path / "out")])

    assert result.exit_code == 5
    assert '"status": "failed"' in result.stdout


def test_single_file_blocked_status_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"SYNTHETIC")
    _patch_pipeline(monkeypatch, fail_stems=set(), status="blocked")

    result = runner.invoke(infer_app, ["infer", "-i", str(video), "-o", str(tmp_path / "out")])

    # A blocked top-level status must not report success.
    assert result.exit_code == 4
    assert '"status": "blocked"' in result.stdout
