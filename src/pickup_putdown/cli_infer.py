"""Typer commands for end-to-end batch inference (task_16).

Exposes the ``infer`` command (single file or directory of videos) plus placeholder
commands for pipeline components not yet implemented (Track A/B, Layer 2/3).
The whole group is attached to the root CLI with a single ``add_typer`` call in
:mod:`pickup_putdown.cli`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import typer

from pickup_putdown.config import AppConfig, load_config
from pickup_putdown.pipeline import atomic_write_json, run_pipeline

infer_app = typer.Typer(
    name="pipeline",
    help="End-to-end and per-stage batch inference.",
    add_completion=False,
)

logger = logging.getLogger(__name__)

#: Container extensions accepted in directory mode (matches the triage CLI).
_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"})

#: Top-level pipeline statuses that must not be reported as a success.
_FAILURE_STATUSES = frozenset({"failed", "blocked"})

#: Compact per-file fields surfaced in ``batch_summary.json``.
_RECORD_FIELDS = ("clip_id", "input", "status", "output_dir", "error")


def _exit_code_for(status: str) -> int:
    """Map a single-clip top-level status to a process exit code."""
    if status == "failed":
        return 5
    if status == "blocked":
        return 4
    return 0


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _resolve_inputs(input_path: str) -> list[Path]:
    """Resolve a file or directory argument to a sorted list of video paths."""
    inp = Path(input_path)
    if inp.is_file():
        return [inp]
    if inp.is_dir():
        videos = sorted(f for f in inp.iterdir() if f.suffix.lower() in _VIDEO_EXTENSIONS)
        if not videos:
            typer.echo(f"No video files found in {inp}", err=True)
            raise typer.Exit(code=2)
        return videos
    typer.echo(f"Input not found: {input_path}", err=True)
    raise typer.Exit(code=2)


def _run_one(video: Path, output_root: Path, config: AppConfig, *, resume: bool) -> dict[str, Any]:
    """Run the pipeline for one video, turning any crash into a failed summary.

    Returns the full per-clip summary on success (enriched with input/output
    paths), or a synthetic ``status="failed"`` summary when ``run_pipeline``
    raises—so one bad clip can never abort a batch.
    """
    clip_dir = output_root / video.stem
    try:
        summary = run_pipeline(video, output_root, config, resume=resume)
    except Exception as exc:  # noqa: BLE001 - one bad clip must not stop the batch
        logger.exception("pipeline failed for %s", video)
        return {
            "clip_id": f"clip_{video.stem}",
            "input": str(video),
            "status": "failed",
            "output_dir": str(clip_dir),
            "error": f"{type(exc).__name__}: {exc}",
        }
    summary.setdefault("input", str(video))
    summary.setdefault("output_dir", str(clip_dir))
    return summary


def _record(summary: dict[str, Any]) -> dict[str, Any]:
    """Project a per-clip summary to the compact fields kept in the batch file."""
    return {field: summary[field] for field in _RECORD_FIELDS if field in summary}


@infer_app.command("infer")
def infer(
    input_path: str = typer.Option(
        ..., "--input", "-i", help="Path to a single video file or a directory of videos."
    ),
    output_dir: str = typer.Option(
        "outputs/infer", "--output-dir", "-o", help="Base directory for run outputs."
    ),
    config: str | None = typer.Option(
        None, "--config", "-c", help="Optional configuration YAML file."
    ),
    resume: bool = typer.Option(
        True, "--resume/--no-resume", help="Skip stages whose inputs are unchanged."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Run the full pipeline end-to-end on one video file or a directory of videos.

    Exit codes:

    \b
      0  success (every clip ok / no_person)
      1  directory mode: at least one clip failed
      2  bad input (file/directory not found, or an empty directory)
      4  single file: pipeline blocked (a stage's upstream dependency failed)
      5  single file: pipeline failed (a stage crashed)
    """
    _setup_logging(verbose)
    videos = _resolve_inputs(input_path)
    app_config = load_config(config)
    output_root = Path(output_dir)

    # Single-file mode: emit the full per-clip summary on stdout. A crash or a
    # blocked/failed top-level status maps to a non-zero exit code.
    if Path(input_path).is_file():
        summary = _run_one(videos[0], output_root, app_config, resume=resume)
        typer.echo(json.dumps(summary, indent=2, default=str))
        code = _exit_code_for(str(summary.get("status", "ok")))
        if code:
            raise typer.Exit(code=code)
        return

    # Directory/batch mode: isolate per-file failures and aggregate a summary.
    summaries = [_run_one(video, output_root, app_config, resume=resume) for video in videos]
    n_failed = sum(1 for summary in summaries if summary.get("status") in _FAILURE_STATUSES)
    batch_summary: dict[str, Any] = {
        "n_total": len(summaries),
        "n_ok": len(summaries) - n_failed,
        "n_failed": n_failed,
        "results": [_record(summary) for summary in summaries],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "batch_summary.json", batch_summary)
    typer.echo(json.dumps(batch_summary, indent=2, default=str))
    if n_failed:
        raise typer.Exit(code=1)


def _unavailable(component: str, depends_on: str) -> None:
    typer.echo(
        f"Component '{component}' is not available yet (depends on {depends_on}).",
        err=True,
    )
    raise typer.Exit(code=2)


@infer_app.command("track-a")
def track_a() -> None:
    """[stub] Track A interpretable detector inference."""
    _unavailable("track-a", "task_9/task_10")


@infer_app.command("track-b1")
def track_b1() -> None:
    """[stub] Track B1 VideoMAE classifier inference."""
    _unavailable("track-b1", "task_12")


@infer_app.command("track-b2")
def track_b2() -> None:
    """[stub] Track B2 VideoMAE+TCN inference."""
    _unavailable("track-b2", "task_13")


@infer_app.command("layer2")
def layer2() -> None:
    """[stub] Layer 2 standalone Qwen inference."""
    _unavailable("layer2", "task_14")


@infer_app.command("verify")
def verify() -> None:
    """[stub] Layer 3 Qwen verification."""
    _unavailable("verify", "task_15")


@infer_app.command("fuse")
def fuse() -> None:
    """[stub] Layer 3 fusion of detector and verifier outputs."""
    _unavailable("fuse", "task_15")
