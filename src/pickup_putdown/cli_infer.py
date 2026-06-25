"""Typer commands for end-to-end batch inference (task_16).

Exposes the ``infer`` command (single-file end-to-end run) plus placeholder
commands for pipeline components not yet implemented (Track A/B, Layer 2/3).
The whole group is attached to the root CLI with a single ``add_typer`` call in
:mod:`pickup_putdown.cli`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from pickup_putdown.config import load_config
from pickup_putdown.pipeline import run_pipeline

infer_app = typer.Typer(
    name="pipeline",
    help="End-to-end and per-stage batch inference.",
    add_completion=False,
)

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@infer_app.command("infer")
def infer(
    input_path: str = typer.Option(..., "--input", "-i", help="Path to a single MP4 file."),
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
    """Run the full pipeline end-to-end on a single video file."""
    _setup_logging(verbose)
    video = Path(input_path)
    if not video.is_file():
        typer.echo(f"Input file not found: {input_path}", err=True)
        raise typer.Exit(code=2)

    app_config = load_config(config)
    summary = run_pipeline(video, Path(output_dir), app_config, resume=resume)
    typer.echo(json.dumps(summary, indent=2, default=str))
    if summary.get("status") == "failed":
        raise typer.Exit(code=5)


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
