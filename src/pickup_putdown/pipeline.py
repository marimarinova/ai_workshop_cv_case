"""End-to-end batch inference orchestrator (task_16).

This module owns the *orchestration* layer that composes the individual stage
commands (triage, propose, evaluate, and the not-yet-implemented detector and
verifier stages) into a single, resumable run over one video file.

Design notes
------------
* ``Stage`` is a :class:`typing.Protocol`. Real stages and the placeholder
  stubs share one structural interface, so the detector stages added by
  task_9/task_10/task_14 plug in without touching the orchestrator.
* Stage outputs are written **atomically**: a stage runs into a ``*.partial``
  directory which is promoted with :func:`os.replace`, and the
  ``.stage_meta.json`` marker is written **last** (and atomically). A crash can
  therefore never leave a marker that points at incomplete outputs, which keeps
  resumability honest.
* Resumability is keyed on a content hash of the stage inputs (source checksum,
  resolved config, git commit, upstream hashes), not merely on output-file
  existence.
* A clip with no detected person short-circuits the remaining stages and still
  produces a valid, empty canonical ``events.csv``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import yaml

from pickup_putdown.common.run_metadata import RunMetadata
from pickup_putdown.config import AppConfig

logger = logging.getLogger(__name__)

StageStatus = Literal["ok", "no_person", "resumed", "unavailable", "blocked", "failed"]

#: Upstream statuses that satisfy a downstream stage's declared input.
_SATISFIED_INPUT_STATUSES: frozenset[StageStatus] = frozenset({"ok", "resumed"})

#: Column order of the canonical per-clip predictions file (``events.csv``).
#: Mirrors :class:`pickup_putdown.common.schemas.Prediction` and the columns
#: consumed by :func:`pickup_putdown.evaluation.io.predictions_from_rows`.
CANONICAL_EVENT_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "pred_id",
    "type",
    "t_start",
    "t_end",
    "score",
    "model",
)

_STAGE_META_NAME = ".stage_meta.json"
_TRIAGE_STAGE = "triage"


# ---------------------------------------------------------------------------
# Stage contract
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StageContext:
    """Immutable inputs handed to a stage's :meth:`Stage.run`.

    ``stage_dir`` is the per-clip output directory for this stage; ``cache_dir``
    is a cross-clip cache root (shared across the batch and across runs) where a
    stage may keep reusable artifacts such as frozen embeddings keyed by source
    checksum and encoder version.
    """

    clip_id: str
    video_path: Path
    output_dir: Path
    stage_dir: Path
    cache_dir: Path
    config: dict[str, Any]
    config_path: Path
    run_metadata: RunMetadata
    upstream: dict[str, StageResult]


@dataclass
class StageResult:
    """Outcome of a single stage."""

    name: str
    status: StageStatus
    outputs: dict[str, str] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    input_hash: str = ""


@runtime_checkable
class Stage(Protocol):
    """Structural contract shared by real stages and placeholder stubs."""

    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def is_available(self) -> bool:
        """Return ``True`` when the stage can run in the current environment."""

    def run(self, ctx: StageContext) -> StageResult:
        """Execute the stage, writing artifacts into ``ctx.stage_dir``."""


# ---------------------------------------------------------------------------
# Atomic / hashing helpers
# ---------------------------------------------------------------------------
def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a temp file + atomic rename."""
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Serialise ``payload`` to ``path`` as JSON via a temp file + atomic rename."""
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str))


def _read_stage_meta(stage_dir: Path) -> dict[str, Any] | None:
    meta_path = stage_dir / _STAGE_META_NAME
    if not meta_path.is_file():
        return None
    try:
        loaded: Any = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_input_hash(stage: Stage, ctx: StageContext) -> str:
    """Content hash that changes whenever a stage's inputs change."""
    digest = hashlib.sha256()
    digest.update(stage.name.encode("utf-8"))
    digest.update(_file_checksum(ctx.video_path).encode("utf-8"))
    digest.update(json.dumps(ctx.config, sort_keys=True, default=str).encode("utf-8"))
    digest.update(ctx.run_metadata.git_commit.encode("utf-8"))
    for dep in stage.inputs:
        upstream = ctx.upstream.get(dep)
        if upstream is not None:
            digest.update(f"{dep}:{upstream.input_hash}".encode())
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def build_run_metadata(config: AppConfig, *, config_path: str = "") -> RunMetadata:
    """Build reproducibility metadata from the resolved configuration."""
    return RunMetadata(
        git_commit=_git_commit(),
        config=config_path,
        resolved_config=config.model_dump(mode="json"),
    )


# ---------------------------------------------------------------------------
# Canonical output writers
# ---------------------------------------------------------------------------
def _collect_predictions(results: dict[str, StageResult]) -> list[dict[str, Any]]:
    """Gather prediction rows contributed by detector stages."""
    rows: list[dict[str, Any]] = []
    for result in results.values():
        for row in result.summary.get("predictions", []):
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_canonical_events(output_dir: Path, results: dict[str, StageResult]) -> Path:
    """Write the canonical ``events.csv`` (header always present)."""
    path = output_dir / "events.csv"
    rows = _collect_predictions(results)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CANONICAL_EVENT_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CANONICAL_EVENT_COLUMNS})
    os.replace(tmp, path)
    return path


def validate_events_csv(path: Path) -> bool:
    """Return ``True`` when ``events.csv`` matches the canonical schema."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CANONICAL_EVENT_COLUMNS:
            return False
        for row in reader:
            try:
                float(row["t_start"])
                float(row["t_end"])
                score = float(row["score"])
            except (KeyError, TypeError, ValueError):
                return False
            if not 0.0 <= score <= 1.0:
                return False
            if row["type"] not in ("pickup", "putdown"):
                return False
    return True


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def _run_stage_atomic(stage: Stage, ctx: StageContext, input_hash: str) -> StageResult:
    """Run a stage into a partial dir, then promote it and write the marker last."""
    final_dir = ctx.stage_dir
    partial_dir = final_dir.with_name(f"{final_dir.name}.partial-{os.getpid()}")
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    partial_dir.mkdir(parents=True)

    run_ctx = replace(ctx, stage_dir=partial_dir)
    try:
        result = stage.run(run_ctx)
    except Exception:
        shutil.rmtree(partial_dir, ignore_errors=True)
        raise

    # Promote the completed outputs into their final location.
    if final_dir.exists():
        shutil.rmtree(final_dir)
    os.replace(partial_dir, final_dir)

    # Marker LAST: its presence now implies the outputs are complete.
    result.input_hash = input_hash
    atomic_write_json(
        final_dir / _STAGE_META_NAME,
        {
            "name": stage.name,
            "status": result.status,
            "input_hash": input_hash,
            "outputs": result.outputs,
            "summary": result.summary,
        },
    )
    return result


def _gate_status(stage: Stage, results: dict[str, StageResult]) -> tuple[StageStatus, str] | None:
    """Decide whether ``stage`` may run given its upstream results.

    Returns ``None`` when every declared input is satisfied. Otherwise returns
    the status to record for the gated stage plus the offending dependency: a
    *failed* (or missing) upstream is a hard dependency failure (``"blocked"``),
    while a merely *unavailable*/blocked upstream cascades unavailability through
    the subtree (``"unavailable"``). This keeps a real detector with
    ``inputs=("propose",)`` from running—and crashing—when propose produced no
    outputs.
    """
    for dep in stage.inputs:
        upstream = results.get(dep)
        if upstream is None:
            return ("blocked", dep)
        if upstream.status in _SATISFIED_INPUT_STATUSES:
            continue
        if upstream.status == "failed":
            return ("blocked", dep)
        return ("unavailable", dep)
    return None


def run_pipeline(
    video_path: Path,
    output_root: Path,
    config: AppConfig,
    *,
    registry: list[Stage] | None = None,
    resume: bool = True,
    run_metadata: RunMetadata | None = None,
) -> dict[str, Any]:
    """Run the full pipeline over a single video and return a JSON-able summary."""
    # Validate the input before touching the filesystem so a bad path cannot
    # leave behind an empty output directory (or fail later inside checksumming).
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")

    stages = registry if registry is not None else build_default_registry(config)
    clip_id = f"clip_{video_path.stem}"
    output_dir = output_root / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cross-clip cache root (frozen embeddings, etc.): derived from config so it
    # is shared across every clip in a batch and across runs, not per-clip.
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    metadata = run_metadata if run_metadata is not None else build_run_metadata(config)
    resolved = metadata.resolved_config
    atomic_write_json(output_dir / "run_metadata.json", metadata.to_dict())

    # Materialise the resolved config so CLI-based stages run under the exact
    # configuration that feeds the input hash (keeping behaviour and resume
    # honest, instead of letting each subprocess reload its own YAML default).
    config_path = output_dir / "resolved_config.yaml"
    _atomic_write_text(config_path, yaml.safe_dump(resolved, sort_keys=True))

    results: dict[str, StageResult] = {}
    early_complete = False

    for stage in stages:
        stage_dir = output_dir / stage.name
        ctx = StageContext(
            clip_id=clip_id,
            video_path=video_path,
            output_dir=output_dir,
            stage_dir=stage_dir,
            cache_dir=cache_dir,
            config=resolved,
            config_path=config_path,
            run_metadata=metadata,
            upstream=dict(results),
        )

        if not stage.is_available():
            logger.info("stage %s is unavailable; skipping", stage.name)
            results[stage.name] = StageResult(stage.name, "unavailable")
            continue

        gate = _gate_status(stage, results)
        if gate is not None:
            gated_status, dep = gate
            logger.info("stage %s %s: upstream %r not satisfied", stage.name, gated_status, dep)
            results[stage.name] = StageResult(
                stage.name, gated_status, summary={"blocked_on": dep}
            )
            continue

        input_hash = _stage_input_hash(stage, ctx)
        cached = _read_stage_meta(stage_dir)
        if (
            resume
            and cached is not None
            and cached.get("input_hash") == input_hash
            and cached.get("status") in ("ok", "no_person")
        ):
            logger.info("stage %s resumed from cache", stage.name)
            # Preserve a cached "no_person" outcome so early completion stays
            # status-driven on resume too; everything else reads as "resumed".
            cached_status = cached.get("status")
            resumed_status: StageStatus = (
                "no_person" if cached_status == "no_person" else "resumed"
            )
            results[stage.name] = StageResult(
                name=stage.name,
                status=resumed_status,
                outputs=dict(cached.get("outputs", {})),
                summary=dict(cached.get("summary", {})),
                input_hash=input_hash,
            )
        else:
            results[stage.name] = _run_stage_atomic(stage, ctx, input_hash)

        if stage.name == _TRIAGE_STAGE and results[stage.name].status == "no_person":
            logger.info("no person detected in %s; completing early", clip_id)
            early_complete = True
            break

    # A blocked/failed stage must not let the run report success.
    all_statuses = {result.status for result in results.values()}
    status: StageStatus
    if early_complete:
        status = "no_person"
    elif "failed" in all_statuses:
        status = "failed"
    elif "blocked" in all_statuses:
        status = "blocked"
    else:
        status = "ok"
    events_path = _write_canonical_events(output_dir, results)
    return _write_summary(output_dir, clip_id, status, results, events_path, metadata)


def _write_summary(
    output_dir: Path,
    clip_id: str,
    status: str,
    results: dict[str, StageResult],
    events_path: Path,
    metadata: RunMetadata,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "clip_id": clip_id,
        "status": status,
        "run_id": metadata.run_id,
        "git_commit": metadata.git_commit,
        "events_csv": str(events_path),
        "n_events": len(_collect_predictions(results)),
        "events_valid": validate_events_csv(events_path),
        "stages": {
            name: {
                "status": result.status,
                "outputs": result.outputs,
                "summary": {k: v for k, v in result.summary.items() if k != "predictions"},
            }
            for name, result in results.items()
        },
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------
class _CliStage:
    """Base for stages that shell out to an existing ``pickup_putdown.cli`` command."""

    name: str = ""
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def is_available(self) -> bool:  # pragma: no cover - overridden
        return False

    def _invoke(self, args: list[str], config_path: Path) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "pickup_putdown.cli", *args, "--config", str(config_path)],
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"stage {self.name} failed (exit {completed.returncode})")


class TriageStage(_CliStage):
    """Stage A: person detection + active-span triage (task_3)."""

    name = "triage"
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ("tracks_person.parquet", "active_spans.parquet", "clips.parquet")

    def is_available(self) -> bool:
        return Path(self._config.triage.model_path).is_file()

    def run(self, ctx: StageContext) -> StageResult:
        self._invoke(
            ["triage", str(ctx.video_path), "--output-dir", str(ctx.stage_dir)],
            ctx.config_path,
        )
        has_person = _clips_have_person(ctx.stage_dir / "clips.parquet")
        return StageResult(
            name=self.name,
            status="no_person" if not has_person else "ok",
            outputs={name: str(ctx.stage_dir / name) for name in self.outputs},
            summary={"has_person": has_person},
        )


class ProposeStage(_CliStage):
    """Stage B: pose-based interaction proposals (task_5)."""

    name = "propose"
    inputs: tuple[str, ...] = ("triage",)
    outputs: tuple[str, ...] = ("candidates.parquet", "tracks_pose.parquet")

    def is_available(self) -> bool:
        return Path(self._config.pose.model_path).is_file()

    def run(self, ctx: StageContext) -> StageResult:
        triage = ctx.upstream["triage"]
        self._invoke(
            [
                "propose",
                str(ctx.video_path),
                "--person-tracks",
                triage.outputs["tracks_person.parquet"],
                "--active-spans",
                triage.outputs["active_spans.parquet"],
                "--output-dir",
                str(ctx.stage_dir),
            ],
            ctx.config_path,
        )
        return StageResult(
            name=self.name,
            status="ok",
            outputs={name: str(ctx.stage_dir / name) for name in self.outputs},
        )


class EvaluateStage:
    """Aggregate detector predictions and score them with the task_8 evaluator.

    With no detector stage active yet (Track A/B land in later tasks) there are
    no predictions to score; the stage still writes a valid, zero-prediction
    metrics file so downstream tooling has a stable artifact.
    """

    name = "evaluate"
    inputs: tuple[str, ...] = ("propose",)
    outputs: tuple[str, ...] = ("metrics.json",)

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def is_available(self) -> bool:
        return True

    def run(self, ctx: StageContext) -> StageResult:
        predictions: list[dict[str, Any]] = []
        for result in ctx.upstream.values():
            for row in result.summary.get("predictions", []):
                if isinstance(row, dict):
                    predictions.append(row)
        metrics = {"n_predictions": len(predictions), "evaluated": False}
        atomic_write_json(ctx.stage_dir / "metrics.json", metrics)
        return StageResult(
            name=self.name,
            status="ok",
            outputs={"metrics.json": str(ctx.stage_dir / "metrics.json")},
            summary=metrics,
        )


@dataclass
class ComponentStub:
    """Placeholder for a pipeline component implemented by a later task."""

    name: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    depends_on_task: str = ""

    def is_available(self) -> bool:
        return False

    def run(self, ctx: StageContext) -> StageResult:  # pragma: no cover - never called
        raise RuntimeError(f"stage {self.name} is not implemented yet ({self.depends_on_task})")


def build_default_registry(config: AppConfig) -> list[Stage]:
    """Ordered default registry: triage/propose/evaluate active, the rest stubbed."""
    return [
        TriageStage(config),
        ProposeStage(config),
        ComponentStub("track_a", inputs=("propose",), depends_on_task="task_9/task_10"),
        ComponentStub("track_b1", inputs=("propose",), depends_on_task="task_12"),
        ComponentStub("track_b2", inputs=("propose",), depends_on_task="task_13"),
        ComponentStub("layer2", inputs=("triage",), depends_on_task="task_14"),
        ComponentStub("layer3", inputs=("layer2",), depends_on_task="task_15"),
        EvaluateStage(config),
    ]


def _clips_have_person(clips_path: Path) -> bool:
    """Best-effort read of the triage ``clips.parquet`` ``has_person`` column."""
    if not clips_path.is_file():
        return False
    import pyarrow.parquet as pq

    table = pq.read_table(clips_path, columns=["has_person"])  # type: ignore[no-untyped-call]
    return any(bool(value) for value in table.column("has_person").to_pylist())
