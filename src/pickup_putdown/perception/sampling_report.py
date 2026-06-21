"""Triage-quality sampling report generation."""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pickup_putdown.common.schemas import ActiveSpan, PersonObservation

logger = logging.getLogger(__name__)


def _select_qa_clips(
    clip_results: list[dict],
    sampling_seed: int,
    sample_rate: float,
) -> set[str]:
    """Deterministically select no-person clips for QA review.

    Parameters
    ----------
    clip_results : list[dict]
        Triage results per clip with 'clip_id' and 'decision' keys.
    sampling_seed : int
        Random seed for deterministic selection.
    sample_rate : float
        Fraction of no-person clips to select (0.0-1.0).

    Returns
    -------
    set[str]
        Set of clip_ids selected for QA.
    """
    no_person_clips = [r for r in clip_results if r["decision"] == "no_person"]
    if not no_person_clips:
        return set()

    n_select = max(1, math.ceil(len(no_person_clips) * sample_rate))

    # Deterministic selection using stable hash of clip_id + seed
    scored: list[tuple[str, float]] = []
    for clip in no_person_clips:
        h = hashlib.sha256(f"{sampling_seed}:{clip['clip_id']}".encode()).hexdigest()
        scored.append((clip["clip_id"], int(h, 16) % 10000))

    scored.sort(key=lambda x: x[1])
    selected = {cid for cid, _ in scored[:n_select]}
    return selected


def generate_sampling_report(
    clip_results: list[dict],
    sampling_seed: int = 42,
    sample_rate: float = 0.10,
) -> list[dict]:
    """Generate a machine-readable triage sampling report.

    Parameters
    ----------
    clip_results : list[dict]
        Per-clip triage results. Each dict must have:
        clip_id, decision, source_duration_s, target_fps,
        effective_sample_fps, n_raw_tracks, n_stable_tracks, n_observations.
    sampling_seed : int
        Seed for deterministic QA selection.
    sample_rate : float
        Fraction of no-person clips to select.

    Returns
    -------
    list[dict]
        Report rows with selection flags.
    """
    selected_qa = _select_qa_clips(clip_results, sampling_seed, sample_rate)

    report: list[dict] = []
    for result in clip_results:
        clip_id = result["clip_id"]
        selected = clip_id in selected_qa

        if result["decision"] == "no_person":
            reason = "auto_rejected" if not selected else "qa_sample"
        else:
            reason = "person_detected"

        report.append(
            {
                "clip_id": clip_id,
                "decision": result["decision"],
                "selected_for_qa": selected,
                "selection_reason": reason,
                "preview_path": result.get("preview_path"),
                "source_duration_s": result["source_duration_s"],
                "target_fps": result["target_fps"],
                "effective_sample_fps": result["effective_sample_fps"],
                "n_raw_tracks": result["n_raw_tracks"],
                "n_stable_tracks": result["n_stable_tracks"],
                "n_observations": result["n_observations"],
                "review_status": "flagged"
                if selected and result["decision"] == "no_person"
                else "ok",
            }
        )

    return report


def write_sampling_report(
    report: list[dict],
    output_path: Path,
) -> Path:
    """Write the sampling report to Parquet.

    Parameters
    ----------
    report : list[dict]
        Report rows from generate_sampling_report().
    output_path : Path
        Destination .parquet file.

    Returns
    -------
    Path
        The output file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not report:
        # Write empty Parquet with correct schema
        schema = pa.schema(
            [
                ("clip_id", pa.string()),
                ("decision", pa.string()),
                ("selected_for_qa", pa.bool_()),
                ("selection_reason", pa.string()),
                ("preview_path", pa.string()),
                ("source_duration_s", pa.float64()),
                ("target_fps", pa.float64()),
                ("effective_sample_fps", pa.float64()),
                ("n_raw_tracks", pa.int64()),
                ("n_stable_tracks", pa.int64()),
                ("n_observations", pa.int64()),
                ("review_status", pa.string()),
            ]
        )
        table = pa.Table.from_pydict({k: [] for k in schema.names}, schema=schema)
    else:
        table = pa.Table.from_pylist(report)

    pq.write_table(table, str(output_path))
    logger.info("Sampling report written: %s (%d rows)", output_path, len(report))
    return output_path


def observations_to_report_rows(
    observations: list[PersonObservation],
    spans: list[ActiveSpan],
    clip_id: str,
    clip_duration_s: float,
    target_fps: float,
    effective_sample_fps: float,
    n_raw_tracks: int,
    n_stable_tracks: int,
    preview_path: str | None = None,
) -> dict:
    """Convert triage results into a sampling report row."""
    has_person = len(spans) > 0
    return {
        "clip_id": clip_id,
        "decision": "person_detected" if has_person else "no_person",
        "source_duration_s": clip_duration_s,
        "target_fps": target_fps,
        "effective_sample_fps": effective_sample_fps,
        "n_raw_tracks": n_raw_tracks,
        "n_stable_tracks": n_stable_tracks,
        "n_observations": len(observations),
        "preview_path": preview_path,
    }
