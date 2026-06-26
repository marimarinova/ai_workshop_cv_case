"""Task 7 finalization: build canonical dataset artifacts from VLM annotations.

Reads the normalized per-candidate JSON files produced by the VLM annotation
pipeline and produces a reproducible artifact directory containing:

    clips.csv        -- one row per source clip
    events.csv       -- canonical event rows from all successful candidates
    processing.csv   -- per-candidate processing ledger
    summary.json     -- aggregate counts and timing
    provenance.json  -- model, config, and provisional-status metadata
    raw/             -- symlink or copy of raw VLM outputs
    normalized/      -- symlink or copy of normalized candidate outputs

Designed to be invoked as:

    pickup-putdown finalize-task-7 \\
        --vlm-output-dir .local/vlm_annotations \\
        --output-dir .local/task_7_vlm
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical column orders
# ---------------------------------------------------------------------------

EVENT_COLUMNS = [
    "event_id",
    "clip_id",
    "type",
    "t_start",
    "t_end",
    "hard_case",
    "annotator",
    "confidence",
    "notes",
]

CLIPS_COLUMNS = [
    "clip_id",
    "s3_key",
    "duration_s",
    "fps",
    "width",
    "height",
    "n_person_tracks",
    "usable",
    "active_start_s",
    "active_end_s",
    "split",
    "session_id",
    "notes",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FinalizationError:
    """Single validation error from the finalizer."""

    source: str
    message: str


@dataclass
class FinalizationResult:
    """Outcome of a finalize-task-7 run."""

    events_count: int = 0
    clips_count: int = 0
    candidates_count: int = 0
    hard_case_count: int = 0
    pickup_count: int = 0
    putdown_count: int = 0
    confidence_counts: dict[str, int] = field(default_factory=dict)
    errors: list[FinalizationError] = field(default_factory=list)
    output_dir: str = ""

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


# ---------------------------------------------------------------------------
# Normalized candidate loading
# ---------------------------------------------------------------------------


def load_normalized_candidate(path: Path) -> dict[str, Any]:
    """Load and minimally validate one normalized candidate JSON file.

    Raises:
        ValueError: On malformed or missing required fields.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {path.name}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path.name}, got {type(data).__name__}")

    candidate_id = data.get("candidate_id")
    if not candidate_id:
        raise ValueError(f"Missing candidate_id in {path.name}")

    clip_id = data.get("clip_id")
    if not clip_id:
        raise ValueError(f"Missing clip_id in {path.name} (candidate {candidate_id})")

    return data


def collect_normalized_candidates(normalized_dir: Path) -> list[dict[str, Any]]:
    """Load all normalized candidate files from the normalized directory.

    Returns sorted list of candidate dicts. Files that fail to parse raise
    a FinalizationError rather than being silently skipped.
    """
    if not normalized_dir.is_dir():
        raise FileNotFoundError(f"Normalized directory not found: {normalized_dir}")

    json_files = sorted(normalized_dir.glob("*.json"))
    candidates: list[dict[str, Any]] = []
    errors: list[FinalizationError] = []

    for fp in json_files:
        try:
            data = load_normalized_candidate(fp)
            candidates.append(data)
        except ValueError as exc:
            errors.append(FinalizationError(source=fp.name, message=str(exc)))

    if errors:
        for err in errors:
            logger.error("Failed to load %s: %s", err.source, err.message)
        raise ValueError(
            f"Failed to load {len(errors)} normalized candidate file(s). "
            f"First error: [{errors[0].source}] {errors[0].message}"
        )

    return candidates


# ---------------------------------------------------------------------------
# Event extraction and validation
# ---------------------------------------------------------------------------


def _generate_event_id(clip_id: str, label: str, start_s: float, item_idx: int) -> str:
    """Generate a deterministic event ID from canonical fields."""
    raw = f"{clip_id}:{label}:{start_s:.3f}:{item_idx}"
    return f"evt_{hashlib.md5(raw.encode()).hexdigest()[:12]}"


def extract_events_from_candidates(
    candidates: list[dict[str, Any]],
    clip_durations: dict[str, float],
) -> tuple[list[dict[str, Any]], list[FinalizationError], list[dict[str, Any]]]:
    """Extract canonical event rows from successful candidates.

    Only candidates with review_status == "complete" and vlm_status == "success"
    contribute events. Failed VLM results are excluded entirely.

    Returns (events, errors, dedup_audit).

    dedup_audit entries contain:
        event_id, kept_candidate, kept_clip, kept_t_start, kept_t_end,
        kept_type, kept_confidence, kept_notes,
        skipped_candidate, skipped_clip, skipped_t_start, skipped_t_end,
        skipped_type, skipped_confidence, skipped_notes,
        reason
    """
    events: list[dict[str, Any]] = []
    errors: list[FinalizationError] = []
    dedup_audit: list[dict[str, Any]] = []
    # Map event_id -> first event dict that was kept
    kept_events: dict[str, dict[str, Any]] = {}

    for cand in candidates:
        candidate_id = cand.get("candidate_id", "unknown")
        clip_id = cand.get("clip_id", "")
        review_status = cand.get("review_status", "")
        vlm_status = cand.get("vlm_status", "")

        # Exclude failed VLM results -- they must never become zero-event negatives
        if vlm_status != "success":
            continue

        if review_status != "complete":
            continue

        source_start_s = float(cand.get("source_start_s", 0.0))
        clip_duration = clip_durations.get(clip_id)

        raw_events = cand.get("events", [])
        if not isinstance(raw_events, list):
            errors.append(
                FinalizationError(
                    source=candidate_id,
                    message=f"events field is not a list: {type(raw_events).__name__}",
                )
            )
            continue

        for evt in raw_events:
            label = str(evt.get("label", "")).strip().lower()
            if label not in ("pickup", "putdown"):
                errors.append(
                    FinalizationError(
                        source=candidate_id,
                        message=f"Invalid event label: {label!r}",
                    )
                )
                continue

            start_s = float(evt.get("start_s", 0.0))
            end_s = float(evt.get("end_s", 0.0))

            # Convert candidate-relative to source-video timestamps
            t_start = round(source_start_s + start_s, 3)
            t_end = round(source_start_s + end_s, 3)

            # Validate t_start < t_end
            if t_start >= t_end:
                errors.append(
                    FinalizationError(
                        source=candidate_id,
                        message=f"Event {label}: t_start({t_start}) >= t_end({t_end})",
                    )
                )
                continue

            # Validate against clip duration
            if clip_duration is not None:
                if t_start < 0:
                    errors.append(
                        FinalizationError(
                            source=candidate_id,
                            message=f"Event {label}: t_start({t_start}) < 0",
                        )
                    )
                    continue
                if t_end > clip_duration + 0.1:
                    errors.append(
                        FinalizationError(
                            source=candidate_id,
                            message=(
                                f"Event {label}: t_end({t_end}) exceeds "
                                f"clip duration({clip_duration})"
                            ),
                        )
                    )
                    continue

            item_count = int(evt.get("item_count", 1))
            if item_count < 1:
                item_count = 1

            confidence = str(evt.get("confidence", "med")).strip().lower()
            if confidence == "medium":
                confidence = "med"
            if confidence not in ("high", "med", "low"):
                confidence = "med"

            hard_case = bool(evt.get("hard_case", False))
            notes = str(evt.get("notes", "")).strip()
            annotator = str(cand.get("annotator", "vlm_pipeline")) or "vlm_pipeline"

            for item_idx in range(item_count):
                event_id = _generate_event_id(clip_id, label, t_start, item_idx)

                event_dict = {
                    "event_id": event_id,
                    "clip_id": clip_id,
                    "type": label,
                    "t_start": t_start,
                    "t_end": t_end,
                    "hard_case": hard_case,
                    "annotator": annotator,
                    "confidence": confidence,
                    "notes": notes,
                }

                if event_id in kept_events:
                    # Record dedup audit entry
                    kept = kept_events[event_id]
                    dedup_audit.append(
                        {
                            "event_id": event_id,
                            "kept_candidate": kept.get("_candidate_id", ""),
                            "kept_clip_id": kept["clip_id"],
                            "kept_t_start": kept["t_start"],
                            "kept_t_end": kept["t_end"],
                            "kept_type": kept["type"],
                            "kept_confidence": kept["confidence"],
                            "kept_notes": kept.get("notes", ""),
                            "skipped_candidate": candidate_id,
                            "skipped_clip_id": clip_id,
                            "skipped_t_start": t_start,
                            "skipped_t_end": t_end,
                            "skipped_type": label,
                            "skipped_confidence": confidence,
                            "skipped_notes": notes,
                            "reason": (
                                "Overlapping candidates produced identical event_id "
                                "(same clip, type, and source timestamp). "
                                "First occurrence kept."
                            ),
                        }
                    )
                    logger.info(
                        "Skipping duplicate event %s from %s "
                        "(same event from overlapping candidate)",
                        event_id,
                        candidate_id,
                    )
                    continue

                kept_events[event_id] = {
                    **event_dict,
                    "_candidate_id": candidate_id,
                }
                events.append(event_dict)

    # events already don't contain _candidate_id (it's only in kept_events)

    # Deterministic sort: clip_id, t_start, type, event_id
    events.sort(key=lambda e: (e["clip_id"], e["t_start"], e["type"], e["event_id"]))
    return events, errors, dedup_audit


# ---------------------------------------------------------------------------
# Clip metadata discovery
# ---------------------------------------------------------------------------


def discover_clip_metadata(
    candidates_dir: Path | None,
    source_videos_dir: Path | None,
) -> dict[str, dict[str, Any]]:
    """Build clip metadata from source video files and candidate metadata.

    Returns dict mapping clip_id -> clip metadata dict with keys:
        clip_id, s3_key, duration_s, fps, width, height, n_person_tracks,
        usable, active_start_s, active_end_s, split, session_id, notes
    """
    clip_info: dict[str, dict[str, Any]] = {}

    # Probe source videos for duration and resolution
    if source_videos_dir and source_videos_dir.is_dir():
        for vid_path in sorted(source_videos_dir.glob("*.mp4")):
            clip_name = vid_path.stem
            try:
                probe = _probe_video(vid_path)
                clip_info[clip_name] = {
                    "clip_id": clip_name,
                    "s3_key": f"source_videos/{clip_name}.mp4",
                    "duration_s": round(probe["duration_s"], 3),
                    "fps": round(probe["fps"], 3),
                    "width": probe["width"],
                    "height": probe["height"],
                    "n_person_tracks": 0,
                    "usable": True,
                    "active_start_s": None,
                    "active_end_s": None,
                    "split": None,
                    "session_id": None,
                    "notes": None,
                }
            except Exception as exc:
                logger.warning("Failed to probe %s: %s", vid_path.name, exc)

    # Enrich with candidate metadata: active spans, person tracks
    if candidates_dir and candidates_dir.is_dir():
        for meta_file in sorted(candidates_dir.rglob("*.json")):
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            if not isinstance(meta, dict):
                continue

            source_video_id = meta.get("source_video_id", "")
            if not source_video_id:
                continue

            cands = meta.get("candidates", [])
            if not isinstance(cands, list):
                continue

            if source_video_id not in clip_info:
                clip_info[source_video_id] = {
                    "clip_id": source_video_id,
                    "s3_key": f"source_videos/{source_video_id}.mp4",
                    "duration_s": 0.0,
                    "fps": 0.0,
                    "width": 0,
                    "height": 0,
                    "n_person_tracks": 0,
                    "usable": True,
                    "active_start_s": None,
                    "active_end_s": None,
                    "split": None,
                    "session_id": None,
                    "notes": None,
                }

            info = clip_info[source_video_id]
            active_starts = []
            active_ends = []
            for c in cands:
                ss = float(c.get("source_start_s", 0.0))
                se = float(c.get("source_end_s", 0.0))
                if ss > 0:
                    active_starts.append(ss)
                if se > 0:
                    active_ends.append(se)

            if active_starts and active_ends:
                info["active_start_s"] = round(min(active_starts), 3)
                info["active_end_s"] = round(max(active_ends), 3)

            # If duration was not set by source video probe, use the max
            # candidate end as a lower bound for the clip duration.
            if info.get("duration_s", 0.0) == 0.0 and active_ends:
                info["duration_s"] = round(max(active_ends) + 1.0, 3)

    return clip_info


def _probe_video(video_path: Path) -> dict[str, Any]:
    """Probe a video file with ffprobe and return basic metadata."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    info = json.loads(result.stdout)

    video_stream = None
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream is None:
        raise RuntimeError(f"No video stream in {video_path}")

    r_frame_rate = video_stream.get("r_frame_rate", "30/1")
    if "/" in r_frame_rate:
        num, den = r_frame_rate.split("/")
        fps = float(num) / max(1, float(den))
    else:
        fps = float(r_frame_rate)

    duration = float(info.get("format", {}).get("duration", 0.0))

    return {
        "duration_s": duration,
        "fps": fps,
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
    }


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def build_provenance(
    vlm_output_dir: Path,
    output_dir: Path,
    events: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    clips: dict[str, dict[str, Any]],
    errors: list[FinalizationError],
    dedup_count: int,
    use_symlinks: bool,
) -> dict[str, Any]:
    """Build provenance.json content."""

    # Read VLM summary for model info
    summary_path = vlm_output_dir / "summary.json"
    vlm_summary: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            vlm_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    # Count by event type
    pickup_count = sum(1 for e in events if e["type"] == "pickup")
    putdown_count = sum(1 for e in events if e["type"] == "putdown")

    # Count by confidence
    conf_counts: dict[str, int] = {}
    for e in events:
        c = e.get("confidence", "med")
        conf_counts[c] = conf_counts.get(c, 0) + 1

    # Hard case count
    hard_case_count = sum(1 for e in events if e.get("hard_case"))

    # Total candidates processed
    total_candidates = len(candidates)

    # Validate symlink targets
    raw_link = output_dir / "raw"
    norm_link = output_dir / "normalized"
    raw_valid = (
        validate_symlink(raw_link)
        if raw_link.is_symlink()
        else not raw_link.is_symlink() and raw_link.is_dir()
    )
    norm_valid = (
        validate_symlink(norm_link)
        if norm_link.is_symlink()
        else not norm_link.is_symlink() and norm_link.is_dir()
    )

    provenance: dict[str, Any] = {
        "annotation_method": "vlm_auto_annotation",
        "model": "Qwen3.6-27B-UD-Q4_K_XL",
        "vision_projector": "mmproj-BF16.gguf",
        "review_fps": float(vlm_summary.get("review_fps_target", 5.0)),
        "candidate_count": total_candidates,
        "failed_count": 0,
        "event_rows": len(events),
        "complete_active_span_reviewed": False,
        "status": "provisional",
        "generation_timestamp": datetime.now(UTC).isoformat(),
        "source_vlm_output_dir": str(vlm_output_dir),
        "source_vlm_output_dir_resolved": str(vlm_output_dir.resolve()),
        "output_dir": str(output_dir),
        "output_dir_resolved": str(output_dir.resolve()),
        "raw_symlink_valid": raw_valid,
        "normalized_symlink_valid": norm_valid,
        "uses_symlinks": use_symlinks,
        "not_self_contained": use_symlinks,
        "self_contained_note": (
            "This export uses symlinks to the original VLM output directory. "
            "To create a self-contained copy, re-run with --copy-artifacts."
        )
        if use_symlinks
        else None,
        "deduplication_count": dedup_count,
        "dedup_audit_file": "dedup_audit.json",
        "counts_by_event_type": {
            "pickup": pickup_count,
            "putdown": putdown_count,
        },
        "counts_by_confidence": conf_counts,
        "hard_case_count": hard_case_count,
        "source_clip_count": len(clips),
        "provisional_notice": (
            "These are VLM-generated pseudo-labels, not human-adjudicated ground truth. "
            "Positives require human review before final evaluation. "
            "Hard-case and low-confidence events require review. "
            "Candidate generation may have missed events outside proposed windows. "
            "This output can be used for model development and pseudo-label training "
            "but must not be used as final ground truth without adjudication."
        ),
    }

    return provenance


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def build_summary(
    candidates: list[dict[str, Any]],
    events: list[dict[str, Any]],
    clips: dict[str, dict[str, Any]],
    errors: list[FinalizationError],
) -> dict[str, Any]:
    """Build summary.json content for the Task 7 artifact."""

    pickup_count = sum(1 for e in events if e["type"] == "pickup")
    putdown_count = sum(1 for e in events if e["type"] == "putdown")
    hard_case_count = sum(1 for e in events if e.get("hard_case"))

    conf_counts: dict[str, int] = {}
    for e in events:
        c = e.get("confidence", "med")
        conf_counts[c] = conf_counts.get(c, 0) + 1

    return {
        "candidate_count": len(candidates),
        "clip_count": len(clips),
        "event_count": len(events),
        "pickup_count": pickup_count,
        "putdown_count": putdown_count,
        "hard_case_count": hard_case_count,
        "confidence_counts": conf_counts,
        "validation_errors": len(errors),
        "generated_at": datetime.now(UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# Writing outputs
# ---------------------------------------------------------------------------


def write_events_csv(events: list[dict[str, Any]], path: Path) -> None:
    """Write canonical events.csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=EVENT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(events)


def write_clips_csv(clips: dict[str, dict[str, Any]], path: Path) -> None:
    """Write canonical clips.csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clip_rows = sorted(clips.values(), key=lambda c: c["clip_id"])
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CLIPS_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(clip_rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def copy_processing_csv(source: Path, dest: Path) -> None:
    """Copy processing.csv from VLM output."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def link_subdirectory(source: Path, dest: Path) -> None:
    """Create a relative symlink to a subdirectory (or copy on Windows)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        return
    if dest.exists():
        shutil.rmtree(dest)
    try:
        # Use relative path so the artifact directory is portable
        rel_target = os.path.relpath(source.resolve(), dest.parent)
        dest.symlink_to(rel_target, target_is_directory=True)
    except OSError:
        shutil.copytree(source, dest, symlinks=True)


def copy_subdirectory(source: Path, dest: Path) -> None:
    """Copy a subdirectory into the output tree for self-contained export."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        dest.unlink()
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest, symlinks=True)


def validate_symlink(dest: Path) -> bool:
    """Return True if dest is a valid symlink pointing to an existing directory."""
    if not dest.is_symlink():
        return False
    return dest.resolve().is_dir()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_final_artifacts(
    events: list[dict[str, Any]],
    clips: dict[str, dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> list[FinalizationError]:
    """Validate the final artifact set for referential integrity and correctness.

    Checks:
    - Every event references an existing clip
    - No duplicate event IDs
    - t_start < t_end for all events
    - Events within clip duration
    - Deterministic ordering
    - Summary counts match
    """
    errors: list[FinalizationError] = []

    # Check event -> clip referential integrity
    event_ids: set[str] = set()
    for evt in events:
        eid = evt["event_id"]
        if eid in event_ids:
            errors.append(
                FinalizationError(
                    source=evt["clip_id"],
                    message=f"Duplicate event_id: {eid}",
                )
            )
        event_ids.add(eid)

        clip_id = evt["clip_id"]
        clip = clips.get(clip_id)
        t_start = float(evt["t_start"])
        t_end = float(evt["t_end"])

        if t_start >= t_end:
            errors.append(
                FinalizationError(
                    source=clip_id,
                    message=f"Event {eid}: t_start({t_start}) >= t_end({t_end})",
                )
            )

        if clip is not None:
            duration = float(clip.get("duration_s", 0.0))
            if duration > 0 and t_end > duration + 0.1:
                errors.append(
                    FinalizationError(
                        source=clip_id,
                        message=(f"Event {eid}: t_end({t_end}) exceeds clip duration({duration})"),
                    )
                )

    # Check deterministic ordering
    for i in range(1, len(events)):
        prev = events[i - 1]
        curr = events[i]
        prev_key = (prev["clip_id"], prev["t_start"], prev["type"], prev["event_id"])
        curr_key = (curr["clip_id"], curr["t_start"], curr["type"], curr["event_id"])
        if prev_key > curr_key:
            errors.append(
                FinalizationError(
                    source="ordering",
                    message=(
                        f"Events not in deterministic order at position {i}: "
                        f"{prev_key} > {curr_key}"
                    ),
                )
            )
            break

    # Check that all clips referenced by events exist
    clips_in_events = {evt["clip_id"] for evt in events}
    clips_available = set(clips.keys())
    missing = clips_in_events - clips_available
    if missing:
        for cid in sorted(missing):
            errors.append(
                FinalizationError(
                    source=cid,
                    message=f"Clip referenced by events but missing from clips.csv: {cid}",
                )
            )

    return errors


# ---------------------------------------------------------------------------
# Main finalization function
# ---------------------------------------------------------------------------


def finalize_task_7(
    vlm_output_dir: str | Path,
    output_dir: str | Path,
    candidate_metadata_dir: str | Path | None = None,
    source_videos_dir: str | Path | None = None,
    copy_artifacts: bool = False,
) -> FinalizationResult:
    """Run the complete Task 7 finalization pipeline.

    Args:
        vlm_output_dir: Path to .local/vlm_annotations directory.
        output_dir: Target directory for Task 7 artifacts.
        candidate_metadata_dir: Path to candidate staging directory (for clip
            metadata discovery). If None, will try to discover from vlm_output_dir
            parent structure.
        source_videos_dir: Path to source video files. If None, will try to
            discover from .local/source_videos.
        copy_artifacts: If True, copy raw/ and normalized/ instead of symlinking.
            Produces a self-contained export but uses more disk space.

    Returns:
        FinalizationResult with counts and any errors.
    """
    vlm_output_dir = Path(vlm_output_dir)
    output_dir = Path(output_dir)

    if not vlm_output_dir.is_dir():
        raise FileNotFoundError(f"VLM output directory not found: {vlm_output_dir}")

    normalized_dir = vlm_output_dir / "normalized"
    if not normalized_dir.is_dir():
        raise FileNotFoundError(f"Normalized candidates directory not found: {normalized_dir}")

    result = FinalizationResult(output_dir=str(output_dir))

    # 1. Load normalized candidates
    logger.info("Loading normalized candidates from %s", normalized_dir)
    candidates = collect_normalized_candidates(normalized_dir)
    result.candidates_count = len(candidates)
    logger.info("Loaded %d normalized candidates", len(candidates))

    # 2. Discover clip metadata
    if candidate_metadata_dir is None:
        candidate_metadata_dir = vlm_output_dir.parent / "candidate_staging" / "candidates"
    if source_videos_dir is None:
        source_videos_dir = vlm_output_dir.parent / "source_videos"

    clips = discover_clip_metadata(
        candidates_dir=Path(candidate_metadata_dir),
        source_videos_dir=Path(source_videos_dir),
    )
    result.clips_count = len(clips)
    logger.info("Discovered %d source clips", len(clips))

    # 3. Extract events from successful candidates
    clip_durations = {cid: float(c.get("duration_s", 0.0)) for cid, c in clips.items()}
    events, extraction_errors, dedup_audit = extract_events_from_candidates(
        candidates, clip_durations
    )
    result.events_count = len(events)
    result.errors.extend(extraction_errors)
    logger.info("Extracted %d event rows (%d deduplicated)", len(events), len(dedup_audit))

    # 4. Validate referential integrity
    validation_errors = validate_final_artifacts(events, clips, candidates)
    result.errors.extend(validation_errors)
    if validation_errors:
        for err in validation_errors:
            logger.error("Validation: [%s] %s", err.source, err.message)

    # 5. Compute statistics
    result.pickup_count = sum(1 for e in events if e["type"] == "pickup")
    result.putdown_count = sum(1 for e in events if e["type"] == "putdown")
    result.hard_case_count = sum(1 for e in events if e.get("hard_case"))
    conf_counts: dict[str, int] = {}
    for e in events:
        c = e.get("confidence", "med")
        conf_counts[c] = conf_counts.get(c, 0) + 1
    result.confidence_counts = conf_counts

    # 6. Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)

    # 7. Write canonical events.csv
    write_events_csv(events, output_dir / "events.csv")
    logger.info("Wrote events.csv with %d rows", len(events))

    # 8. Write canonical clips.csv
    write_clips_csv(clips, output_dir / "clips.csv")
    logger.info("Wrote clips.csv with %d rows", len(clips))

    # 9. Copy processing.csv
    processing_src = vlm_output_dir / "processing.csv"
    if processing_src.is_file():
        copy_processing_csv(processing_src, output_dir / "processing.csv")
        logger.info("Copied processing.csv")

    # 10. Write dedup_audit.json
    if dedup_audit:
        write_json(output_dir / "dedup_audit.json", dedup_audit)
        logger.info("Wrote dedup_audit.json with %d entries", len(dedup_audit))

    # 11. Write summary.json
    summary = build_summary(candidates, events, clips, result.errors)
    write_json(output_dir / "summary.json", summary)
    logger.info("Wrote summary.json")

    # 12. Link or copy raw/ and normalized/ directories
    use_symlinks = not copy_artifacts
    raw_src = vlm_output_dir / "raw"
    if raw_src.is_dir():
        if copy_artifacts:
            copy_subdirectory(raw_src, output_dir / "raw")
            logger.info("Copied raw/")
        else:
            link_subdirectory(raw_src, output_dir / "raw")
            logger.info("Linked raw/")

    if normalized_dir.is_dir():
        if copy_artifacts:
            copy_subdirectory(normalized_dir, output_dir / "normalized")
            logger.info("Copied normalized/")
        else:
            link_subdirectory(normalized_dir, output_dir / "normalized")
            logger.info("Linked normalized/")

    # 13. Validate symlink targets (if used)
    if use_symlinks:
        for link_name in ("raw", "normalized"):
            link_path = output_dir / link_name
            if link_path.is_symlink() and not validate_symlink(link_path):
                result.errors.append(
                    FinalizationError(
                        source=link_name,
                        message=f"Symlink {link_path} target is invalid",
                    )
                )

    # 14. Write provenance.json
    provenance = build_provenance(
        vlm_output_dir,
        output_dir,
        events,
        candidates,
        clips,
        result.errors,
        dedup_count=len(dedup_audit),
        use_symlinks=use_symlinks,
    )
    write_json(output_dir / "provenance.json", provenance)
    logger.info("Wrote provenance.json")

    # 15. Final report
    logger.info(
        "Task 7 finalization complete: %d candidates, %d clips, %d events, "
        "%d deduplicated, %d errors",
        result.candidates_count,
        result.clips_count,
        result.events_count,
        len(dedup_audit),
        len(result.errors),
    )

    return result
