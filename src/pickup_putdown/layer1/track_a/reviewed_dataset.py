"""Build the reviewed Track A feature dataset from manually reviewed Task 7 data.

Pipeline:
1. Load and validate review manifest, events CSV, clips CSV
2. Resolve reviewed examples (positives → match events, negatives → confirmed zero-event)
3. Assign train/val/test splits by recording day
4. Run pose inference on source video windows
5. Extract features via existing build_feature_dataset() with label overrides
6. Save manifest, cached embeddings, split metadata, build summary
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pickup_putdown.common.schemas import Candidate, Event, PoseObservation
    from pickup_putdown.config import PoseConfig, TrackAFeaturesConfig
    from pickup_putdown.layer1.track_a.contracts import FeatureDataset
    from pickup_putdown.layer1.track_a.image_features import AbstractImageEmbedder
    from pickup_putdown.perception.shelf_regions import Polygon

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ReviewRecord:
    """One row from the review manifest."""

    candidate_id: str
    clip_id: str
    review_groups: str
    video_path: str
    json_path: str
    event_count: int
    reviewed: bool
    review_notes: str


@dataclass
class CandidateMetadata:
    """Candidate metadata from the candidate staging JSON."""

    candidate_id: str
    clip_id: str
    source_start_s: float
    source_end_s: float
    duration_s: float
    actor_id: str | None = None
    hand_side: str | None = None
    region_id: str | None = None


@dataclass
class ReviewedExample:
    """A resolved reviewed example with label and matched events."""

    candidate_id: str
    clip_id: str
    label: str  # "pickup", "putdown", or "negative"
    source_start_s: float
    source_end_s: float
    matched_event_ids: list[str] = field(default_factory=list)
    matched_event_labels: list[str] = field(default_factory=list)
    actor_id: str | None = None
    hand_side: str | None = None
    region_id: str | None = None
    review_status: str = "reviewed"
    review_notes: str = ""


@dataclass
class BuildSummary:
    """Summary of the reviewed dataset build."""

    total_reviewed: int = 0
    positives: int = 0
    negatives: int = 0
    excluded_unreviewed: int = 0
    excluded_no_match: int = 0
    no_pose: int = 0
    errors: list[str] = field(default_factory=list)
    records_by_split: dict[str, int] = field(default_factory=dict)
    records_by_label: dict[str, int] = field(default_factory=dict)
    records_by_position: dict[str, int] = field(default_factory=dict)
    records_by_crop_type: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


def load_review_manifest(manifest_path: Path | str) -> list[ReviewRecord]:
    """Load the review manifest CSV.

    Args:
        manifest_path: Path to review_manifest.csv.

    Returns:
        List of ReviewRecord objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Review manifest not found: {manifest_path}")

    required_cols = {
        "candidate_id",
        "clip_id",
        "review_groups",
        "video_path",
        "json_path",
        "event_count",
        "reviewed",
        "review_notes",
    }

    records: list[ReviewRecord] = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty review manifest: {manifest_path}")

        actual_cols = set(reader.fieldnames)
        missing = required_cols - actual_cols
        if missing:
            raise ValueError(f"Review manifest missing columns: {sorted(missing)}")

        for _i, row in enumerate(reader):
            try:
                event_count = int(row.get("event_count", 0))
            except ValueError:
                event_count = 0

            reviewed_str = row.get("reviewed", "").strip().lower()
            reviewed = reviewed_str in ("true", "1", "yes")

            records.append(
                ReviewRecord(
                    candidate_id=row["candidate_id"].strip(),
                    clip_id=row["clip_id"].strip(),
                    review_groups=row.get("review_groups", "").strip(),
                    video_path=row.get("video_path", "").strip(),
                    json_path=row.get("json_path", "").strip(),
                    event_count=event_count,
                    reviewed=reviewed,
                    review_notes=row.get("review_notes", "").strip(),
                )
            )

    logger.info("Loaded %d review records from %s", len(records), manifest_path)
    return records


def load_events_csv(events_path: Path | str) -> list[Event]:
    """Load the canonical events CSV.

    Args:
        events_path: Path to events.csv.

    Returns:
        List of Event objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    from pickup_putdown.common.schemas import Confidence, Event, EventType

    events_path = Path(events_path)
    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")

    required_cols = {"event_id", "clip_id", "type", "t_start", "t_end"}

    events: list[Event] = []
    with open(events_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty events file: {events_path}")

        actual_cols = set(reader.fieldnames)
        missing = required_cols - actual_cols
        if missing:
            raise ValueError(f"Events file missing columns: {sorted(missing)}")

        for row in reader:
            events.append(
                Event(
                    event_id=row["event_id"].strip(),
                    clip_id=row["clip_id"].strip(),
                    type=EventType(row["type"].strip()),
                    t_start=float(row["t_start"]),
                    t_end=float(row["t_end"]),
                    hard_case=row.get("hard_case", "False").strip().lower() in ("true", "1"),
                    annotator=row.get("annotator", "").strip() or None,
                    confidence=Confidence(row.get("confidence", "high").strip().lower())
                    if row.get("confidence", "").strip()
                    else Confidence.HIGH,
                    notes=row.get("notes", "").strip() or None,
                )
            )

    logger.info("Loaded %d events from %s", len(events), events_path)
    return events


def load_clips_csv(clips_path: Path | str) -> dict[str, dict]:
    """Load the clips CSV as a dict keyed by clip_id.

    Args:
        clips_path: Path to clips.csv.

    Returns:
        Dict mapping clip_id -> row dict.
    """
    clips_path = Path(clips_path)
    if not clips_path.exists():
        raise FileNotFoundError(f"Clips file not found: {clips_path}")

    clips: dict[str, dict] = {}
    with open(clips_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_id = row.get("clip_id", "").strip()
            if clip_id:
                clips[clip_id] = row

    logger.info("Loaded %d clips from %s", len(clips), clips_path)
    return clips


def load_candidate_metadata_index(
    candidate_staging_dir: Path | str,
) -> dict[str, CandidateMetadata]:
    """Load candidate metadata JSONs and index by candidate_id.

    Scans <candidate_staging_dir>/candidates/<clip_id>/<clip_id>.json files
    and extracts the candidate list.

    Args:
        candidate_staging_dir: Path to the candidate staging directory.

    Returns:
        Dict mapping candidate_id -> CandidateMetadata.
    """
    candidate_staging_dir = Path(candidate_staging_dir)
    candidates_dir = candidate_staging_dir / "candidates"

    if not candidates_dir.exists():
        logger.warning("Candidate staging directory not found: %s", candidates_dir)
        return {}

    index: dict[str, CandidateMetadata] = {}
    clip_dirs = sorted(candidates_dir.iterdir())

    for clip_dir in clip_dirs:
        if not clip_dir.is_dir():
            continue
        meta_file = clip_dir / f"{clip_dir.name}.json"
        if not meta_file.exists():
            continue

        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load %s: %s", meta_file, e)
            continue

        clip_id = data.get("source_video_id", clip_dir.name)
        for cand in data.get("candidates", []):
            cid = cand.get("candidate_id")
            if not cid:
                continue
            index[cid] = CandidateMetadata(
                candidate_id=cid,
                clip_id=clip_id,
                source_start_s=float(cand.get("source_start_s", 0)),
                source_end_s=float(cand.get("source_end_s", 0)),
                duration_s=float(cand.get("duration_s", 0)),
                actor_id=cand.get("actor_id"),
                hand_side=cand.get("hand_side"),
                region_id=cand.get("region_id"),
            )

    logger.info("Indexed %d candidates from %s", len(index), candidates_dir)
    return index


# ---------------------------------------------------------------------------
# VLM annotation path resolution
# ---------------------------------------------------------------------------


def resolve_video_path(
    candidate_id: str,
    clip_id: str,
    manifest_video_path: str,
    candidate_staging_dir: Path,
    source_video_dir: Path,
) -> Path | None:
    """Resolve the actual video path for a candidate.

    Resolution order:
    1. Local candidate video: <staging>/candidates/<clip>/<cand>.mp4
    2. Source video: <source_dir>/<clip>.mp4
    3. Manifest path (if local)

    Args:
        candidate_id: Candidate identifier.
        clip_id: Clip identifier.
        manifest_video_path: Video path from the review manifest.
        candidate_staging_dir: Path to candidate staging directory.
        source_video_dir: Path to source video directory.

    Returns:
        Resolved Path or None if not found.
    """
    # Try candidate video
    cand_video = candidate_staging_dir / "candidates" / clip_id / f"{candidate_id}.mp4"
    if cand_video.exists():
        return cand_video

    # Try manifest path
    manifest_path = Path(manifest_video_path)
    if manifest_path.exists():
        return manifest_path

    # Try source video (for pose inference)
    src_video = source_video_dir / f"{clip_id}.mp4"
    if src_video.exists():
        return src_video

    return None


def resolve_source_video_path(
    clip_id: str,
    source_video_dir: Path,
) -> Path | None:
    """Resolve source video path for pose inference.

    Args:
        clip_id: Clip identifier.
        source_video_dir: Path to source video directory.

    Returns:
        Resolved Path or None.
    """
    src_video = source_video_dir / f"{clip_id}.mp4"
    if src_video.exists():
        return src_video
    return None


# ---------------------------------------------------------------------------
# Reviewed example resolution
# ---------------------------------------------------------------------------


def _is_zero_event(notes: str, event_count: int) -> bool:
    """Check if review notes indicate zero events."""
    if event_count == 0:
        return True
    lower = notes.lower()
    zero_phrases = [
        "no events",
        "no event",
        "zero events",
        "zero event",
        "not an event",
        "not an action",
        "no pickup",
        "no putdown",
        "no interaction",
    ]
    return any(phrase in lower for phrase in zero_phrases)


def load_reviewed_candidate_jsons(
    review_records: list[ReviewRecord],
) -> dict[str, dict]:
    """Load reviewed candidate JSON files and index by candidate_id.

    Returns only candidates that are marked as reviewed in the manifest.

    Args:
        review_records: Loaded review manifest records.

    Returns:
        Dict mapping candidate_id → JSON contents.
    """
    reviewed_ids = {r.candidate_id for r in review_records if r.reviewed}
    index: dict[str, dict] = {}

    for record in review_records:
        if not record.reviewed:
            continue
        json_path = Path(record.json_path)
        if not json_path.exists():
            logger.warning(
                "JSON not found for candidate %s: %s",
                record.candidate_id,
                json_path,
            )
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            index[record.candidate_id] = data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load JSON for %s: %s", record.candidate_id, e)

    logger.info(
        "Loaded %d reviewed candidate JSON(s) for %d reviewed records",
        len(index),
        len(reviewed_ids),
    )
    return index


def export_reviewed_events(
    review_records: list[ReviewRecord],
    output_path: Path | str,
) -> int:
    """Export reviewed events from candidate JSONs to events.csv.

    Loads each reviewed candidate JSON, converts candidate-relative
    timestamps to source-video timestamps, and writes a compliant
    events.csv with one row per reviewed event.

    Args:
        review_records: Loaded review manifest records.
        output_path: Path for the output events.csv.

    Returns:
        Number of events written.
    """
    reviewed_jsons = load_reviewed_candidate_jsons(review_records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for cid, data in reviewed_jsons.items():
        events = data.get("events", [])
        if not events:
            continue

        source_start = float(data.get("source_start_s", 0))
        clip_id = data.get("clip_id", "")

        for evt in events:
            start_s = float(evt.get("start_s", 0))
            end_s = float(evt.get("end_s", 0))

            # Convert to source-video timestamps
            t_start = source_start + start_s
            t_end = source_start + end_s

            # Generate deterministic event_id
            event_id = f"evt_{cid}_{start_s:.2f}_{end_s:.2f}"

            rows.append(
                {
                    "event_id": event_id,
                    "clip_id": clip_id,
                    "type": evt.get("label", "pickup"),
                    "t_start": t_start,
                    "t_end": t_end,
                    "hard_case": str(evt.get("hard_case", False) is True).capitalize(),
                    "annotator": "reviewed_manifest",
                    "confidence": evt.get("confidence", "high"),
                    "notes": evt.get("notes", ""),
                }
            )

    rows.sort(key=lambda r: (r["clip_id"], r["t_start"]))

    fieldnames = [
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
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Exported %d reviewed events to %s", len(rows), output_path)
    return len(rows)


def load_reviewed_events_csv(events_path: Path | str) -> list[Event]:
    """Load the reviewed events CSV (same schema as canonical events).

    Args:
        events_path: Path to reviewed_events.csv.

    Returns:
        List of Event objects.
    """
    return load_events_csv(events_path)


def resolve_reviewed_examples(
    review_records: list[ReviewRecord],
    events: list[Event],
    candidate_metadata: dict[str, CandidateMetadata],
) -> tuple[list[ReviewedExample], BuildSummary]:
    """Resolve reviewed examples from the review manifest.

    The review manifest is the ground truth. Labels come from the
    reviewed candidate JSON events, not from VLM events.csv.

    For each reviewed task:
    - JSON has events → positive examples with JSON-provided labels
    - JSON has no events → verified negative
    - Unreviewed or missing metadata → excluded

    Args:
        review_records: Loaded review manifest records.
        events: Canonical events from events.csv (for temporal reference only).
        candidate_metadata: Indexed candidate metadata.

    Returns:
        Tuple of (resolved examples, build summary).
    """
    # Index events by clip_id for optional temporal reference
    events_by_clip: dict[str, list[Event]] = {}
    for evt in events:
        events_by_clip.setdefault(evt.clip_id, []).append(evt)

    # Load reviewed candidate JSONs
    reviewed_jsons = load_reviewed_candidate_jsons(review_records)

    examples: list[ReviewedExample] = []
    summary = BuildSummary()

    for record in review_records:
        # Skip unreviewed
        if not record.reviewed:
            summary.excluded_unreviewed += 1
            continue

        summary.total_reviewed += 1

        # Get candidate metadata
        meta = candidate_metadata.get(record.candidate_id)
        if meta is None:
            logger.warning(
                "No metadata for candidate %s (clip %s), skipping",
                record.candidate_id,
                record.clip_id,
            )
            summary.excluded_no_match += 1
            continue

        # Load reviewed JSON
        json_data = reviewed_jsons.get(record.candidate_id)
        if json_data is None:
            logger.warning(
                "No reviewed JSON for candidate %s, skipping",
                record.candidate_id,
            )
            summary.excluded_no_match += 1
            continue

        reviewed_events = json_data.get("events", [])

        # Zero-event → verified negative
        if not reviewed_events:
            examples.append(
                ReviewedExample(
                    candidate_id=record.candidate_id,
                    clip_id=record.clip_id,
                    label="negative",
                    source_start_s=meta.source_start_s,
                    source_end_s=meta.source_end_s,
                    actor_id=meta.actor_id,
                    hand_side=meta.hand_side,
                    region_id=meta.region_id,
                    review_status="reviewed",
                    review_notes=record.review_notes,
                )
            )
            summary.negatives += 1
            continue

        # Positive: use label from reviewed JSON events
        # If multiple events, use the first one's label for the candidate
        best_label = reviewed_events[0].get("label", "pickup")

        # Optional: find matching canonical events for temporal reference
        clip_events = events_by_clip.get(record.clip_id, [])
        matched_events = _match_events_to_candidate(
            clip_events, meta.source_start_s, meta.source_end_s
        )

        event_ids: list[str] = []
        event_labels: list[str] = []
        if matched_events:
            event_ids = [e["event_id"] for e in matched_events]
            event_labels = [e["type"] for e in matched_events]

        examples.append(
            ReviewedExample(
                candidate_id=record.candidate_id,
                clip_id=record.clip_id,
                label=best_label,
                source_start_s=meta.source_start_s,
                source_end_s=meta.source_end_s,
                actor_id=meta.actor_id,
                hand_side=meta.hand_side,
                region_id=meta.region_id,
                matched_event_ids=event_ids,
                matched_event_labels=event_labels,
                review_status="reviewed",
                review_notes=record.review_notes,
            )
        )
        summary.positives += 1

    return examples, summary


def _match_events_to_candidate(
    clip_events: list[Event],
    cand_start: float,
    cand_end: float,
    min_overlap_ratio: float = 0.1,
) -> list[dict]:
    """Find events that overlap with a candidate's time window.

    Uses event-relative overlap ratio: overlap / event_duration.
    A candidate is considered a match if it captures at least
    min_overlap_ratio of the event's duration.

    Returns matching events as dicts with event_id, type, overlap_ratio.
    """
    matched: list[dict] = []

    for evt in clip_events:
        overlap_start = max(cand_start, evt.t_start)
        overlap_end = min(cand_end, evt.t_end)
        overlap = max(0.0, overlap_end - overlap_start)

        # Event-relative ratio: what fraction of the event does the candidate cover?
        event_duration = evt.t_end - evt.t_start
        ratio = overlap / event_duration if event_duration > 0 else 0.0

        if ratio >= min_overlap_ratio:
            matched.append(
                {
                    "event_id": evt.event_id,
                    "type": str(evt.type),
                    "overlap_ratio": ratio,
                    "t_start": evt.t_start,
                    "t_end": evt.t_end,
                }
            )

    matched.sort(key=lambda m: m["overlap_ratio"], reverse=True)
    return matched


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------


def extract_recording_day(clip_id: str) -> str | None:
    """Extract the recording day from a clip ID.

    Clip IDs follow pattern: D{camera}_S{start_time}_E{end_time}_anon
    Start times include the date: 20260520141725 → 20260520

    Args:
        clip_id: Clip identifier.

    Returns:
        Date string (YYYYMMDD) or None.
    """
    match = re.search(r"S(\d{8})", clip_id)
    if match:
        return match.group(1)
    # Fallback: use full clip_id as grouping key
    return clip_id


def assign_splits_by_recording_day(
    clip_ids: Sequence[str],
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, str]:
    """Assign train/val/test splits by recording day.

    Groups clips by recording day, then assigns each day to a split
    using deterministic shuffling. Ensures all clips from the same
    day stay in one split.

    Args:
        clip_ids: List of clip IDs to assign.
        seed: Random seed for deterministic assignment.
        train_ratio: Fraction for training split.
        val_ratio: Fraction for validation split.

    Returns:
        Dict mapping clip_id -> split name.
    """
    # Group by recording day
    day_to_clips: dict[str, list[str]] = {}
    for cid in clip_ids:
        day = extract_recording_day(cid) or cid
        day_to_clips.setdefault(day, []).append(cid)

    # Deterministic shuffle of days
    import random

    rng = random.Random(seed)
    days = sorted(day_to_clips.keys())
    rng.shuffle(days)

    # Assign days to splits proportionally
    n_days = len(days)
    n_train = max(1, round(n_days * train_ratio))
    n_val = max(1, round(n_days * val_ratio))

    splits: dict[str, str] = {}
    for i, day in enumerate(days):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"

        for cid in day_to_clips[day]:
            splits[cid] = split

    return splits


def validate_split_isolation(
    splits: dict[str, str],
    examples: list[ReviewedExample],
) -> bool:
    """Validate that no clip appears in multiple splits.

    Args:
        splits: Clip-to-split mapping.
        examples: Resolved reviewed examples.

    Returns:
        True if valid.

    Raises:
        ValueError: If split leakage detected.
    """
    clips_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for ex in examples:
        split = splits.get(ex.clip_id)
        if split:
            clips_by_split[split].add(ex.clip_id)

    train = clips_by_split["train"]
    val = clips_by_split["val"]
    test = clips_by_split["test"]

    overlaps = []
    if train & val:
        overlaps.append(f"train∩val: {len(train & val)} clips")
    if train & test:
        overlaps.append(f"train∩test: {len(train & test)} clips")
    if val & test:
        overlaps.append(f"val∩test: {len(val & test)} clips")

    if overlaps:
        raise ValueError(f"Split leakage detected: {', '.join(overlaps)}")

    logger.info("Split isolation validated: no clip appears in multiple splits")
    return True


# ---------------------------------------------------------------------------
# Pose inference
# ---------------------------------------------------------------------------


def run_pose_inference_for_clips(
    clip_video_paths: dict[str, Path],
    time_windows: dict[str, list[tuple[float, float]]],
    pose_cfg: PoseConfig,
) -> list[PoseObservation]:
    """Run pose inference on source video windows.

    For each clip, runs the YOLO pose tracker on the time windows
    that contain reviewed candidates.

    Args:
        clip_video_paths: Map of clip_id -> source video path.
        time_windows: Map of clip_id -> list of (start_s, end_s) windows.
        pose_cfg: Pose configuration.

    Returns:
        List of PoseObservation objects.
    """
    from pickup_putdown.common.schemas import ActiveSpan
    from pickup_putdown.perception.pose_tracker import PoseTracker

    all_poses: list[PoseObservation] = []

    for clip_id, video_path in clip_video_paths.items():
        windows = time_windows.get(clip_id, [])
        if not windows:
            continue

        # Merge overlapping windows for efficiency
        merged = _merge_intervals(windows)

        # Create active spans for the windows
        spans = []
        for i, (start, end) in enumerate(merged):
            spans.append(
                ActiveSpan(
                    clip_id=clip_id,
                    active_span_id=f"review_window_{i}",
                    t_start=start,
                    t_end=end,
                )
            )

        try:
            tracker = PoseTracker(
                video_path=video_path,
                pose_cfg=pose_cfg,
                active_spans=spans,
            )
            poses = tracker.run()
            all_poses.extend(poses)
            logger.info(
                "Pose inference for %s: %d observations from %d windows",
                clip_id,
                len(poses),
                len(windows),
            )
        except Exception as e:
            logger.error("Pose inference failed for %s: %s", clip_id, e)

    return all_poses


def _merge_intervals(
    intervals: list[tuple[float, float]],
    gap: float = 1.0,
) -> list[tuple[float, float]]:
    """Merge overlapping or close intervals."""
    if not intervals:
        return []

    sorted_intervals = sorted(intervals)
    merged = [sorted_intervals[0]]

    for start, end in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------


def build_reviewed_feature_dataset(
    review_manifest_path: Path | str,
    events_path: Path | str,
    clips_path: Path | str,
    candidate_staging_dir: Path | str,
    source_video_dir: Path | str,
    output_dir: Path | str,
    pose_cfg: PoseConfig,
    track_a_cfg: TrackAFeaturesConfig,
    shelf_regions: dict[str, Polygon],
    split_seed: int = 42,
    embedder: AbstractImageEmbedder | None = None,
) -> tuple[FeatureDataset, BuildSummary]:
    """Build the reviewed Track A feature dataset.

    Complete pipeline:
    1. Load and validate inputs
    2. Resolve reviewed examples
    3. Assign splits
    4. Run pose inference
    5. Extract features
    6. Save outputs

    Args:
        review_manifest_path: Path to review_manifest.csv.
        events_path: Path to events.csv.
        clips_path: Path to clips.csv.
        candidate_staging_dir: Path to candidate staging directory.
        source_video_dir: Path to source video directory.
        output_dir: Output directory for features and manifest.
        pose_cfg: Pose inference configuration.
        track_a_cfg: Track A features configuration.
        shelf_regions: Map of region_id -> polygon.
        split_seed: Random seed for split assignment.
        embedder: Optional pre-created embedder.

    Returns:
        Tuple of (FeatureDataset, BuildSummary).
    """
    from pickup_putdown.common.schemas import Candidate
    from pickup_putdown.layer1.track_a.dataset_builder import (
        build_feature_dataset,
    )
    from pickup_putdown.layer1.track_a.dataset_builder import (
        validate_split_isolation as validate_ds_split,
    )
    from pickup_putdown.layer1.track_a.manifest import save_manifest

    clips = load_clips_csv(clips_path)  # noqa: F841 - loaded for diagnostics
    output_dir = Path(output_dir)
    candidate_staging_dir = Path(candidate_staging_dir)
    source_video_dir = Path(source_video_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load inputs
    logger.info("Loading inputs...")
    review_records = load_review_manifest(review_manifest_path)
    _events = load_events_csv(events_path)  # VLM events, not ground truth
    _clips = load_clips_csv(clips_path)  # loaded for diagnostics
    candidate_meta = load_candidate_metadata_index(candidate_staging_dir)

    # Step 1b: Regenerate events.csv from reviewed JSONs
    logger.info("Regenerating events.csv from reviewed candidate JSONs...")
    reviewed_events_path = output_dir / "reviewed_events.csv"
    n_exported = export_reviewed_events(review_records, reviewed_events_path)
    logger.info("Exported %d reviewed events to %s", n_exported, reviewed_events_path)

    # Load the regenerated events for downstream use
    events = load_reviewed_events_csv(reviewed_events_path)

    # Step 2: Resolve reviewed examples
    logger.info("Resolving reviewed examples...")
    examples, summary = resolve_reviewed_examples(review_records, events, candidate_meta)

    if summary.errors:
        for err in summary.errors:
            logger.warning("Review resolution issue: %s", err)

    if not examples:
        raise ValueError(
            f"No reviewed examples resolved. "
            f"{summary.total_reviewed} reviewed, "
            f"{summary.excluded_unreviewed} unreviewed, "
            f"{summary.excluded_no_match} no metadata, "
            f"{len(summary.errors)} no matching event."
        )

    logger.info(
        "Resolved %d examples: %d positive, %d negative",
        len(examples),
        summary.positives,
        summary.negatives,
    )

    # Step 3: Assign splits
    logger.info("Assigning splits...")
    clip_ids = list({ex.clip_id for ex in examples})
    splits = assign_splits_by_recording_day(clip_ids, seed=split_seed)
    validate_split_isolation(splits, examples)

    split_counts = {}
    for s in splits.values():
        split_counts[s] = split_counts.get(s, 0) + 1
    logger.info("Split assignment: %s", split_counts)

    # Step 4: Build time windows and resolve video paths
    logger.info("Building time windows for pose inference...")
    clip_windows: dict[str, list[tuple[float, float]]] = {}
    for ex in examples:
        clip_windows.setdefault(ex.clip_id, []).append((ex.source_start_s, ex.source_end_s))

    clip_video_paths: dict[str, Path] = {}
    for clip_id in clip_windows:
        video_path = resolve_source_video_path(clip_id, source_video_dir)
        if video_path is None:
            logger.warning("No source video for clip %s, skipping pose inference", clip_id)
            continue
        clip_video_paths[clip_id] = video_path

    # Step 5: Run pose inference
    logger.info("Running pose inference on %d clips...", len(clip_video_paths))
    pose_observations = run_pose_inference_for_clips(clip_video_paths, clip_windows, pose_cfg)
    logger.info("Total pose observations: %d", len(pose_observations))

    # Step 6: Build Candidate objects
    candidates: list[Candidate] = []
    label_overrides: dict[str, str] = {}

    for ex in examples:
        candidates.append(
            Candidate(
                candidate_id=ex.candidate_id,
                clip_id=ex.clip_id,
                actor_id=ex.actor_id or "",
                hand_side=ex.hand_side,
                region_id=ex.region_id,
                raw_start_s=ex.source_start_s,
                raw_end_s=ex.source_end_s,
                window_start_s=ex.source_start_s,
                window_end_s=ex.source_end_s,
                review_status=ex.review_status,
            )
        )
        label_overrides[ex.candidate_id] = ex.label

    # Step 7: Build video paths map (use source videos for feature extraction)
    video_paths: dict[str, Path] = {}
    for clip_id, path in clip_video_paths.items():
        video_paths[clip_id] = path

    # Step 8: Run feature extraction
    logger.info("Extracting features...")
    dataset = build_feature_dataset(
        candidates=candidates,
        events=events,
        pose_observations=pose_observations,
        shelf_regions=shelf_regions,
        splits=splits,
        video_paths=video_paths,
        config=track_a_cfg,
        embedder=embedder,
        label_overrides=label_overrides,
    )

    if examples and not dataset.records:
        raise RuntimeError(
            f"Feature dataset contains zero records despite {len(examples)} reviewed inputs. "
            "Check candidate-to-pose association, actor_id/hand_side fields, "
            "and timestamp domains."
        )

    # Validate
    validate_ds_split(dataset)

    # Step 9: Compute summary stats
    for record in dataset.records:
        summary.records_by_split[record.split] = summary.records_by_split.get(record.split, 0) + 1
        summary.records_by_label[record.label] = summary.records_by_label.get(record.label, 0) + 1
        summary.records_by_position[record.sample_position] = (
            summary.records_by_position.get(record.sample_position, 0) + 1
        )
        summary.records_by_crop_type[record.crop_type] = (
            summary.records_by_crop_type.get(record.crop_type, 0) + 1
        )

    # Step 10: Save outputs
    logger.info("Saving outputs to %s...", output_dir)

    # Save feature manifest
    manifest_path = output_dir / "feature_dataset.parquet"
    batch_id = f"reviewed_{split_seed}"
    save_manifest(dataset, manifest_path, batch_id=batch_id)

    # Save splits
    splits_path = output_dir / "splits.json"
    splits_data = {
        "seed": split_seed,
        "assignments": splits,
        "clip_counts": split_counts,
    }
    splits_path.write_text(json.dumps(splits_data, indent=2) + "\n")

    # Save build summary
    summary_path = output_dir / "build_summary.json"
    summary_dict = {
        "total_reviewed": summary.total_reviewed,
        "positives": summary.positives,
        "negatives": summary.negatives,
        "excluded_unreviewed": summary.excluded_unreviewed,
        "excluded_no_match": summary.excluded_no_match,
        "errors": summary.errors,
        "records_by_split": summary.records_by_split,
        "records_by_label": summary.records_by_label,
        "records_by_position": summary.records_by_position,
        "records_by_crop_type": summary.records_by_crop_type,
        "total_records": len(dataset.records),
        "manifest_path": str(manifest_path),
    }
    summary_path.write_text(json.dumps(summary_dict, indent=2) + "\n")

    logger.info("Build complete. Summary saved to %s", summary_path)
    return dataset, summary
