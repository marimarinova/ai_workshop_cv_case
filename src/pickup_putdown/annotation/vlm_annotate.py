"""VLM-assisted visual annotation pipeline for candidate videos.

Discovers candidate videos, extracts review frames, inspects them visually,
and produces canonical event annotations aligned with the repository schema.

Designed to be run as:
    pickup-putdown annotate-vlm <candidate-dir> --output-dir .local/vlm_annotations
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from pickup_putdown.annotation.schemas import (
    ConfidenceLevel,
    EventLabel,
)

if TYPE_CHECKING:
    from pickup_putdown.annotation.vlm_client import VlmClientConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schemas for VLM annotation output
# ---------------------------------------------------------------------------


class VlMEventAnnotation(BaseModel):
    """One event annotation produced by VLM visual review."""

    label: EventLabel
    start_s: float = Field(ge=0.0)
    end_s: float = Field(ge=0.0)
    item_count: int = Field(default=1, ge=1)
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH
    hard_case: bool = False
    group_id: str | None = None
    notes: str = ""

    @field_validator("end_s")
    @classmethod
    def end_after_start(cls, v: float, info) -> float:
        start = info.data.get("start_s")
        if start is not None and v <= start:
            raise ValueError("end_s must be greater than start_s")
        return v


class VlMCandidateResult(BaseModel):
    """Normalized result for a single candidate after VLM review."""

    candidate_id: str
    clip_id: str
    video_path: str
    candidate_duration_s: float = Field(ge=0.0)
    source_start_s: float = Field(ge=0.0)
    source_end_s: float = Field(ge=0.0)
    review_status: str = "complete"
    events: list[VlMEventAnnotation] = Field(default_factory=list)
    ignore_intervals: list[dict[str, Any]] = Field(default_factory=list)
    complete_active_span_reviewed: bool = True
    fps: float = 0.0
    notes: str = ""

    # VLM execution metadata. A failed VLM call is not a valid no-event result.
    vlm_status: str = "not_run"
    vlm_error: str = ""
    vlm_attempts: int = Field(default=0, ge=0)
    vlm_finish_reason: str | None = None
    vlm_usage: dict[str, Any] = Field(default_factory=dict)
    vlm_raw_response: str = ""

    @field_validator("source_end_s")
    @classmethod
    def source_end_after_start(cls, v: float, info) -> float:
        start = info.data.get("source_start_s")
        if start is not None and v <= start:
            raise ValueError("source_end_s must be greater than source_start_s")
        return v


class ProcessingRecord(BaseModel):
    """Processing ledger entry for one candidate."""

    candidate_id: str
    video_path: str
    status: str  # success, failure, review_required, skipped
    error: str = ""
    processed_at: str = ""
    frames_extracted: int = 0
    events_found: int = 0
    vlm_status: str = ""
    vlm_attempts: int = 0
    vlm_finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ---------------------------------------------------------------------------
# Candidate discovery
# ---------------------------------------------------------------------------


def discover_candidates(candidates_dir: str | Path) -> list[dict[str, Any]]:
    """Discover candidate videos from metadata JSON files.

    Walks the candidates directory, finds metadata JSON files (one per source
    clip), and extracts candidate records with video paths and source offsets.

    Returns a sorted list of candidate dicts with keys:
        candidate_id, clip_id, source_start_s, source_end_s,
        candidate_video, duration_s, codec, fps (if available)
    """
    candidates_dir = Path(candidates_dir)
    if not candidates_dir.exists():
        raise FileNotFoundError(f"Candidates directory not found: {candidates_dir}")

    all_candidates: list[dict[str, Any]] = []

    json_files = sorted(candidates_dir.rglob("*.json"))
    for json_file in json_files:
        try:
            content = json.loads(json_file.read_text())
        except json.JSONDecodeError as exc:
            logger.warning("Malformed JSON in %s: %s", json_file, exc)
            continue

        if isinstance(content, list):
            # Flat array of candidates
            for item in content:
                if isinstance(item, dict):
                    all_candidates.append(item)
            continue

        if not isinstance(content, dict):
            continue

        # Source-level metadata with nested candidates
        if "candidates" in content:
            source_video_id = content.get("source_video_id", json_file.stem)
            nested = content.get("candidates", [])
            if not isinstance(nested, list):
                continue

            for cand in nested:
                if not isinstance(cand, dict):
                    continue
                enriched = dict(cand)
                enriched.setdefault("clip_id", source_video_id)
                all_candidates.append(enriched)

    # Sort deterministically
    all_candidates.sort(key=lambda c: (c.get("clip_id", ""), c.get("candidate_id", "")))
    return all_candidates


# ---------------------------------------------------------------------------
# Video probing
# ---------------------------------------------------------------------------


def probe_candidate_video(video_path: str | Path) -> dict[str, Any]:
    """Probe a candidate video with ffprobe.

    Returns dict with duration_s, fps, width, height, codec, nb_frames.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    try:
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
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        raise RuntimeError(f"ffprobe failed for {video_path}: {exc}") from exc

    video_stream = None
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream is None:
        raise RuntimeError(f"No video stream found in {video_path}")

    # Parse frame rate
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
        "codec": video_stream.get("codec_name", ""),
        "nb_frames": int(video_stream.get("nb_frames", 0)),
    }


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def extract_review_frames(
    video_path: str | Path,
    output_dir: str | Path,
    target_fps: float = 5.0,
    max_width: int = 640,
) -> list[Path]:
    """Extract review frames from a candidate video.

    Extracts frames at target_fps, resized to max_width for efficient review.
    Frames are saved as frame_0001.jpg, frame_0002.jpg, etc.

    Returns sorted list of extracted frame paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Calculate stride from actual video FPS
    probe = probe_candidate_video(video_path)
    actual_fps = probe["fps"]
    if actual_fps <= 0:
        actual_fps = 30.0

    stride = max(1, round(actual_fps / target_fps))

    # Extract frames
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"select='not(mod(n,{stride}))',scale={max_width}:-1",
            "-vsync",
            "vfr",
            str(output_dir / "frame_%04d.jpg"),
        ],
        capture_output=True,
        timeout=60,
    )

    frames = sorted(output_dir.glob("frame_*.jpg"))
    return frames


def create_contact_sheet(
    frame_paths: list[Path],
    output_path: str | Path,
    cols: int = 8,
    frame_width: int = 320,
) -> Path:
    """Create a single contact sheet image from extracted frames.

    Arranges frames in a grid with timestamps overlay.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        logger.warning("Pillow not available, skipping contact sheet creation")
        return Path(output_path)

    if not frame_paths:
        return Path(output_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = []
    for frame_idx, fp in enumerate(frame_paths):
        try:
            img = Image.open(fp)
            if img.width > frame_width:
                ratio = frame_width / img.width
                new_height = max(1, int(img.height * ratio))
                img = img.resize((frame_width, new_height), Image.LANCZOS)
            # Add the zero-based review-frame index used by the VLM schema.
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            ts = f"#{frame_idx}"
            draw.text((5, 5), ts, fill=(255, 255, 0), font=font)
            images.append(img)
        except Exception:
            continue

    if not images:
        return output_path

    # Calculate grid
    rows = (len(images) + cols - 1) // cols
    max_h = max(img.height for img in images)
    max_w = max(img.width for img in images)

    sheet = Image.new("RGB", (cols * max_w, rows * max_h), (240, 240, 240))
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        sheet.paste(img, (col * max_w, row * max_h))

    sheet.save(str(output_path))
    return output_path


# ---------------------------------------------------------------------------
# Frame-to-timestamp conversion
# ---------------------------------------------------------------------------


def frame_index_to_time(frame_idx: int, fps: float, source_start_s: float) -> tuple[float, float]:
    """Convert frame index to (candidate_relative_s, source_absolute_s)."""
    if fps <= 0:
        fps = 30.0
    candidate_rel_s = frame_idx / fps
    source_abs_s = source_start_s + candidate_rel_s
    return candidate_rel_s, source_abs_s


# ---------------------------------------------------------------------------
# VLM annotation analysis (visual inspection)
# ---------------------------------------------------------------------------


def analyze_candidate_frames(
    frame_paths: list[Path],
    fps: float,
    candidate_duration_s: float,
    candidate_id: str,
    contact_sheet_path: Path,
    proposal_info: dict[str, Any] | None = None,
    vlm_config: VlmClientConfig | None = None,
) -> VlMCandidateResult:
    """Analyze extracted frames with the VLM.

    Expected inference failures are represented explicitly through
    ``review_status="failed"`` and VLM metadata. They are never converted into
    a successful empty-event annotation.
    """
    del proposal_info  # Reserved for future prompt enrichment.

    from pickup_putdown.annotation.vlm_client import (
        VlmClientError,
        call_vlm,
        vlm_result_to_annotations,
    )

    result = VlMCandidateResult(
        candidate_id=candidate_id,
        clip_id="",
        video_path="",
        candidate_duration_s=candidate_duration_s,
        source_start_s=0.0,
        source_end_s=candidate_duration_s,
        review_status="pending_review",
        complete_active_span_reviewed=False,
        fps=fps,
    )

    if vlm_config is None:
        result.vlm_status = "disabled"
        result.notes = "VLM analysis disabled; manual review required."
        logger.info("VLM disabled for %s; manual review required", candidate_id)
        return result

    if not frame_paths:
        result.review_status = "failed"
        result.vlm_status = "failed"
        result.vlm_error = "No review frames were extracted"
        logger.error("VLM analysis failed for %s: %s", candidate_id, result.vlm_error)
        return result

    if not contact_sheet_path.is_file():
        result.review_status = "failed"
        result.vlm_status = "failed"
        result.vlm_error = f"Contact sheet not found: {contact_sheet_path}"
        logger.error("VLM analysis failed for %s: %s", candidate_id, result.vlm_error)
        return result

    vlm_response = call_vlm(
        contact_sheet_path=contact_sheet_path,
        frame_count=len(frame_paths),
        fps=fps,
        duration_s=candidate_duration_s,
        config=vlm_config,
    )

    result.vlm_status = str(vlm_response.get("status", "failed"))
    result.vlm_error = str(vlm_response.get("error") or "")
    result.vlm_attempts = int(vlm_response.get("attempts", 0) or 0)
    result.vlm_finish_reason = vlm_response.get("finish_reason")
    result.vlm_usage = dict(vlm_response.get("usage") or {})
    result.vlm_raw_response = str(vlm_response.get("raw_response") or "")

    if result.vlm_status != "success":
        result.review_status = "failed"
        result.complete_active_span_reviewed = False
        logger.error(
            "VLM annotation failed for %s after %d attempt(s): %s",
            candidate_id,
            result.vlm_attempts,
            result.vlm_error or "unknown VLM error",
        )
        return result

    reasoning = str(vlm_response.get("reasoning") or "").strip()

    try:
        annotations = vlm_result_to_annotations(
            vlm_response,
            fps,
            duration_s=candidate_duration_s,
        )
        events = [VlMEventAnnotation(**annotation) for annotation in annotations]
    except (VlmClientError, ValidationError, TypeError, ValueError) as exc:
        result.review_status = "failed"
        result.complete_active_span_reviewed = False
        result.vlm_status = "failed"
        result.vlm_error = f"Failed to convert VLM response: {exc}"
        logger.exception("Failed to convert VLM response for %s", candidate_id)
        return result

    result.review_status = "complete"
    result.complete_active_span_reviewed = True
    result.events = events
    result.notes = reasoning[:2_000]

    if reasoning:
        # Do not slice this log message: the previous implementation made valid
        # reasoning appear truncated even when the JSON response was complete.
        logger.info("VLM reasoning for %s: %s", candidate_id, reasoning)

    return result


# ---------------------------------------------------------------------------
# Annotation normalization
# ---------------------------------------------------------------------------


def normalize_candidate_result(
    result: VlMCandidateResult,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize VLM candidate result into canonical event dicts and ignore intervals.

    Returns (events, ignore_intervals) as lists of dicts compatible with
    the canonical CSV schema.
    """
    events: list[dict[str, Any]] = []
    ignores: list[dict[str, Any]] = []

    for evt in result.events:
        # Convert candidate-relative timestamps to source-video timestamps
        source_event_start = result.source_start_s + evt.start_s
        source_event_end = result.source_start_s + evt.end_s

        # Generate deterministic event group ID
        group_raw = f"{result.clip_id}:{evt.label}:{evt.start_s:.2f}"
        group_id = f"group_{hashlib.sha256(group_raw.encode()).hexdigest()[:12]}"

        # Handle multi-item: emit separate rows
        for item_idx in range(evt.item_count):
            event_raw = f"{result.clip_id}:{evt.label}:{group_id}:{item_idx}"
            event_id = f"evt_{hashlib.md5(event_raw.encode()).hexdigest()[:12]}"

            events.append(
                {
                    "event_id": event_id,
                    "clip_id": result.clip_id,
                    "type": str(evt.label),
                    "t_start": round(source_event_start, 3),
                    "t_end": round(source_event_end, 3),
                    "hard_case": evt.hard_case,
                    "annotator": "vlm_pipeline",
                    "confidence": str(evt.confidence),
                    "notes": evt.notes,
                }
            )

    # Process ignore intervals
    for ig in result.ignore_intervals:
        ig_start = result.source_start_s + float(ig.get("start_s", 0.0))
        ig_end = result.source_start_s + float(ig.get("end_s", 0.0))
        ig_raw = f"{result.clip_id}:ignore:{ig_start:.2f}"
        ig_id = f"ign_{hashlib.md5(ig_raw.encode()).hexdigest()[:12]}"

        ignores.append(
            {
                "ignore_id": ig_id,
                "clip_id": result.clip_id,
                "t_start": round(ig_start, 3),
                "t_end": round(ig_end, 3),
                "reason": ig.get("reason", "UNLABELABLE"),
                "annotator": "vlm_pipeline",
                "notes": ig.get("notes", ""),
            }
        )

    return events, ignores


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_candidate_annotation(
    result: VlMCandidateResult,
) -> list[str]:
    """Validate a candidate annotation result.

    Returns list of error messages (empty if valid).
    """
    errors: list[str] = []

    # Check timestamps against duration
    for evt in result.events:
        if evt.start_s < 0:
            errors.append(f"Event {evt.label} has negative start_s={evt.start_s}")
        if evt.end_s > result.candidate_duration_s + 0.1:
            errors.append(
                f"Event {evt.label} end_s={evt.end_s} exceeds "
                f"candidate duration {result.candidate_duration_s}"
            )
        if evt.end_s <= evt.start_s:
            errors.append(f"Event {evt.label} has end_s={evt.end_s} <= start_s={evt.start_s}")

    # Check source timestamps
    for evt in result.events:
        source_start = result.source_start_s + evt.start_s
        source_end = result.source_start_s + evt.end_s
        if source_start >= source_end:
            errors.append(
                f"Event {evt.label} source timestamps invalid: {source_start} >= {source_end}"
            )

    # Check label values
    valid_labels = {EventLabel.PICKUP, EventLabel.PUTDOWN}
    for evt in result.events:
        if evt.label not in valid_labels:
            errors.append(f"Invalid event label: {evt.label}")

    # Check confidence values
    valid_confidence = {ConfidenceLevel.HIGH, ConfidenceLevel.MED, ConfidenceLevel.LOW}
    for evt in result.events:
        if evt.confidence not in valid_confidence:
            errors.append(f"Invalid confidence: {evt.confidence}")

    return errors


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Configuration for the VLM annotation pipeline."""

    candidates_dir: str
    output_dir: str
    review_fps: float = 5.0
    max_frame_width: int = 640
    contact_sheet_cols: int = 8
    force: bool = False
    limit: int | None = None
    annotator: str = "vlm_pipeline"
    # VLM client settings
    vlm_base_url: str = "http://localhost:8080"
    vlm_model: str = ""
    vlm_temperature: float = 0.0
    vlm_max_tokens: int = 2048
    vlm_retry_max_tokens: int = 4096
    vlm_max_attempts: int = 2
    vlm_retry_delay_s: float = 1.0
    vlm_timeout_s: int = 180
    vlm_disable_thinking: bool = True
    vlm_enforce_json_schema: bool = True
    vlm_enabled: bool = True


@dataclass
class PipelineSummary:
    """Summary of pipeline execution."""

    total_candidates: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    review_required: int = 0
    events_found: int = 0
    processing_time_s: float = 0.0
    errors: list[str] = field(default_factory=list)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON object atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _effective_review_fps(actual_fps: float, target_fps: float) -> float:
    """Return the effective FPS produced by the extraction stride."""

    if actual_fps <= 0:
        return target_fps if target_fps > 0 else 5.0
    if target_fps <= 0:
        return actual_fps

    stride = max(1, round(actual_fps / target_fps))
    return actual_fps / stride


def _update_processing_from_result(
    record: ProcessingRecord,
    result: VlMCandidateResult,
) -> None:
    """Copy VLM execution metadata into the processing ledger."""

    usage = result.vlm_usage
    record.vlm_status = result.vlm_status
    record.vlm_attempts = result.vlm_attempts
    record.vlm_finish_reason = result.vlm_finish_reason or ""
    record.prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    record.completion_tokens = int(usage.get("completion_tokens", 0) or 0)


def run_pipeline(config: PipelineConfig) -> PipelineSummary:
    """Run the VLM annotation pipeline.

    Only successfully inferred and validated candidates are normalized into
    canonical outputs. Failed VLM calls are persisted in ``raw/`` and recorded
    in ``processing.csv`` without being converted into zero-event negatives.
    """
    import time

    start_time = time.time()
    summary = PipelineSummary()

    output_base = Path(config.output_dir)
    raw_dir = output_base / "raw"
    normalized_dir = output_base / "normalized"
    frames_dir = output_base / "review_frames"
    for directory in (raw_dir, normalized_dir, frames_dir):
        directory.mkdir(parents=True, exist_ok=True)

    candidates = discover_candidates(config.candidates_dir)
    if config.limit is not None:
        candidates = candidates[: config.limit]

    summary.total_candidates = len(candidates)
    logger.info("Discovered %d candidates", len(candidates))

    all_events: list[dict[str, Any]] = []
    all_processing: list[ProcessingRecord] = []

    for idx, candidate in enumerate(candidates):
        candidate_id = str(candidate.get("candidate_id", f"unknown_{idx}"))
        candidate_key = str(candidate.get("candidate_key", ""))
        video_path = Path(candidate_key)
        total_candidates = len(candidates)

        if (idx + 1) % 10 == 0 or idx == 0:
            percentage = ((idx + 1) / total_candidates * 100) if total_candidates else 100.0
            print(
                f"Progress: {idx + 1}/{total_candidates} ({percentage:.1f}%)",
                file=sys.stderr,
                flush=True,
            )

        norm_path = normalized_dir / f"{candidate_id}.json"
        raw_path = raw_dir / f"{candidate_id}.json"

        if norm_path.exists() and not config.force:
            existing_data = json.loads(norm_path.read_text(encoding="utf-8"))
            existing_result = VlMCandidateResult.model_validate(existing_data)
        
            # Ignore intervals in normalized files are already canonical.
            existing_result.ignore_intervals = []
        
            existing_events, _ = normalize_candidate_result(existing_result)
            all_events.extend(existing_events)
        
            summary.skipped += 1
            all_processing.append(
                ProcessingRecord(
                    candidate_id=candidate_id,
                    video_path=str(video_path),
                    status="skipped",
                    processed_at=datetime.now(UTC).isoformat(),
                    events_found=len(existing_result.events),
                    vlm_status=existing_result.vlm_status,
                    vlm_attempts=existing_result.vlm_attempts,
                    vlm_finish_reason=existing_result.vlm_finish_reason or "",
                )
            )
            continue

        if config.force:
            # A failed forced re-run must not leave an older successful result
            # that could later be mistaken for the current output.
            norm_path.unlink(missing_ok=True)

        record = ProcessingRecord(
            candidate_id=candidate_id,
            video_path=str(video_path),
            status="pending",
            processed_at=datetime.now(UTC).isoformat(),
        )

        try:
            probe_info = probe_candidate_video(video_path)
            source_fps = float(probe_info["fps"])
            duration_s = float(probe_info["duration_s"])
            review_fps = _effective_review_fps(source_fps, config.review_fps)

            candidate_frames_dir = frames_dir / candidate_id
            existing_frames = sorted(candidate_frames_dir.glob("frame_*.jpg"))
            if existing_frames and not config.force:
                frame_paths = existing_frames
                logger.info(
                    "Reusing %d existing frames for %s",
                    len(frame_paths),
                    candidate_id,
                )
            else:
                frame_paths = extract_review_frames(
                    video_path=video_path,
                    output_dir=candidate_frames_dir,
                    target_fps=config.review_fps,
                    max_width=config.max_frame_width,
                )

            record.frames_extracted = len(frame_paths)
            if not frame_paths:
                raise RuntimeError("No review frames were extracted")

            contact_sheet_path = candidate_frames_dir / "contact_sheet.jpg"
            if not contact_sheet_path.exists() or config.force:
                create_contact_sheet(
                    frame_paths=frame_paths,
                    output_path=contact_sheet_path,
                    cols=config.contact_sheet_cols,
                )

            if not contact_sheet_path.is_file():
                raise RuntimeError(
                    f"Contact sheet was not created: {contact_sheet_path}"
                )

            vlm_config: VlmClientConfig | None = None
            if config.vlm_enabled:
                from pickup_putdown.annotation.vlm_client import VlmClientConfig

                vlm_config = VlmClientConfig(
                    base_url=config.vlm_base_url,
                    model=config.vlm_model,
                    temperature=config.vlm_temperature,
                    max_tokens=config.vlm_max_tokens,
                    retry_max_tokens=config.vlm_retry_max_tokens,
                    max_attempts=config.vlm_max_attempts,
                    retry_delay_s=config.vlm_retry_delay_s,
                    timeout_s=config.vlm_timeout_s,
                    disable_thinking=config.vlm_disable_thinking,
                    enforce_json_schema=config.vlm_enforce_json_schema,
                )

            analysis_result = analyze_candidate_frames(
                frame_paths=frame_paths,
                fps=review_fps,
                candidate_duration_s=duration_s,
                candidate_id=candidate_id,
                contact_sheet_path=contact_sheet_path,
                vlm_config=vlm_config,
            )

            source_start_s = float(candidate.get("source_start_s", 0.0))
            source_end_s = float(
                candidate.get("source_end_s", source_start_s + duration_s)
            )
            if source_end_s <= source_start_s:
                source_end_s = source_start_s + duration_s

            result = analysis_result.model_copy(
                update={
                    "clip_id": str(candidate.get("clip_id", "")),
                    "video_path": str(video_path),
                    "source_start_s": source_start_s,
                    "source_end_s": source_end_s,
                    "fps": review_fps,
                }
            )
            _update_processing_from_result(record, result)

            raw_data = result.model_dump()
            raw_data.update(
                {
                    "frame_count": len(frame_paths),
                    "contact_sheet": str(contact_sheet_path),
                    "probe_info": probe_info,
                    "source_video_fps": source_fps,
                    "effective_review_fps": review_fps,
                    "metadata": candidate,
                }
            )
            _write_json(raw_path, raw_data)

            if result.review_status == "failed":
                record.status = "failure"
                record.error = result.vlm_error or "VLM annotation failed"
                summary.failed += 1
                summary.errors.append(f"{candidate_id}: {record.error}")
                print(
                    f"[{idx + 1}/{total_candidates}] FAILED {candidate_id}: "
                    f"{record.error}",
                    file=sys.stderr,
                    flush=True,
                )

            elif result.review_status != "complete":
                record.status = "review_required"
                record.error = result.notes
                summary.review_required += 1
                print(
                    f"[{idx + 1}/{total_candidates}] REVIEW {candidate_id} "
                    f"({record.frames_extracted} frames)",
                    file=sys.stderr,
                    flush=True,
                )

            else:
                validation_errors = validate_candidate_annotation(result)
                if validation_errors:
                    result.review_status = "failed"
                    result.complete_active_span_reviewed = False
                    result.vlm_status = "failed"
                    result.vlm_error = "; ".join(validation_errors)
                    record.status = "failure"
                    record.error = result.vlm_error
                    summary.failed += 1
                    summary.errors.append(f"{candidate_id}: {record.error}")

                    raw_data.update(result.model_dump())
                    _write_json(raw_path, raw_data)

                    logger.warning(
                        "Validation errors for %s: %s",
                        candidate_id,
                        validation_errors,
                    )
                    print(
                        f"[{idx + 1}/{total_candidates}] FAILED {candidate_id}: "
                        f"{record.error}",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    events, ignores = normalize_candidate_result(result)
                    all_events.extend(events)

                    normalized_data = result.model_dump(
                        exclude={"vlm_raw_response"}
                    )
                    normalized_data["ignore_intervals"] = ignores
                    _write_json(norm_path, normalized_data)

                    record.status = "success"
                    record.events_found = len(result.events)
                    summary.processed += 1
                    print(
                        f"[{idx + 1}/{total_candidates}] OK {candidate_id} "
                        f"({record.frames_extracted} frames, "
                        f"{record.events_found} events, "
                        f"{record.vlm_attempts} attempt(s))",
                        file=sys.stderr,
                        flush=True,
                    )

        except FileNotFoundError as exc:
            record.status = "failure"
            record.error = f"File not found: {exc}"
            summary.failed += 1
            summary.errors.append(f"{candidate_id}: {record.error}")
            logger.warning("Failed %s: %s", candidate_id, exc)
            print(
                f"[{idx + 1}/{total_candidates}] FAILED {candidate_id}: "
                f"{record.error}",
                file=sys.stderr,
                flush=True,
            )

        except Exception as exc:
            record.status = "failure"
            record.error = str(exc)
            summary.failed += 1
            summary.errors.append(f"{candidate_id}: {record.error}")
            logger.exception("Failed %s: %s", candidate_id, exc)
            print(
                f"[{idx + 1}/{total_candidates}] FAILED {candidate_id}: "
                f"{record.error}",
                file=sys.stderr,
                flush=True,
            )

        all_processing.append(record)

    all_events.sort(
        key=lambda event: (
            event["clip_id"],
            event["t_start"],
            event["type"],
            event["event_id"],
        )
    )

    events_csv = output_base / "events.csv"
    event_columns = [
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
    with events_csv.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=event_columns,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(all_events)

    processing_csv = output_base / "processing.csv"
    processing_columns = [
        "candidate_id",
        "video_path",
        "status",
        "error",
        "processed_at",
        "frames_extracted",
        "events_found",
        "vlm_status",
        "vlm_attempts",
        "vlm_finish_reason",
        "prompt_tokens",
        "completion_tokens",
    ]
    with processing_csv.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=processing_columns)
        writer.writeheader()
        for record in all_processing:
            writer.writerow(record.model_dump())

    summary.processing_time_s = round(time.time() - start_time, 2)
    summary.events_found = len(all_events)
    _write_json(
        output_base / "summary.json",
        {
            "total_candidates": summary.total_candidates,
            "processed": summary.processed,
            "skipped": summary.skipped,
            "failed": summary.failed,
            "review_required": summary.review_required,
            "events_found": summary.events_found,
            "processing_time_s": summary.processing_time_s,
            "errors": summary.errors,
            "annotator": config.annotator,
            "review_fps_target": config.review_fps,
            "force": config.force,
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )

    logger.info(
        "Pipeline complete: %d successful, %d skipped, %d failed, "
        "%d review-required, %d event rows",
        summary.processed,
        summary.skipped,
        summary.failed,
        summary.review_required,
        summary.events_found,
    )
    return summary
