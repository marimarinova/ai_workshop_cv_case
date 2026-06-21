"""Preview renderer for triage QA with overlay drawing and video encoding."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pickup_putdown.common.schemas import ActiveSpan, PersonObservation

logger = logging.getLogger(__name__)


@dataclass
class OverlayConfig:
    """Configuration for overlay drawing on preview frames."""

    draw_boxes: bool = True
    draw_track_ids: bool = True
    draw_timestamps: bool = True
    draw_confidence: bool = True
    draw_span_status: bool = True

    span_context_before_s: float = 2.0
    span_context_after_s: float = 2.0

    # Preview frames are sampled from the source video. This avoids implying
    # full-frame-rate tracking when detections were produced at a low rate.
    preview_fps: float = 1.0
    no_person_preview_duration_s: float = 10.0

    # Downscale large source videos before encoding the preview.
    max_output_width: int = 1280
    max_output_height: int = 720

    box_color: tuple[int, int, int] = (0, 255, 0)  # BGR green
    unstable_box_color: tuple[int, int, int] = (0, 0, 255)  # BGR red
    text_color: tuple[int, int, int] = (255, 255, 255)  # BGR white
    status_background_color: tuple[int, int, int] = (0, 0, 0)

    text_scale: float = 0.5
    line_thickness: int = 1


def draw_overlay(
    frame: np.ndarray,
    observations: list[PersonObservation],
    spans: list[ActiveSpan],
    config: OverlayConfig,
    *,
    frame_timestamp_s: float | None = None,
    source_frame_index: int | None = None,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> np.ndarray:
    """Draw observations belonging to one source frame.

    The caller must pass only observations associated with the source frame
    being rendered. The function does not filter observations by timestamp or
    frame index itself.

    Parameters
    ----------
    frame:
        BGR frame as read by OpenCV, optionally resized.
    observations:
        Observations belonging to the current source frame only.
    spans:
        Active spans for the clip.
    config:
        Overlay rendering configuration.
    frame_timestamp_s:
        Timestamp of the source frame currently being rendered.
    source_frame_index:
        Original source-video frame index.
    scale_x:
        Horizontal scale applied to bounding-box coordinates.
    scale_y:
        Vertical scale applied to bounding-box coordinates.

    Returns
    -------
    np.ndarray
        A copy of the frame containing the requested overlays.
    """
    import cv2

    out = frame.copy()
    height, width = out.shape[:2]

    span_ranges = [(span.t_start, span.t_end) for span in spans]

    if config.draw_boxes:
        for observation in observations:
            x1 = int(round(observation.bbox_x1 * scale_x))
            y1 = int(round(observation.bbox_y1 * scale_y))
            x2 = int(round(observation.bbox_x2 * scale_x))
            y2 = int(round(observation.bbox_y2 * scale_y))

            x1 = min(max(0, x1), max(0, width - 1))
            y1 = min(max(0, y1), max(0, height - 1))
            x2 = min(max(0, x2), max(0, width - 1))
            y2 = min(max(0, y2), max(0, height - 1))

            if x2 <= x1 or y2 <= y1:
                logger.debug(
                    "Skipping invalid preview box for frame %s: (%d, %d, %d, %d)",
                    source_frame_index,
                    x1,
                    y1,
                    x2,
                    y2,
                )
                continue

            color = config.box_color if observation.is_stable else config.unstable_box_color

            cv2.rectangle(
                out,
                (x1, y1),
                (x2, y2),
                color,
                config.line_thickness,
            )

            label_parts: list[str] = []

            if config.draw_track_ids and observation.tracker_track_id is not None:
                label_parts.append(f"ID:{observation.tracker_track_id}")

            if config.draw_confidence:
                label_parts.append(f"{observation.confidence:.2f}")

            if config.draw_timestamps:
                label_parts.append(f"{observation.timestamp_s:.1f}s")

            if config.draw_span_status:
                is_active = _is_in_span(
                    observation.timestamp_s,
                    span_ranges,
                )
                label_parts.append("SPAN" if is_active else "idle")

            if label_parts:
                _draw_label(
                    out,
                    " ".join(label_parts),
                    x=x1,
                    y=y1,
                    background_color=color,
                    text_color=config.text_color,
                    text_scale=config.text_scale,
                    line_thickness=config.line_thickness,
                )

    frame_status_parts: list[str] = []

    # Draw frame-level status only when the caller provides frame metadata.
    # This preserves the pure overlay behavior expected by callers that only
    # request bounding-box drawing.
    if frame_timestamp_s is not None or source_frame_index is not None:
        if config.draw_timestamps and frame_timestamp_s is not None:
            frame_status_parts.append(f"t={frame_timestamp_s:.2f}s")

        if source_frame_index is not None:
            frame_status_parts.append(f"frame={source_frame_index}")

        if config.draw_span_status and frame_timestamp_s is not None:
            is_active = _is_in_span(frame_timestamp_s, span_ranges)
            frame_status_parts.append("ACTIVE" if is_active else "context")

        frame_status_parts.append(f"detections={len(observations)}")

    if frame_status_parts:
        _draw_label(
            out,
            " | ".join(frame_status_parts),
            x=8,
            y=24,
            background_color=config.status_background_color,
            text_color=config.text_color,
            text_scale=config.text_scale,
            line_thickness=config.line_thickness,
            label_below_anchor=True,
        )

    return out


def _draw_label(
    frame: np.ndarray,
    label: str,
    *,
    x: int,
    y: int,
    background_color: tuple[int, int, int],
    text_color: tuple[int, int, int],
    text_scale: float,
    line_thickness: int,
    label_below_anchor: bool = False,
) -> None:
    """Draw a clipped text label with a filled background."""
    import cv2

    height, width = frame.shape[:2]

    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        line_thickness,
    )

    padding = 4
    box_width = text_width + 2 * padding
    box_height = text_height + baseline + 2 * padding

    box_x1 = min(max(0, x), max(0, width - 1))
    box_x2 = min(width, box_x1 + box_width)

    if label_below_anchor:
        box_y1 = min(max(0, y - text_height - padding), max(0, height - box_height))
    else:
        box_y1 = max(0, y - box_height)

    box_y2 = min(height, box_y1 + box_height)

    cv2.rectangle(
        frame,
        (box_x1, box_y1),
        (box_x2, box_y2),
        background_color,
        -1,
    )

    text_x = min(width - 1, box_x1 + padding)
    text_y = min(
        height - 1,
        box_y1 + padding + text_height,
    )

    cv2.putText(
        frame,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        text_color,
        line_thickness,
        cv2.LINE_AA,
    )


def _is_in_span(
    timestamp_s: float,
    span_ranges: list[tuple[float, float]],
) -> bool:
    """Return whether a timestamp falls within any active span."""
    return any(start <= timestamp_s <= end for start, end in span_ranges)


def _merge_frame_ranges(
    ranges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Merge overlapping or directly adjacent half-open frame ranges."""
    if not ranges:
        return []

    ordered = sorted(ranges)
    merged = [ordered[0]]

    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]

        if start <= previous_end:
            merged[-1] = (
                previous_start,
                max(previous_end, end),
            )
        else:
            merged.append((start, end))

    return merged


def _frame_in_ranges(
    frame_index: int,
    ranges: list[tuple[int, int]],
) -> bool:
    """Return whether a source frame belongs to any half-open range."""
    return any(start <= frame_index < end for start, end in ranges)


def _compute_output_size(
    source_width: int,
    source_height: int,
    config: OverlayConfig,
) -> tuple[int, int]:
    """Calculate a bounded output size while preserving aspect ratio."""
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"Invalid source dimensions: {source_width}x{source_height}")

    width_scale = config.max_output_width / source_width
    height_scale = config.max_output_height / source_height
    scale = min(1.0, width_scale, height_scale)

    output_width = max(2, int(round(source_width * scale)))
    output_height = max(2, int(round(source_height * scale)))

    # Some codecs require even dimensions.
    if output_width % 2:
        output_width -= 1
    if output_height % 2:
        output_height -= 1

    return output_width, output_height


def _build_preview_frame_indices(
    *,
    ranges: list[tuple[int, int]],
    observations: list[PersonObservation],
    source_fps: float,
    preview_fps: float,
    total_frames: int,
) -> list[int]:
    """Build deterministic source-frame indices for the QA preview.

    Regularly sampled context frames are included, and every exact observation
    frame inside a selected range is included. The union is sorted and
    deduplicated.
    """
    if preview_fps <= 0:
        raise ValueError("preview_fps must be greater than zero")

    effective_preview_fps = min(preview_fps, source_fps)
    sample_step = max(1, int(round(source_fps / effective_preview_fps)))

    selected: set[int] = set()

    for start, end in ranges:
        bounded_start = min(max(0, start), total_frames)
        bounded_end = min(max(bounded_start, end), total_frames)

        selected.update(range(bounded_start, bounded_end, sample_step))

        if bounded_end > bounded_start:
            selected.add(bounded_end - 1)

    for observation in observations:
        frame_index = int(observation.source_frame_index)

        if 0 <= frame_index < total_frames and _frame_in_ranges(frame_index, ranges):
            selected.add(frame_index)

    return sorted(selected)


def render_triage_preview(
    video_path: Path,
    observations: list[PersonObservation],
    spans: list[ActiveSpan],
    output_path: Path,
    config: OverlayConfig | None = None,
) -> Path:
    """Render a sampled QA preview with frame-aligned track overlays.

    Person-positive previews contain all active spans with context. No-person
    previews contain a regular sample from the beginning of the clip.

    Only observations whose ``source_frame_index`` matches the currently
    rendered source frame are drawn. Historical observations are never carried
    forward to later frames.
    """
    import cv2

    if config is None:
        config = OverlayConfig()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    writer = None

    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if source_fps <= 0:
            raise RuntimeError(f"Invalid source FPS for preview video: {video_path}")

        if total_frames <= 0:
            raise RuntimeError(f"Source video contains no readable frames: {video_path}")

        output_width, output_height = _compute_output_size(
            source_width,
            source_height,
            config,
        )

        scale_x = output_width / source_width
        scale_y = output_height / source_height

        if spans:
            frame_ranges: list[tuple[int, int]] = []

            for span in spans:
                start_s = max(
                    0.0,
                    span.t_start - config.span_context_before_s,
                )
                end_s = max(
                    start_s,
                    span.t_end + config.span_context_after_s,
                )

                start_frame = max(
                    0,
                    int(math.floor(start_s * source_fps)),
                )
                end_frame = min(
                    total_frames,
                    int(math.ceil(end_s * source_fps)) + 1,
                )

                if end_frame > start_frame:
                    frame_ranges.append((start_frame, end_frame))

            merged_ranges = _merge_frame_ranges(frame_ranges)
        else:
            preview_duration_s = max(
                0.0,
                config.no_person_preview_duration_s,
            )
            render_end = min(
                total_frames,
                max(
                    1,
                    int(math.ceil(preview_duration_s * source_fps)),
                ),
            )
            merged_ranges = [(0, render_end)]

        if not merged_ranges:
            raise RuntimeError(f"No valid preview ranges could be derived for {video_path}")

        observations_by_frame: dict[int, list[PersonObservation]] = defaultdict(list)

        for observation in observations:
            observations_by_frame[int(observation.source_frame_index)].append(observation)

        for frame_observations in observations_by_frame.values():
            frame_observations.sort(
                key=lambda observation: (
                    observation.tracker_track_id
                    if observation.tracker_track_id is not None
                    else -1,
                    observation.bbox_x1,
                    observation.bbox_y1,
                )
            )

        preview_frame_indices = _build_preview_frame_indices(
            ranges=merged_ranges,
            observations=observations,
            source_fps=source_fps,
            preview_fps=config.preview_fps,
            total_frames=total_frames,
        )

        if not preview_frame_indices:
            raise RuntimeError(f"No frames selected for preview: {video_path}")

        output_fps = min(config.preview_fps, source_fps)
        if output_fps <= 0:
            raise ValueError("Preview output FPS must be greater than zero")

        writer = _open_video_writer(
            output_path=output_path,
            output_fps=output_fps,
            output_size=(output_width, output_height),
        )

        rendered_frames = 0

        for frame_index in preview_frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()

            if not success:
                logger.warning(
                    "Failed to read preview frame %d from %s",
                    frame_index,
                    video_path,
                )
                continue

            if frame.shape[1] != output_width or frame.shape[0] != output_height:
                frame = cv2.resize(
                    frame,
                    (output_width, output_height),
                    interpolation=cv2.INTER_AREA,
                )

            frame_timestamp_s = frame_index / source_fps
            current_observations = observations_by_frame.get(
                frame_index,
                [],
            )

            overlayed = draw_overlay(
                frame,
                current_observations,
                spans,
                config,
                frame_timestamp_s=frame_timestamp_s,
                source_frame_index=frame_index,
                scale_x=scale_x,
                scale_y=scale_y,
            )

            writer.write(overlayed)
            rendered_frames += 1

        if rendered_frames == 0:
            raise RuntimeError(f"Could not render any preview frames from {video_path}")

    finally:
        capture.release()

        if writer is not None:
            writer.release()

    logger.info(
        "Preview rendered: %s (%d sampled frames at %.2f FPS, %dx%d)",
        output_path,
        rendered_frames,
        output_fps,
        output_width,
        output_height,
    )

    return output_path


def _open_video_writer(
    *,
    output_path: Path,
    output_fps: float,
    output_size: tuple[int, int],
):
    """Open an MP4 writer using the first supported codec."""
    import cv2

    codec_candidates = ("mp4v", "avc1")

    for codec in codec_candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            output_fps,
            output_size,
        )

        if writer.isOpened():
            logger.debug(
                "Using preview codec %s for %s",
                codec,
                output_path,
            )
            return writer

        writer.release()

    raise RuntimeError(
        f"Could not open an MP4 video writer using codecs: {', '.join(codec_candidates)}"
    )
