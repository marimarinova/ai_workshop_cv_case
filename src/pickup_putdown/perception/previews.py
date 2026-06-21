"""Preview renderer for triage QA with overlay drawing and video encoding."""

from __future__ import annotations

import logging
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
    box_color: tuple[int, int, int] = (0, 255, 0)  # BGR green
    text_color: tuple[int, int, int] = (255, 255, 255)  # BGR white
    text_scale: float = 0.5
    line_thickness: int = 1


def draw_overlay(
    frame: np.ndarray,
    observations: list[PersonObservation],
    spans: list[ActiveSpan],
    config: OverlayConfig,
) -> np.ndarray:
    """Draw bounding boxes and metadata on a single frame (pure function).

    Parameters
    ----------
    frame : np.ndarray
        BGR frame (H, W, 3) as read by OpenCV.
    observations : list[PersonObservation]
        All observations for the clip.
    spans : list[ActiveSpan]
        Active spans for the clip.
    config : OverlayConfig
        Overlay rendering configuration.

    Returns
    -------
    np.ndarray
        New frame with overlays (does not modify input).
    """
    import cv2

    out = frame.copy()
    h, w = out.shape[:2]

    # Build span time ranges for quick lookup
    span_ranges: list[tuple[float, float]] = [(s.t_start, s.t_end) for s in spans]

    # Group observations by source_frame_index for this frame
    # We need to know which frame index this overlay corresponds to.
    # Since this is a pure function, the caller must pass observations
    # filtered to the relevant frame range.
    for obs in observations:
        if not config.draw_boxes:
            break

        x1, y1, x2, y2 = int(obs.bbox_x1), int(obs.bbox_y1), int(obs.bbox_x2), int(obs.bbox_y2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Determine color: green for stable, red for unstable
        color = config.box_color if obs.is_stable else (0, 0, 255)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, config.line_thickness)

        # Build label
        label_parts: list[str] = []
        if config.draw_track_ids and obs.tracker_track_id is not None:
            label_parts.append(f"ID:{obs.tracker_track_id}")
        if config.draw_confidence:
            label_parts.append(f"{obs.confidence:.2f}")
        if config.draw_timestamps:
            label_parts.append(f"{obs.timestamp_s:.1f}s")
        if config.draw_span_status:
            in_span = _is_in_span(obs.timestamp_s, span_ranges)
            label_parts.append("SPAN" if in_span else "idle")

        if label_parts:
            label = " ".join(label_parts)
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, config.text_scale, config.line_thickness
            )
            cv2.rectangle(
                out,
                (x1, y1 - th - 4),
                (x1 + tw, y1),
                color,
                -1,
            )
            cv2.putText(
                out,
                label,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.text_scale,
                config.text_color,
                config.line_thickness,
            )

    return out


def _is_in_span(timestamp_s: float, span_ranges: list[tuple[float, float]]) -> bool:
    """Check if a timestamp falls within any active span."""
    return any(lo <= timestamp_s <= hi for lo, hi in span_ranges)


def render_triage_preview(
    video_path: Path,
    observations: list[PersonObservation],
    spans: list[ActiveSpan],
    output_path: Path,
    config: OverlayConfig | None = None,
) -> Path:
    """Render a preview MP4 with track overlays.

    For person-positive clips, renders active spans with context.
    For no-person clips, renders a representative sample of the clip.

    Parameters
    ----------
    video_path : Path
        Path to the source video.
    observations : list[PersonObservation]
        All person observations for the clip.
    spans : list[ActiveSpan]
        Active spans for the clip.
    output_path : Path
        Destination MP4 path.
    config : OverlayConfig | None
        Overlay configuration. Defaults used if None.

    Returns
    -------
    Path
        The output file path.
    """
    import cv2

    if config is None:
        config = OverlayConfig()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine frame range to render
    if spans:
        # Render active spans with context
        span_frames: list[tuple[int, int]] = []
        for span in spans:
            frame_start = max(0, int((span.t_start - config.span_context_before_s) * src_fps))
            frame_end = min(
                total_frames, int((span.t_end + config.span_context_after_s) * src_fps)
            )
            span_frames.append((frame_start, frame_end))

        # Merge overlapping span ranges
        span_frames.sort()
        merged_ranges: list[tuple[int, int]] = [span_frames[0]]
        for start, end in span_frames[1:]:
            if start <= merged_ranges[-1][1]:
                merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))
            else:
                merged_ranges.append((start, end))
    else:
        # No-person clip: render a representative sample (first 10 seconds or full clip)
        render_end = min(total_frames, int(10.0 * src_fps))
        merged_ranges = [(0, render_end)]

    # Use first span's observations for overlay (or all if no spans)
    if spans and observations:
        # Filter observations near the first span for overlay
        first_span = spans[0]
        span_obs = [
            o
            for o in observations
            if first_span.t_start - config.span_context_before_s
            <= o.timestamp_s
            <= first_span.t_end + config.span_context_after_s
        ]
    else:
        span_obs = observations

    # Determine output codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(output_path), fourcc, min(src_fps, 15.0), (src_w, src_h))

    if not out_writer.isOpened():
        # Fallback: try avc1
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out_writer = cv2.VideoWriter(str(output_path), fourcc, min(src_fps, 15.0), (src_w, src_h))

    if not out_writer.isOpened():
        # Last resort: write without codec specification
        out_writer = cv2.VideoWriter(str(output_path), -1, min(src_fps, 15.0), (src_w, src_h))

    rendered_frames = 0
    for start_frame, end_frame in merged_ranges:
        for frame_idx in range(start_frame, end_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            overlayed = draw_overlay(frame, span_obs, spans, config)
            out_writer.write(overlayed)
            rendered_frames += 1

    cap.release()
    out_writer.release()

    logger.info("Preview rendered: %s (%d frames)", output_path, rendered_frames)
    return output_path
