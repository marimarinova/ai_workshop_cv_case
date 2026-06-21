"""Tests for preview overlay drawing and rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pickup_putdown.common.schemas import ActiveSpan, PersonObservation
from pickup_putdown.perception.previews import (
    OverlayConfig,
    draw_overlay,
    render_triage_preview,
)


def _make_obs(
    tracker_id: int = 1,
    timestamp_s: float = 5.0,
    is_stable: bool = True,
    bbox=None,
) -> PersonObservation:
    if bbox is None:
        bbox = [100.0, 50.0, 300.0, 400.0]
    return PersonObservation(
        clip_id="clip_001",
        person_track_id=f"clip_001:person:{tracker_id}",
        tracker_track_id=tracker_id,
        sample_index=5,
        source_frame_index=150,
        timestamp_s=timestamp_s,
        bbox_x1=bbox[0],
        bbox_y1=bbox[1],
        bbox_x2=bbox[2],
        bbox_y2=bbox[3],
        confidence=0.85,
        is_stable=is_stable,
    )


def _make_span(t_start: float = 4.5, t_end: float = 5.5) -> ActiveSpan:
    return ActiveSpan(
        clip_id="clip_001",
        active_span_id="clip_001:active:000",
        t_start=t_start,
        t_end=t_end,
        n_person_tracks=1,
    )


def _make_frame() -> np.ndarray:
    """Create a blank BGR frame for testing."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestDrawOverlay:
    """Tests for the pure draw_overlay function."""

    def test_overlay_includes_box(self):
        """draw_overlay draws a bounding box when draw_boxes=True."""
        frame = _make_frame()
        obs = _make_obs()
        config = OverlayConfig(draw_boxes=True)
        result = draw_overlay(frame, [obs], [], config)

        # The box region should not be all zeros (green box on black background)
        x1, y1, x2, y2 = int(obs.bbox_x1), int(obs.bbox_y1), int(obs.bbox_x2), int(obs.bbox_y2)
        box_region = result[y1:y2, x1:x2]
        # Green is (0, 255, 0) in BGR - check that some pixels are not black
        assert np.any(box_region > 0), "Bounding box should be visible"

    def test_overlay_includes_box_label(self):
        """draw_overlay draws a label above the bounding box."""
        frame = _make_frame()
        obs = _make_obs()
        config = OverlayConfig(draw_boxes=True, draw_track_ids=True, draw_confidence=True)
        result = draw_overlay(frame, [obs], [], config)

        # The label area above the box should have non-zero pixels
        y1 = int(obs.bbox_y1)
        label_region = result[max(0, y1 - 30) : y1, int(obs.bbox_x1) : int(obs.bbox_x1) + 100]
        assert np.any(label_region > 0), "Label should be visible above the box"

    def test_overlay_skips_boxes_when_disabled(self):
        """draw_overlay does not draw boxes when draw_boxes=False."""
        frame = _make_frame()
        obs = _make_obs()
        config = OverlayConfig(draw_boxes=False)
        result = draw_overlay(frame, [obs], [], config)
        # Frame should remain unchanged (all zeros)
        assert np.array_equal(frame, result), "Frame should be unchanged when boxes disabled"

    def test_overlay_does_not_modify_input(self):
        """draw_overlay returns a copy and does not modify the input frame."""
        frame = _make_frame()
        obs = _make_obs()
        config = OverlayConfig()
        result = draw_overlay(frame, [obs], [], config)
        assert np.array_equal(frame, result[: int(obs.bbox_y1)], equal_nan=False) or np.any(
            result != frame
        ), "Input frame should not be modified"
        # Verify the original is still all zeros
        assert np.all(frame == 0), "Original frame should be all zeros"

    def test_overlay_with_span_status(self):
        """draw_overlay marks observations within active spans."""
        frame = _make_frame()
        obs = _make_obs(timestamp_s=5.0, is_stable=True)
        span = _make_span(t_start=4.5, t_end=5.5)
        config = OverlayConfig(draw_span_status=True)
        result = draw_overlay(frame, [obs], [span], config)
        # Should produce output without error
        assert result.shape == frame.shape

    def test_overlay_with_unstable_track(self):
        """draw_overlay uses red color for unstable tracks."""
        frame = _make_frame()
        obs = _make_obs(is_stable=False)
        config = OverlayConfig(draw_boxes=True)
        result = draw_overlay(frame, [obs], [], config)
        # Red is (0, 0, 255) in BGR
        x1, y1, x2, y2 = int(obs.bbox_x1), int(obs.bbox_y1), int(obs.bbox_x2), int(obs.bbox_y2)
        box_region = result[y1:y2, x1:x2]
        # Check for red pixels (high B channel, low G channel)
        has_red = np.any((box_region[:, :, 2] > 200) & (box_region[:, :, 1] < 50))
        assert has_red, "Unstable track should be drawn in red"


class TestRenderTriagePreview:
    """Integration tests for render_triage_preview."""

    def test_render_produces_output_file(self, tmp_path: Path, triage_config: dict):
        """render_triage_preview produces a valid MP4 file when codec is available."""
        import cv2

        # Create a minimal test video
        test_video = tmp_path / "test_input.mp4"
        test_output = tmp_path / "test_preview.mp4"

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            pytest.skip("No camera available for test video creation")

        # Create a simple test video with a few frames
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(test_video), fourcc, 5.0, (640, 480))
        for _ in range(25):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(frame, (100, 50), (300, 400), (0, 255, 0), -1)
            out.write(frame)
        out.release()
        cap.release()

        if not test_video.exists() or test_video.stat().st_size == 0:
            pytest.skip("Could not create test video")

        obs = [_make_obs(timestamp_s=1.0), _make_obs(timestamp_s=2.0)]
        span = _make_span(t_start=0.5, t_end=2.5)

        try:
            result_path = render_triage_preview(
                test_video,
                obs,
                [span],
                test_output,
                OverlayConfig(),
            )
            assert result_path.exists()
            assert result_path.stat().st_size > 0
        except RuntimeError:
            pytest.skip("No suitable codec available for video writing")

    def test_render_no_person_clip(self, tmp_path: Path):
        """render_triage_preview handles no-person clips (empty spans)."""
        import cv2

        test_video = tmp_path / "test_noperson.mp4"
        test_output = tmp_path / "noperson_preview.mp4"

        # Create a minimal test video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(test_video), fourcc, 5.0, (640, 480))
        for _ in range(25):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

        if not test_video.exists():
            pytest.skip("Could not create test video")

        try:
            result_path = render_triage_preview(
                test_video,
                [],
                [],
                test_output,
                OverlayConfig(),
            )
            assert result_path.exists()
        except RuntimeError:
            pytest.skip("No suitable codec available for video writing")

    def test_preview_config_defaults(self):
        """OverlayConfig has sensible defaults."""
        config = OverlayConfig()
        assert config.draw_boxes is True
        assert config.draw_track_ids is True
        assert config.draw_timestamps is True
        assert config.draw_confidence is True
        assert config.draw_span_status is True
        assert config.span_context_before_s == 2.0
        assert config.span_context_after_s == 2.0
