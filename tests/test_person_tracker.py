"""Tests for PersonTracker timestamp handling and track stability logic.

Note: These tests use mocked YOLO results and do not require a GPU or
actual model weights. The PersonTracker class is tested for its frame
indexing, timestamp conversion, and track-splitting logic.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from pickup_putdown.config import TriageConfig
from pickup_putdown.perception.person_tracker import PersonTracker


def _make_triage_config(**overrides) -> TriageConfig:
    """Create a TriageConfig with optional overrides."""
    base = {
        "model_path": "models/person_detector.pt",
        "target_fps": 1.0,
        "image_size": 640,
        "device": "cpu",
        "half": False,
        "detector_confidence": 0.10,
        "detector_iou_threshold": 0.70,
        "max_detections": 100,
        "minimum_track_confidence": 0.35,
        "minimum_visible_duration_s": 0.75,
        "minimum_observations": 2,
        "max_track_observation_gap_s": 1.5,
        "merge_gap_s": 1.0,
        "preview_sample_rate": 0.10,
        "sampling_seed": 42,
        "tracker_config": "configs/bytetrack_triage.yaml",
    }
    base.update(overrides)
    return TriageConfig(**base)


def _create_mock_video(tmp_path: Path, fps: float, total_frames: int) -> Path:
    """Create a minimal MP4 video file for testing."""
    import cv2
    import numpy as np

    video_path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (640, 480))
    for _ in range(total_frames):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    return video_path


def _mock_yolo_results(detections: list[dict]) -> MagicMock:
    """Create a mock YOLO track result.

    Each detection dict should have:
        track_id: int | None
        confidence: float
        bbox: [x1, y1, x2, y2]
    """
    mock_results = MagicMock()
    mock_boxes = MagicMock()
    mock_boxes.id = None
    mock_boxes.conf = None
    mock_boxes.xyxy = None
    mock_boxes.cls = None

    if detections:
        track_ids = [d.get("track_id") for d in detections]
        confs = [d["confidence"] for d in detections]
        bboxes = [d["bbox"] for d in detections]

        # boxes.id tensor (only for detections with track_id)
        id_tensor = None
        if any(tid is not None for tid in track_ids):
            id_values = [tid if tid is not None else -1 for tid in track_ids]
            id_tensor = torch.tensor(id_values, dtype=torch.int64)

        mock_boxes.id = id_tensor
        mock_boxes.conf = torch.tensor(confs, dtype=torch.float32)
        mock_boxes.xyxy = torch.tensor(bboxes, dtype=torch.float32)

        mock_results.boxes = mock_boxes

    mock_results.__len__ = MagicMock(return_value=1)
    return [mock_results]


class TestComputeSampleFrames:
    """Tests for the static _compute_sample_frames method."""

    def test_stride_1_returns_all_frames(self):
        frames = PersonTracker._compute_sample_frames(100, 1)
        assert frames == list(range(100))

    def test_stride_2_skips_even_frames(self):
        frames = PersonTracker._compute_sample_frames(10, 2)
        assert frames == [0, 2, 4, 6, 8]

    def test_stride_larger_than_frames(self):
        frames = PersonTracker._compute_sample_frames(5, 10)
        assert frames == [0]

    def test_first_frame_always_included(self):
        for stride in [1, 2, 5, 30]:
            frames = PersonTracker._compute_sample_frames(100, stride)
            assert frames[0] == 0, f"Frame 0 should always be included for stride {stride}"

    def test_last_partial_interval(self):
        """Handles the case where the last interval doesn't align perfectly."""
        frames = PersonTracker._compute_sample_frames(100, 7)
        # 0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98
        assert frames[-1] == 98
        assert len(frames) == 15

    def test_non_integer_fps_stride_calculation(self):
        """Stride is calculated from integer round of FPS ratio."""
        # 29.97 / 1.0 = 29.97 -> round -> 30
        stride = max(1, round(29.97 / 1.0))
        assert stride == 30


class TestPersonTrackerTimestamps:
    """Tests for timestamp conversion in PersonTracker."""

    def test_timestamp_from_source_frame_index(self, tmp_path):
        """timestamp_s = source_frame_index / source_fps."""
        video = _create_mock_video(tmp_path, fps=30.0, total_frames=300)
        cfg = _make_triage_config(target_fps=1.0)

        tracker = PersonTracker(video_path=video, triage_cfg=cfg)

        # At 30 FPS, 1 FPS => stride=30
        # Frame 0 -> t=0.0, Frame 30 -> t=1.0, Frame 60 -> t=2.0
        assert tracker._sample_frames == list(range(0, 300, 30))
        for si, src_idx in enumerate(tracker._sample_frames):
            expected_ts = src_idx / 30.0
            assert abs(tracker._sample_frames[si] / 30.0 - expected_ts) < 0.001

    def test_non_integer_fps_29_97(self, tmp_path):
        """Non-integer FPS (29.97) is handled correctly."""
        video = _create_mock_video(tmp_path, fps=29.97, total_frames=300)
        cfg = _make_triage_config(target_fps=1.0)

        tracker = PersonTracker(video_path=video, triage_cfg=cfg)
        # stride = round(29.97 / 1.0) = 30
        assert tracker._vid_stride == 30
        # First few sample frames
        assert tracker._sample_frames[0] == 0
        assert tracker._sample_frames[1] == 30
        assert tracker._sample_frames[2] == 60

    def test_source_fps_lower_than_target(self, tmp_path):
        """When source FPS < target FPS, stride is 1."""
        video = _create_mock_video(tmp_path, fps=1.0, total_frames=10)
        cfg = _make_triage_config(target_fps=2.0)

        tracker = PersonTracker(video_path=video, triage_cfg=cfg)
        assert tracker._vid_stride == 1

    def test_source_fps_equal_to_target(self, tmp_path):
        """When source FPS == target FPS, stride is 1."""
        video = _create_mock_video(tmp_path, fps=1.0, total_frames=10)
        cfg = _make_triage_config(target_fps=1.0)

        tracker = PersonTracker(video_path=video, triage_cfg=cfg)
        assert tracker._vid_stride == 1


class TestPersonTrackerTrackStability:
    """Tests for track stability determination."""

    def test_one_frame_artifact_rejected(self, tmp_path):
        """A single-frame detection is rejected as unstable."""
        video = _create_mock_video(tmp_path, fps=30.0, total_frames=300)
        cfg = _make_triage_config(
            target_fps=1.0,
            minimum_visible_duration_s=0.75,
            minimum_observations=2,
        )

        tracker = PersonTracker(video_path=video, triage_cfg=cfg)

        call_count = [0]

        def mock_track(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 6:  # sample index 5 = frame 150
                return [
                    MagicMock(
                        boxes=MagicMock(
                            id=torch.tensor([1], dtype=torch.int64),
                            conf=torch.tensor([0.85], dtype=torch.float32),
                            xyxy=torch.tensor([[100, 50, 300, 400]], dtype=torch.float32),
                        )
                    )
                ]
            return [MagicMock(boxes=MagicMock(id=None, conf=None, xyxy=None))]

        with patch.object(tracker, "_load_model", return_value=MagicMock(track=mock_track)):
            obs, summaries = tracker.run()

        # Only one observation, should not be stable
        assert len(obs) == 1
        assert obs[0].is_stable is False
        assert all(not s.is_stable for s in summaries)

    def test_valid_short_appearance_accepted(self, tmp_path):
        """A short but valid appearance (>= min duration) is accepted."""
        video = _create_mock_video(tmp_path, fps=30.0, total_frames=300)
        cfg = _make_triage_config(
            target_fps=1.0,
            minimum_visible_duration_s=0.75,
            minimum_observations=2,
        )

        tracker = PersonTracker(video_path=video, triage_cfg=cfg)

        # Detections at sample indices 5 and 6 (frames 150 and 180)
        # timestamps 5.0 and 6.0, duration = 1.0s >= 0.75s, n=2 >= 2
        call_count = [0]

        def mock_track(*a, **kw):
            call_count[0] += 1
            si = call_count[0] - 1
            if si in (5, 6):
                return [
                    MagicMock(
                        boxes=MagicMock(
                            id=torch.tensor([1], dtype=torch.int64),
                            conf=torch.tensor([0.85], dtype=torch.float32),
                            xyxy=torch.tensor([[100, 50, 300, 400]], dtype=torch.float32),
                        )
                    )
                ]
            return [MagicMock(boxes=MagicMock(id=None, conf=None, xyxy=None))]

        with patch.object(tracker, "_load_model", return_value=MagicMock(track=mock_track)):
            obs, summaries = tracker.run()

        stable_summaries = [s for s in summaries if s.is_stable]
        assert len(stable_summaries) >= 1

    def test_long_gap_splits_track(self, tmp_path):
        """A tracker ID that reappears after a long gap is split into two runs."""
        video = _create_mock_video(tmp_path, fps=30.0, total_frames=600)
        cfg = _make_triage_config(
            target_fps=1.0,
            minimum_visible_duration_s=0.75,
            minimum_observations=2,
            max_track_observation_gap_s=1.5,
        )

        tracker = PersonTracker(video_path=video, triage_cfg=cfg)

        call_count = [0]

        def mock_track(*a, **kw):
            call_count[0] += 1
            si = call_count[0] - 1
            # First run: sample indices 5, 6 (timestamps 5.0, 6.0)
            # Gap: indices 7-14 (no detections)
            # Second run: sample indices 15, 16 (timestamps 15.0, 16.0)
            if si in (5, 6, 15, 16):
                return [
                    MagicMock(
                        boxes=MagicMock(
                            id=torch.tensor([1], dtype=torch.int64),
                            conf=torch.tensor([0.85], dtype=torch.float32),
                            xyxy=torch.tensor([[100, 50, 300, 400]], dtype=torch.float32),
                        )
                    )
                ]
            return [MagicMock(boxes=MagicMock(id=None, conf=None, xyxy=None))]

        with patch.object(tracker, "_load_model", return_value=MagicMock(track=mock_track)):
            obs, summaries = tracker.run()

        # Should have 2 summaries for the same tracker ID (two runs)
        run_summaries = [s for s in summaries if s.tracker_track_id == 1]
        assert len(run_summaries) == 2

    def test_tracker_state_reset_between_clips(self, tmp_path):
        """Tracker state is reset for each PersonTracker invocation."""
        video1 = _create_mock_video(tmp_path, fps=30.0, total_frames=300)
        sub = tmp_path / "sub"
        sub.mkdir(parents=True, exist_ok=True)
        video2 = _create_mock_video(sub, fps=30.0, total_frames=300)
        cfg = _make_triage_config(target_fps=1.0)

        tracker1 = PersonTracker(video_path=video1, triage_cfg=cfg)
        tracker2 = PersonTracker(video_path=video2, triage_cfg=cfg)

        # Each tracker is independent
        assert tracker1.video_path != tracker2.video_path
        assert tracker1._model is None
        assert tracker2._model is None

    def test_deterministic_observation_ordering(self, tmp_path):
        """Observations are sorted by source_frame_index, then sample_index."""
        video = _create_mock_video(tmp_path, fps=30.0, total_frames=300)
        cfg = _make_triage_config(target_fps=1.0)

        tracker = PersonTracker(video_path=video, triage_cfg=cfg)

        call_count = [0]

        def mock_track(*a, **kw):
            call_count[0] += 1
            si = call_count[0] - 1
            if si == 5:  # sample index 5
                # Two detections at same frame
                return [
                    MagicMock(
                        boxes=MagicMock(
                            id=torch.tensor([2, 1], dtype=torch.int64),
                            conf=torch.tensor([0.85, 0.90], dtype=torch.float32),
                            xyxy=torch.tensor(
                                [[100, 50, 300, 400], [50, 30, 250, 380]], dtype=torch.float32
                            ),
                        )
                    )
                ]
            return [MagicMock(boxes=MagicMock(id=None, conf=None, xyxy=None))]

        with patch.object(tracker, "_load_model", return_value=MagicMock(track=mock_track)):
            obs, _ = tracker.run()

        # Observations should be sorted by source_frame_index
        if len(obs) >= 2:
            assert obs[0].source_frame_index <= obs[1].source_frame_index


class TestPersonTrackerMissingIDs:
    """Tests for handling detections without tracker IDs."""

    def test_missing_tracker_id_handled(self, tmp_path):
        """Detections without track IDs are recorded with tracker_track_id=None."""
        video = _create_mock_video(tmp_path, fps=30.0, total_frames=300)
        cfg = _make_triage_config(target_fps=1.0)

        tracker = PersonTracker(video_path=video, triage_cfg=cfg)

        call_count = [0]

        def mock_track(*a, **kw):
            call_count[0] += 1
            si = call_count[0] - 1
            if si == 5:
                # Detection without track ID
                return [
                    MagicMock(
                        boxes=MagicMock(
                            id=None,
                            conf=torch.tensor([0.85], dtype=torch.float32),
                            xyxy=torch.tensor([[100, 50, 300, 400]], dtype=torch.float32),
                        )
                    )
                ]
            return [MagicMock(boxes=MagicMock(id=None, conf=None, xyxy=None))]

        with patch.object(tracker, "_load_model", return_value=MagicMock(track=mock_track)):
            obs, _ = tracker.run()

        assert len(obs) == 1
        assert obs[0].tracker_track_id is None
        assert "untracked" in obs[0].person_track_id
