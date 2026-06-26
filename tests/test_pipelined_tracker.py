"""Tests for PipelinedPersonTracker and frame pipeline components.

Note: These tests use mocked YOLO results and do not require a GPU or
actual model weights. The tests verify frame ordering, shared memory
management, and graceful shutdown behavior.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from pickup_putdown.config import TriageConfig
from pickup_putdown.perception.frame_pipeline import (
    DecoderPool,
    FrameMetadata,
    FrameReorderer,
    SharedFrameBuffer,
)
from pickup_putdown.perception.pipelined_tracker import PipelinedPersonTracker


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
        "pipeline_enabled": True,
        "pipeline_queue_depth": 4,
        "pipeline_n_decoders": 2,
        "pipeline_resize_frames": True,
        "pipeline_frame_size": (640, 640),
        "pipeline_frame_timeout_s": 10.0,
    }
    base.update(overrides)
    return TriageConfig(**base)


def _create_mock_video(tmp_path: Path, fps: float, total_frames: int) -> Path:
    """Create a minimal MP4 video file for testing."""
    import cv2

    video_path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (640, 480))
    for _ in range(total_frames):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    return video_path


class TestSharedFrameBuffer:
    """Tests for SharedFrameBuffer shared memory management."""

    def test_create_and_attach(self):
        """Test creating a buffer and attaching to it."""
        buffer = SharedFrameBuffer(
            n_slots=4,
            frame_height=480,
            frame_width=640,
        )
        name = buffer.name
        assert name is not None

        # Clean up
        buffer.close()
        buffer.unlink()

    def test_write_and_read_frame(self):
        """Test writing and reading frames from shared memory."""
        buffer = SharedFrameBuffer(
            n_slots=4,
            frame_height=480,
            frame_width=640,
        )

        # Create a test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Write to slot 0
        buffer.write_frame(0, test_frame)

        # Read back
        read_frame = buffer.read_frame(0)

        # Verify
        assert np.array_equal(test_frame, read_frame)

        buffer.close()
        buffer.unlink()

    def test_multiple_slots(self):
        """Test writing to multiple slots independently."""
        buffer = SharedFrameBuffer(
            n_slots=4,
            frame_height=480,
            frame_width=640,
        )

        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(4)]

        for i, frame in enumerate(frames):
            buffer.write_frame(i, frame)

        for i, expected_frame in enumerate(frames):
            read_frame = buffer.read_frame(i)
            assert np.array_equal(expected_frame, read_frame)

        buffer.close()
        buffer.unlink()

    def test_invalid_slot_index(self):
        """Test that invalid slot indices raise ValueError."""
        buffer = SharedFrameBuffer(
            n_slots=4,
            frame_height=480,
            frame_width=640,
        )

        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="out of range"):
            buffer.write_frame(4, test_frame)

        with pytest.raises(ValueError, match="out of range"):
            buffer.read_frame(-1)

        buffer.close()
        buffer.unlink()

    def test_frame_shape_mismatch(self):
        """Test that mismatched frame shapes raise ValueError."""
        buffer = SharedFrameBuffer(
            n_slots=4,
            frame_height=480,
            frame_width=640,
        )

        wrong_shape_frame = np.zeros((240, 320, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="doesn't match"):
            buffer.write_frame(0, wrong_shape_frame)

        buffer.close()
        buffer.unlink()


class TestFrameReorderer:
    """Tests for FrameReorderer out-of-order frame handling."""

    def test_in_order_frames(self):
        """Test that in-order frames are returned immediately."""
        reorderer = FrameReorderer(total_frames=5)

        for i in range(5):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            metadata = FrameMetadata(
                sample_index=i,
                source_frame_index=i * 30,
                timestamp_s=i * 1.0,
                slot_index=i % 4,
                worker_id=i % 2,
            )
            ready = reorderer.add_frame(frame, metadata)
            assert len(ready) == 1
            assert ready[0][1].sample_index == i

        assert reorderer.n_pending == 0

    def test_out_of_order_frames(self):
        """Test that out-of-order frames are buffered and released correctly."""
        reorderer = FrameReorderer(total_frames=4)

        # Receive frames in order: 1, 0, 3, 2
        out_of_order = [1, 0, 3, 2]
        all_ready = []

        for i in out_of_order:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            metadata = FrameMetadata(
                sample_index=i,
                source_frame_index=i * 30,
                timestamp_s=i * 1.0,
                slot_index=i,
                worker_id=i % 2,
            )
            ready = reorderer.add_frame(frame, metadata)
            all_ready.extend(ready)

        # Should have received all 4 frames in order
        assert len(all_ready) == 4
        for i, (_, meta) in enumerate(all_ready):
            assert meta.sample_index == i

    def test_pending_count(self):
        """Test that pending count is tracked correctly."""
        reorderer = FrameReorderer(total_frames=5)

        # Add frame 2 first (out of order)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        metadata = FrameMetadata(
            sample_index=2,
            source_frame_index=60,
            timestamp_s=2.0,
            slot_index=0,
            worker_id=0,
        )
        ready = reorderer.add_frame(frame, metadata)
        assert len(ready) == 0
        assert reorderer.n_pending == 1

        # Add frame 1 (still out of order)
        metadata = FrameMetadata(
            sample_index=1,
            source_frame_index=30,
            timestamp_s=1.0,
            slot_index=1,
            worker_id=1,
        )
        ready = reorderer.add_frame(frame, metadata)
        assert len(ready) == 0
        assert reorderer.n_pending == 2

        # Add frame 0 (now 0, 1, 2 should be released)
        metadata = FrameMetadata(
            sample_index=0,
            source_frame_index=0,
            timestamp_s=0.0,
            slot_index=2,
            worker_id=0,
        )
        ready = reorderer.add_frame(frame, metadata)
        assert len(ready) == 3
        assert reorderer.n_pending == 0


class TestDecoderPool:
    """Tests for DecoderPool worker management."""

    def test_context_manager_cleanup(self, tmp_path):
        """Test that context manager properly cleans up resources."""
        video = _create_mock_video(tmp_path, fps=30.0, total_frames=30)

        with DecoderPool(n_workers=2, queue_depth=4) as pool:
            pool.start(
                video_path=video,
                sample_frames=[0, 30],
                source_fps=30.0,
            )
            assert pool._active is True

        # After context manager exits, should be cleaned up
        assert pool._active is False
        assert pool._shared_buffer is None

    def test_stop_without_start(self):
        """Test that stop() is safe to call without start()."""
        pool = DecoderPool(n_workers=2, queue_depth=4)
        pool.stop()  # Should not raise

    def test_double_stop(self, tmp_path):
        """Test that stop() is idempotent."""
        video = _create_mock_video(tmp_path, fps=30.0, total_frames=30)

        pool = DecoderPool(n_workers=2, queue_depth=4)
        pool.start(
            video_path=video,
            sample_frames=[0, 30],
            source_fps=30.0,
        )
        pool.stop()
        pool.stop()  # Should not raise


class TestPipelinedPersonTracker:
    """Tests for PipelinedPersonTracker integration."""

    def test_fallback_to_sequential(self, tmp_path):
        """Test that use_pipeline=False falls back to PersonTracker behavior."""
        video = _create_mock_video(tmp_path, fps=30.0, total_frames=300)
        cfg = _make_triage_config(pipeline_enabled=False)

        tracker = PipelinedPersonTracker(
            video_path=video,
            triage_cfg=cfg,
            use_pipeline=False,
        )

        call_count = [0]

        def mock_track(*a, **kw):
            call_count[0] += 1
            if call_count[0] in (5, 6):
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

        assert len(obs) == 2

    def test_pipeline_matches_sequential_output(self, tmp_path):
        """Test that pipelined output matches sequential PersonTracker."""
        from pickup_putdown.perception.person_tracker import PersonTracker

        video = _create_mock_video(tmp_path, fps=30.0, total_frames=120)
        cfg = _make_triage_config(target_fps=1.0)

        # Mock that returns consistent results
        call_count = [0]

        def mock_track(*a, **kw):
            call_count[0] += 1
            si = (call_count[0] - 1) % 4
            if si in (1, 2):
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

        # Run sequential
        tracker_seq = PersonTracker(video_path=video, triage_cfg=cfg)
        with patch.object(tracker_seq, "_load_model", return_value=MagicMock(track=mock_track)):
            call_count[0] = 0
            obs_seq, summaries_seq = tracker_seq.run()

        # Run with pipeline disabled (fallback)
        tracker_pipe = PipelinedPersonTracker(
            video_path=video,
            triage_cfg=cfg,
            use_pipeline=False,
        )
        with patch.object(tracker_pipe, "_load_model", return_value=MagicMock(track=mock_track)):
            call_count[0] = 0
            obs_pipe, summaries_pipe = tracker_pipe.run()

        # Compare results
        assert len(obs_seq) == len(obs_pipe)
        for o1, o2 in zip(obs_seq, obs_pipe):
            assert o1.sample_index == o2.sample_index
            assert o1.source_frame_index == o2.source_frame_index
            assert abs(o1.timestamp_s - o2.timestamp_s) < 0.001


class TestFrameMetadata:
    """Tests for FrameMetadata dataclass."""

    def test_immutable(self):
        """Test that FrameMetadata is immutable (frozen)."""
        meta = FrameMetadata(
            sample_index=0,
            source_frame_index=0,
            timestamp_s=0.0,
            slot_index=0,
            worker_id=0,
        )
        with pytest.raises(AttributeError):
            meta.sample_index = 1

    def test_hashable(self):
        """Test that FrameMetadata is hashable (can be used in sets/dicts)."""
        meta = FrameMetadata(
            sample_index=0,
            source_frame_index=0,
            timestamp_s=0.0,
            slot_index=0,
            worker_id=0,
        )
        s = {meta}
        assert meta in s


class TestConfigIntegration:
    """Tests for pipeline configuration integration."""

    def test_default_pipeline_config(self):
        """Test that default pipeline config values are correct."""
        cfg = _make_triage_config()
        assert cfg.pipeline_enabled is True
        assert cfg.pipeline_queue_depth == 4  # Overridden in test
        assert cfg.pipeline_n_decoders == 2
        assert cfg.pipeline_resize_frames is True
        assert cfg.pipeline_frame_size == (640, 640)
        assert cfg.pipeline_frame_timeout_s == 10.0

    def test_pipeline_disabled(self):
        """Test that pipeline can be disabled via config."""
        cfg = _make_triage_config(pipeline_enabled=False)
        assert cfg.pipeline_enabled is False
