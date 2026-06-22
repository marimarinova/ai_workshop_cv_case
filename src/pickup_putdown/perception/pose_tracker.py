"""Pose inference on person-active spans for Layer 0B.

Runs a configured YOLO pose model over person-active spans at a configurable
target FPS. Preserves source timestamps, clamps to clip duration, and records
valid wrist observations without silently interpolating missing keypoints.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pickup_putdown.common.schemas import ActiveSpan, PoseObservation
from pickup_putdown.config import PoseConfig
from pickup_putdown.ingestion.video_probe import probe_video

logger = logging.getLogger(__name__)

_LEFT_WRIST_INDEX = 9
_RIGHT_WRIST_INDEX = 10


@dataclass
class PoseTracker:
    """Run YOLO pose detection on a single video file.

    Parameters
    ----------
    video_path : Path
        Path to the source MP4/video file.
    pose_cfg : PoseConfig
        Pose inference configuration.
    active_spans : list[ActiveSpan] | None
        Active spans to restrict pose inference to. When ``None`` the entire
        clip is processed. An explicitly empty list processes no frames.
    """

    video_path: Path
    pose_cfg: PoseConfig
    active_spans: list[ActiveSpan] | None = None

    _model: object | None = field(default=None, repr=False)
    _device: str = field(default="cpu", repr=False)
    _source_fps: float = field(default=0.0, repr=False)
    _total_frames: int = field(default=0, repr=False)
    _clip_duration_s: float = field(default=0.0, repr=False)
    _vid_stride: int = field(default=1, repr=False)
    _sample_frames: list[int] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.video_path = Path(self.video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self._source_fps = self._read_source_fps()
        self._total_frames = self._read_total_frames()
        self._clip_duration_s = self._total_frames / max(self._source_fps, 1e-9)
        self._vid_stride = max(
            1,
            round(self._source_fps / max(float(self.pose_cfg.target_fps), 1e-9)),
        )
        self._sample_frames = self._compute_sample_frames(
            total_frames=self._total_frames,
            source_fps=self._source_fps,
            target_fps=float(self.pose_cfg.target_fps),
        )

        effective_fps = (
            len(self._sample_frames) / self._clip_duration_s if self._clip_duration_s > 0 else 0.0
        )
        logger.info(
            "Video %s: fps=%.2f, frames=%d, nominal_stride=%d, "
            "samples=%d, effective_sample_fps=%.2f",
            self.video_path,
            self._source_fps,
            self._total_frames,
            self._vid_stride,
            len(self._sample_frames),
            effective_fps,
        )

    # ------------------------------------------------------------------
    # Video metadata helpers
    # ------------------------------------------------------------------

    def _read_source_fps(self) -> float:
        result = probe_video(self.video_path)
        if not result.decode_ok:
            raise RuntimeError(f"Cannot decode video: {result.probe_error}")

        fps = result.fps or result.probe_fps
        if fps is None or fps <= 0:
            raise RuntimeError(f"Could not determine FPS for {self.video_path}")
        return float(fps)

    def _read_total_frames(self) -> int:
        import cv2

        cap = cv2.VideoCapture(str(self.video_path))
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {self.video_path}")

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()

        if total <= 0:
            raise RuntimeError(f"Could not determine frame count for {self.video_path}")
        return total

    @staticmethod
    def _compute_sample_frames(
        total_frames: int,
        vid_stride: int | None = None,
        *,
        source_fps: float | None = None,
        target_fps: float | None = None,
    ) -> list[int]:
        """Return deterministic sampled frame indices.

        The legacy ``(total_frames, vid_stride)`` call remains supported for
        compatibility with existing tests. Internal use supplies
        ``source_fps`` and ``target_fps`` so sampling is timestamp-based and
        avoids the 20 FPS / 8 FPS rounding problem.
        """
        if total_frames <= 0:
            return []

        if source_fps is None or target_fps is None:
            if vid_stride is None or vid_stride <= 0:
                raise ValueError("vid_stride must be positive")
            return list(range(0, total_frames, vid_stride))

        if source_fps <= 0:
            raise ValueError("source_fps must be positive")
        if target_fps <= 0:
            raise ValueError("target_fps must be positive")

        effective_target = min(target_fps, source_fps)
        frame_step = source_fps / effective_target

        frames: list[int] = []
        position = 0.0
        previous = -1

        while True:
            frame_index = int(round(position))
            if frame_index >= total_frames:
                break
            if frame_index != previous:
                frames.append(frame_index)
                previous = frame_index
            position += frame_step

        return frames

    # ------------------------------------------------------------------
    # Active-span filtering
    # ------------------------------------------------------------------

    def _active_frame_indices(self) -> set[int]:
        """Return sampled source-frame indices inside configured active spans."""
        if self.active_spans is None:
            return set(self._sample_frames)
        if not self.active_spans:
            return set()

        active_frames: set[int] = set()
        sampled = self._sample_frames

        for span in self.active_spans:
            start_s = max(0.0, float(span.t_start))
            end_s = min(self._clip_duration_s, float(span.t_end))
            if end_s < start_s:
                logger.warning(
                    "Skipping invalid active span %.3f-%.3f for %s",
                    start_s,
                    end_s,
                    self.video_path,
                )
                continue

            start_frame = max(0, math.floor(start_s * self._source_fps))
            end_frame = min(
                self._total_frames - 1,
                math.ceil(end_s * self._source_fps),
            )

            for frame_index in sampled:
                if start_frame <= frame_index <= end_frame:
                    active_frames.add(frame_index)

        return active_frames

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _resolve_device(self) -> str:
        configured = str(self.pose_cfg.device)
        if configured != "auto":
            return configured

        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self) -> object:
        from ultralytics import YOLO

        if self._model is not None:
            return self._model

        self._device = self._resolve_device()
        logger.info(
            "Loading YOLO pose model from %s on %s",
            self.pose_cfg.model_path,
            self._device,
        )
        self._model = YOLO(self.pose_cfg.model_path)
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> list[PoseObservation]:
        """Run pose inference and return timestamped wrist observations."""
        import cv2

        model = self._load_model()
        active_frames = self._active_frame_indices()
        clip_id = self._extract_clip_id()

        if not active_frames:
            logger.info("No active sample frames for %s", self.video_path)
            return []

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open video: {self.video_path}")

        all_observations: list[PoseObservation] = []
        max_required_frame = max(active_frames)
        processed_frames = 0
        frame_index = 0

        try:
            # Sequential decoding is substantially faster than seeking with
            # CAP_PROP_POS_FRAMES for every sampled frame.
            while frame_index <= max_required_frame:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(
                        "Video ended while reading frame %d of %s",
                        frame_index,
                        self.video_path,
                    )
                    break

                if frame_index not in active_frames:
                    frame_index += 1
                    continue

                processed_frames += 1
                timestamp_s = min(
                    frame_index / self._source_fps,
                    self._clip_duration_s,
                )

                results = model.track(
                    frame,
                    persist=True,
                    conf=self.pose_cfg.pose_confidence,
                    max_det=self.pose_cfg.max_detections,
                    classes=[0],
                    verbose=False,
                    imgsz=self.pose_cfg.image_size,
                    half=(bool(self.pose_cfg.half) and self._device.startswith("cuda")),
                    device=self._device,
                )

                if not results:
                    frame_index += 1
                    continue

                result = results[0]
                boxes = result.boxes
                if boxes is None or boxes.xyxy is None:
                    frame_index += 1
                    continue

                n_detections = len(boxes.xyxy)
                for detection_index in range(n_detections):
                    track_id = (
                        int(boxes.id[detection_index].item()) if boxes.id is not None else None
                    )
                    detection_confidence = float(boxes.conf[detection_index].item())
                    x1, y1, x2, y2 = [
                        float(value) for value in boxes.xyxy[detection_index].tolist()
                    ]

                    keypoints = self._extract_keypoints(
                        result,
                        detection_index,
                    )
                    actor_id = f"actor_{track_id}" if track_id is not None else "actor_untracked"

                    for hand_side, keypoint_name in (
                        ("left", "left_wrist"),
                        ("right", "right_wrist"),
                    ):
                        keypoint = keypoints.get(keypoint_name)
                        if keypoint is None:
                            continue

                        wrist_x, wrist_y, wrist_confidence = keypoint
                        if wrist_confidence < self.pose_cfg.pose_confidence:
                            continue

                        all_observations.append(
                            PoseObservation(
                                clip_id=clip_id,
                                timestamp_s=timestamp_s,
                                source_frame_index=frame_index,
                                sample_index=processed_frames - 1,
                                actor_id=actor_id,
                                hand_side=hand_side,
                                wrist_x=wrist_x,
                                wrist_y=wrist_y,
                                wrist_confidence=wrist_confidence,
                                person_bbox_x1=x1,
                                person_bbox_y1=y1,
                                person_bbox_x2=x2,
                                person_bbox_y2=y2,
                                pose_association_confidence=(detection_confidence),
                                is_valid=True,
                            )
                        )

                frame_index += 1
        finally:
            cap.release()

        all_observations.sort(
            key=lambda observation: (
                observation.clip_id,
                observation.timestamp_s,
                observation.actor_id,
                observation.hand_side,
            )
        )

        logger.info(
            "Pose complete: %d wrist observations from %d processed frames",
            len(all_observations),
            processed_frames,
        )
        return all_observations

    @staticmethod
    def _extract_keypoints(
        result: Any,
        detection_index: int,
    ) -> dict[str, tuple[float, float, float]]:
        """Extract COCO left/right wrists from one Ultralytics result.

        Ultralytics exposes pose keypoints on ``result.keypoints``. COCO
        indices 9 and 10 correspond to the left and right wrist.
        """
        extracted: dict[str, tuple[float, float, float]] = {}

        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or keypoints.xy is None:
            return extracted

        if detection_index < 0 or detection_index >= len(keypoints.xy):
            return extracted

        coordinates = keypoints.xy[detection_index]
        confidences = keypoints.conf[detection_index] if keypoints.conf is not None else None

        wrist_indices = {
            _LEFT_WRIST_INDEX: "left_wrist",
            _RIGHT_WRIST_INDEX: "right_wrist",
        }

        for keypoint_index, keypoint_name in wrist_indices.items():
            if keypoint_index >= len(coordinates):
                continue

            wrist_x = float(coordinates[keypoint_index][0].item())
            wrist_y = float(coordinates[keypoint_index][1].item())
            wrist_confidence = (
                float(confidences[keypoint_index].item()) if confidences is not None else 1.0
            )

            if not all(
                math.isfinite(value)
                for value in (
                    wrist_x,
                    wrist_y,
                    wrist_confidence,
                )
            ):
                logger.debug(
                    "Ignoring non-finite %s keypoint for detection %d",
                    keypoint_name,
                    detection_index,
                )
                continue

            extracted[keypoint_name] = (
                wrist_x,
                wrist_y,
                wrist_confidence,
            )

        return extracted

    def _extract_clip_id(self) -> str:
        return f"clip_{self.video_path.stem}"
