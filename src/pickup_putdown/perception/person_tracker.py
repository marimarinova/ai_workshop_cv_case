"""Person detection and tracking on video files for Layer 0A triage."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from pickup_putdown.common.schemas import PersonObservation, TrackSummary
from pickup_putdown.config import AppConfig, TriageConfig
from pickup_putdown.ingestion.video_probe import probe_video

logger = logging.getLogger(__name__)


@dataclass
class PersonTracker:
    """Run YOLO person detection + ByteTrack on a single video file.

    Parameters
    ----------
    video_path : Path
        Path to the source MP4/video file.
    triage_cfg : TriageConfig
        Triage configuration with detection and acceptance thresholds.
    tracker_cfg : dict | None
        ByteTrack configuration overrides. If None, uses defaults.
    app_cfg : AppConfig | None
        Full application config (used to resolve tracker config path).
    """

    video_path: Path
    triage_cfg: TriageConfig
    tracker_cfg: dict | None = None
    app_cfg: AppConfig | None = None

    _model: object | None = field(default=None, repr=False)
    _source_fps: float = field(default=0.0, repr=False)
    _total_frames: int = field(default=0, repr=False)
    _vid_stride: int = field(default=1, repr=False)
    _sample_frames: list[int] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self._source_fps = self._read_source_fps()
        self._total_frames = self._read_total_frames()
        self._vid_stride = max(1, round(self._source_fps / self.triage_cfg.target_fps))
        self._sample_frames = self._compute_sample_frames(self._total_frames, self._vid_stride)
        logger.info(
            "Video %s: fps=%.2f, frames=%d, stride=%d, samples=%d",
            self.video_path,
            self._source_fps,
            self._total_frames,
            self._vid_stride,
            len(self._sample_frames),
        )

    def _read_source_fps(self) -> float:
        """Read source FPS using ffprobe via probe_video."""
        result = probe_video(self.video_path)
        if not result.decode_ok:
            raise RuntimeError(f"Cannot decode video: {result.probe_error}")
        fps = result.fps or result.probe_fps
        if fps is None or fps <= 0:
            raise RuntimeError(f"Could not determine FPS for {self.video_path}")
        return float(fps)

    def _read_total_frames(self) -> int:
        """Read total frame count using OpenCV."""
        import cv2

        cap = cv2.VideoCapture(str(self.video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total <= 0:
            raise RuntimeError(f"Could not determine frame count for {self.video_path}")
        return total

    @staticmethod
    def _compute_sample_frames(total_frames: int, vid_stride: int) -> list[int]:
        """Return source frame indices for sampling.

        Always includes frame 0. Handles non-integer FPS and last partial interval.
        """
        return list(range(0, total_frames, vid_stride))

    def _load_model(self) -> object:
        """Lazy-load the YOLO person detection model."""
        from ultralytics import YOLO

        if self._model is not None:
            return self._model

        device = self.triage_cfg.device
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading YOLO model from %s on %s", self.triage_cfg.model_path, device)
        model = YOLO(self.triage_cfg.model_path)
        self._model = model
        return model

    def run(self) -> tuple[list[PersonObservation], list[TrackSummary]]:
        """Run person detection and tracking on the video.

        Returns
        -------
        observations : list[PersonObservation]
            Flat list of timestamped person detections.
        summaries : list[TrackSummary]
            Per-tracker summaries with stability flags.
        """
        import cv2

        model = self._load_model()
        cap = cv2.VideoCapture(str(self.video_path))

        all_observations: list[PersonObservation] = []
        # tracker_track_id -> list of (sample_index, timestamp_s, bbox, confidence)
        track_points: dict[int | str, list[tuple[int, float, list[float], float]]] = {}

        clip_id = self._extract_clip_id()

        for si, src_frame_idx in enumerate(self._sample_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, src_frame_idx)
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame %d of %s", src_frame_idx, self.video_path)
                continue

            timestamp_s = src_frame_idx / self._source_fps

            results = model.track(
                frame,
                persist=True,
                conf=self.triage_cfg.detector_confidence,
                iou=self.triage_cfg.detector_iou_threshold,
                max_det=self.triage_cfg.max_detections,
                classes=[0],  # person class
                verbose=False,
                imgsz=self.triage_cfg.image_size,
                half=self.triage_cfg.half,
            )

            boxes = results[0].boxes
            if boxes is None:
                continue

            n_dets = len(boxes.xyxy) if boxes.xyxy is not None else 0
            if n_dets == 0:
                continue

            for i in range(n_dets):
                track_id = int(boxes.id[i].item()) if boxes.id is not None else None
                conf = float(boxes.conf[i].item())

                if conf < self.triage_cfg.minimum_track_confidence:
                    continue

                xyxy = boxes.xyxy[i].tolist()
                bbox = [float(v) for v in xyxy]

                obs = PersonObservation(
                    clip_id=clip_id,
                    person_track_id=f"{clip_id}:person:{track_id}"
                    if track_id is not None
                    else f"{clip_id}:person:untracked",
                    tracker_track_id=track_id,
                    sample_index=si,
                    source_frame_index=src_frame_idx,
                    timestamp_s=timestamp_s,
                    bbox_x1=bbox[0],
                    bbox_y1=bbox[1],
                    bbox_x2=bbox[2],
                    bbox_y2=bbox[3],
                    confidence=conf,
                    is_stable=False,
                )
                all_observations.append(obs)

                tid = track_id if track_id is not None else -1
                if tid not in track_points:
                    track_points[tid] = []
                track_points[tid].append((si, timestamp_s, bbox, conf))

        cap.release()

        # Mark stable tracks
        summaries = self._compute_summaries(clip_id, track_points)
        stable_tracker_ids = {s.tracker_track_id for s in summaries if s.is_stable}
        for obs in all_observations:
            tid = obs.tracker_track_id if obs.tracker_track_id is not None else -1
            obs.is_stable = tid in stable_tracker_ids

        # Sort observations deterministically
        all_observations.sort(
            key=lambda o: (o.source_frame_index, o.sample_index, o.tracker_track_id or -1)
        )

        logger.info(
            "Triage complete: %d observations, %d tracks (%d stable)",
            len(all_observations),
            len(summaries),
            sum(1 for s in summaries if s.is_stable),
        )

        return all_observations, summaries

    def _compute_summaries(
        self,
        clip_id: str,
        track_points: dict[int | str, list[tuple[int, float, list[float], float]]],
    ) -> list[TrackSummary]:
        """Compute track summaries, splitting by contiguous runs."""
        summaries: list[TrackSummary] = []

        for tid, points in track_points.items():
            points.sort(key=lambda p: p[1])  # sort by timestamp

            # Split into contiguous runs using max_track_observation_gap_s
            runs: list[list[tuple[int, float, list[float], float]]] = []
            current_run: list[tuple[int, float, list[float], float]] = [points[0]]

            for j in range(1, len(points)):
                gap = points[j][1] - points[j - 1][1]
                if gap > self.triage_cfg.max_track_observation_gap_s:
                    runs.append(current_run)
                    current_run = [points[j]]
                else:
                    current_run.append(points[j])
            runs.append(current_run)

            for run in runs:
                timestamps = [p[1] for p in run]
                confs = [p[3] for p in run]
                n_obs = len(run)
                visible_dur = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0

                # Compute max observation gap within this run
                max_gap = 0.0
                for k in range(1, len(timestamps)):
                    max_gap = max(max_gap, timestamps[k] - timestamps[k - 1])

                # A run is stable if it meets all criteria
                is_stable = (
                    n_obs >= self.triage_cfg.minimum_observations
                    and visible_dur >= self.triage_cfg.minimum_visible_duration_s
                    and all(c >= self.triage_cfg.minimum_track_confidence for c in confs)
                )

                summary = TrackSummary(
                    clip_id=clip_id,
                    tracker_track_id=int(tid),
                    first_seen_s=min(timestamps),
                    last_seen_s=max(timestamps),
                    visible_duration_s=visible_dur,
                    n_observations=n_obs,
                    mean_confidence=sum(confs) / len(confs),
                    max_observation_gap_s=max_gap,
                    is_stable=is_stable,
                )
                summaries.append(summary)

        summaries.sort(key=lambda s: (s.tracker_track_id, s.first_seen_s))
        return summaries

    def _extract_clip_id(self) -> str:
        """Derive a clip_id from the video filename."""
        stem = self.video_path.stem
        return f"clip_{stem}"
