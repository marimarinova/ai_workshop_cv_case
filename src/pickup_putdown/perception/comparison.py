"""1 FPS vs 2 FPS comparison for triage acceptance testing."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pickup_putdown.config import TriageConfig
from pickup_putdown.perception.active_spans import compute_clip_summary, derive_active_spans
from pickup_putdown.perception.person_tracker import PersonTracker

logger = logging.getLogger(__name__)


@dataclass
class _ComparisonResult:
    clip_id: str
    decision_at_1_fps: str
    decision_at_2_fps: str
    stable_tracks_at_1_fps: int
    stable_tracks_at_2_fps: int
    active_spans_at_1_fps: int
    active_spans_at_2_fps: int
    runtime_at_1_fps: float
    runtime_at_2_fps: float
    decision_changed: bool


def run_fps_comparison(
    video_paths: list[Path],
    triage_cfg: TriageConfig,
    max_clips: int = 5,
) -> pd.DataFrame:
    """Run triage at 1 FPS and 2 FPS on a subset of videos and compare.

    Parameters
    ----------
    video_paths : list[Path]
        Sorted list of video paths to test.
    triage_cfg : TriageConfig
        Base triage config (target_fps will be overridden).
    max_clips : int
        Maximum number of clips to test.

    Returns
    -------
    pd.DataFrame
        Comparison results sorted by clip_id.
    """
    sorted_paths = sorted(video_paths)[:max_clips]
    results: list[_ComparisonResult] = []

    for vp in sorted_paths:
        logger.info("Comparing FPS for %s", vp)

        # Run at 1 FPS
        cfg_1 = _clone_cfg(triage_cfg, target_fps=1.0)
        start_1 = time.perf_counter()
        obs_1, summaries_1 = _run_tracker(vp, cfg_1)
        runtime_1 = time.perf_counter() - start_1

        spans_1 = derive_active_spans(
            obs_1,
            clip_id=obs_1[0].clip_id if obs_1 else f"clip_{vp.stem}",
            clip_duration_s=cfg_1.target_fps * 100,  # placeholder, corrected below
            merge_gap_s=cfg_1.merge_gap_s,
            effective_sample_fps=1.0,
        )
        summary_1 = compute_clip_summary(obs_1, spans_1)
        n_stable_1 = summary_1["n_person_tracks"]
        has_person_1 = summary_1["has_person"]

        # Run at 2 FPS
        cfg_2 = _clone_cfg(triage_cfg, target_fps=2.0)
        start_2 = time.perf_counter()
        obs_2, summaries_2 = _run_tracker(vp, cfg_2)
        runtime_2 = time.perf_counter() - start_2

        spans_2 = derive_active_spans(
            obs_2,
            clip_id=obs_2[0].clip_id if obs_2 else f"clip_{vp.stem}",
            clip_duration_s=cfg_2.target_fps * 100,
            merge_gap_s=cfg_2.merge_gap_s,
            effective_sample_fps=2.0,
        )
        summary_2 = compute_clip_summary(obs_2, spans_2)
        n_stable_2 = summary_2["n_person_tracks"]
        has_person_2 = summary_2["has_person"]

        result = _ComparisonResult(
            clip_id=f"clip_{vp.stem}",
            decision_at_1_fps="person" if has_person_1 else "no_person",
            decision_at_2_fps="person" if has_person_2 else "no_person",
            stable_tracks_at_1_fps=n_stable_1,
            stable_tracks_at_2_fps=n_stable_2,
            active_spans_at_1_fps=len(spans_1),
            active_spans_at_2_fps=len(spans_2),
            runtime_at_1_fps=round(runtime_1, 3),
            runtime_at_2_fps=round(runtime_2, 3),
            decision_changed=has_person_1 != has_person_2,
        )
        results.append(result)
        logger.info(
            "  %s: 1fps=%s(%d tracks) %.3fs, 2fps=%s(%d tracks) %.3fs, changed=%s",
            result.clip_id,
            result.decision_at_1_fps,
            result.stable_tracks_at_1_fps,
            result.runtime_at_1_fps,
            result.decision_at_2_fps,
            result.stable_tracks_at_2_fps,
            result.runtime_at_2_fps,
            result.decision_changed,
        )

    df = pd.DataFrame([vars(r) for r in results])
    df = df.sort_values("clip_id").reset_index(drop=True)
    return df


def _clone_cfg(cfg: TriageConfig, target_fps: float) -> TriageConfig:
    """Create a copy of TriageConfig with a different target_fps."""
    return TriageConfig(
        model_path=cfg.model_path,
        target_fps=target_fps,
        image_size=cfg.image_size,
        device=cfg.device,
        half=cfg.half,
        detector_confidence=cfg.detector_confidence,
        detector_iou_threshold=cfg.detector_iou_threshold,
        max_detections=cfg.max_detections,
        minimum_track_confidence=cfg.minimum_track_confidence,
        minimum_visible_duration_s=cfg.minimum_visible_duration_s,
        minimum_observations=cfg.minimum_observations,
        max_track_observation_gap_s=cfg.max_track_observation_gap_s,
        merge_gap_s=cfg.merge_gap_s,
        preview_sample_rate=cfg.preview_sample_rate,
        sampling_seed=cfg.sampling_seed,
        tracker_config=cfg.tracker_config,
    )


def _run_tracker(
    video_path: Path,
    cfg: TriageConfig,
) -> tuple[list, list]:
    """Run PersonTracker and return observations + summaries."""
    tracker = PersonTracker(video_path=video_path, triage_cfg=cfg)
    return tracker.run()
