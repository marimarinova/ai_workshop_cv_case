"""Track B1 inference: sliding window prediction and event decoding.

This module provides:
- Sliding window inference over candidate intervals
- Temporal smoothing of prediction scores
- Peak detection for event localization
- Same-type merging (never merge pickup + putdown)
- Canonical predictions.csv output

Inference pipeline:
    1. Generate sliding windows over candidate (reuses dataset.py logic)
    2. Run model on each window → class probabilities
    3. Smooth scores temporally
    4. Detect peaks above threshold
    5. Merge same-type overlapping detections
    6. Output canonical event predictions
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from pickup_putdown.layer1.track_b1.dataset import (
    LABEL_BACKGROUND,
    LABEL_NAMES,
    LABEL_PICKUP,
    LABEL_PUTDOWN,
    InferenceWindow,
    WindowConfig,
    apply_crop_and_resize,
    compute_actor_crop_box,
    decode_window_frames,
    generate_inference_windows_for_candidate,
    normalize_frames,
)
from pickup_putdown.layer1.track_b1.videomae_classifier import (
    VideoMAEClassifier,
    load_checkpoint,
)

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass
class InferenceConfig:
    """Inference configuration."""

    # Window parameters (must match training)
    window_duration_s: float = 2.5
    window_stride_s: float = 0.5
    num_frames: int = 16
    image_size: tuple[int, int] = (224, 224)
    crop_margin: float = 0.15

    # Thresholds (tune on validation set)
    pickup_threshold: float = 0.5
    putdown_threshold: float = 0.5

    # Temporal smoothing
    smoothing_window: int = 3  # Number of adjacent predictions to average (must be odd)

    # Merging
    same_type_merge_gap_s: float = 0.75  # Merge same-type if gap < this
    min_event_duration_s: float = 0.3  # Discard events shorter than this

    # Output
    model_name: str = "layer1_track_b1_videomae_window_v1"

    # Processing
    batch_size: int = 8

    def to_window_config(self) -> WindowConfig:
        """Convert to WindowConfig for window generation."""
        return WindowConfig(
            window_duration_s=self.window_duration_s,
            window_stride_s=self.window_stride_s,
            num_frames=self.num_frames,
            image_size=self.image_size,
            crop_margin=self.crop_margin,
        )


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass
class WindowPrediction:
    """Prediction for a single window."""

    window_start_s: float
    window_end_s: float
    window_center_s: float
    probs: np.ndarray  # [3] probabilities: [bg, pickup, putdown]
    predicted_class: int
    confidence: float

    @property
    def background_prob(self) -> float:
        return float(self.probs[LABEL_BACKGROUND])

    @property
    def pickup_prob(self) -> float:
        return float(self.probs[LABEL_PICKUP])

    @property
    def putdown_prob(self) -> float:
        return float(self.probs[LABEL_PUTDOWN])


@dataclass
class ScoreRegion:
    """A contiguous region where score exceeds threshold."""

    event_type: str  # "pickup" or "putdown"
    start_s: float
    end_s: float
    peak_score: float
    mean_score: float


@dataclass
class EventPrediction:
    """A detected event in canonical format."""

    pred_id: str
    clip_id: str
    candidate_id: str
    actor_id: str
    event_type: str  # "pickup" or "putdown"
    t_start: float
    t_end: float
    score: float
    model: str

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame/CSV export."""
        return {
            "pred_id": self.pred_id,
            "clip_id": self.clip_id,
            "type": self.event_type,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "score": self.score,
            "model": self.model,
            # Internal fields (not in canonical schema but useful)
            "candidate_id": self.candidate_id,
            "actor_id": self.actor_id,
        }


# ============================================================
# MODEL INFERENCE
# ============================================================


@torch.no_grad()
def predict_windows(
    model: VideoMAEClassifier,
    windows: list[InferenceWindow],
    video_path: Path,
    pose_track_df: pd.DataFrame,
    shelf_region: Optional[dict],
    config: InferenceConfig,
    device: torch.device,
) -> list[WindowPrediction]:
    """Run model inference on all windows for a candidate.

    Args:
        model: Trained VideoMAE classifier.
        windows: List of InferenceWindow objects.
        video_path: Path to video file.
        pose_track_df: Pose track DataFrame for this actor.
        shelf_region: Shelf region config dict.
        config: Inference configuration.
        device: Device to run inference on.

    Returns:
        List of WindowPrediction for each window.
    """
    if not windows:
        return []

    model.eval()
    predictions: list[WindowPrediction] = []

    # Process in batches
    for batch_start in range(0, len(windows), config.batch_size):
        batch_end = min(batch_start + config.batch_size, len(windows))
        batch_windows = windows[batch_start:batch_end]

        # Prepare batch tensor
        batch_tensor = _prepare_window_batch(
            windows=batch_windows,
            video_path=video_path,
            pose_track_df=pose_track_df,
            shelf_region=shelf_region,
            config=config,
        )

        if batch_tensor is None:
            # Failed to prepare batch, skip
            logger.warning(f"Failed to prepare batch {batch_start}-{batch_end}")
            continue

        # Move to device and predict
        batch_tensor = batch_tensor.to(device)
        logits = model(batch_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # Create WindowPrediction for each
        for i, window in enumerate(batch_windows):
            window_probs = probs[i]
            predicted_class = int(np.argmax(window_probs))
            confidence = float(window_probs[predicted_class])

            predictions.append(
                WindowPrediction(
                    window_start_s=window.window_start_s,
                    window_end_s=window.window_end_s,
                    window_center_s=window.window_center_s,
                    probs=window_probs,
                    predicted_class=predicted_class,
                    confidence=confidence,
                )
            )

    return predictions


def _prepare_window_batch(
    windows: list[InferenceWindow],
    video_path: Path,
    pose_track_df: pd.DataFrame,
    shelf_region: Optional[dict],
    config: InferenceConfig,
) -> Optional[torch.Tensor]:
    """Prepare batch of windows for model input.

    Args:
        windows: List of InferenceWindow objects.
        video_path: Path to video file.
        pose_track_df: Pose track for actor (for cropping).
        shelf_region: Shelf region config.
        config: Inference configuration.

    Returns:
        Tensor of shape [batch_size, num_frames, channels, height, width],
        or None if preparation failed.
    """
    batch_frames: list[torch.Tensor] = []

    for window in windows:
        # Decode frames for this window
        frames = decode_window_frames(
            video_path=video_path,
            start_s=window.window_start_s,
            end_s=window.window_end_s,
            num_frames=config.num_frames,
        )

        if frames is None:
            # Use zero tensor as fallback
            logger.warning(
                f"Failed to decode frames for window "
                f"[{window.window_start_s:.2f}-{window.window_end_s:.2f}]"
            )
            frames = np.zeros(
                (config.num_frames, config.image_size[1], config.image_size[0], 3),
                dtype=np.uint8,
            )

        # Get frame dimensions
        frame_h, frame_w = frames.shape[1:3]

        # Compute actor-conditioned crop box
        crop_box = compute_actor_crop_box(
            pose_track_df=pose_track_df,
            shelf_region=shelf_region,
            start_s=window.window_start_s,
            end_s=window.window_end_s,
            margin=config.crop_margin,
            frame_size=(frame_w, frame_h),
        )

        # Apply crop and resize
        frames = apply_crop_and_resize(
            frames=frames,
            crop_box=crop_box,
            target_size=config.image_size,
        )

        # Normalize to tensor
        tensor = normalize_frames(frames)
        batch_frames.append(tensor)

    if not batch_frames:
        return None

    # Stack into batch: [batch_size, num_frames, channels, height, width]
    return torch.stack(batch_frames, dim=0)


# ============================================================
# TEMPORAL SMOOTHING
# ============================================================


def smooth_predictions(
    predictions: list[WindowPrediction],
    window_size: int = 3,
) -> list[WindowPrediction]:
    """Apply temporal smoothing to predictions.

    Averages probabilities over adjacent windows to reduce noise.
    Uses a simple moving average.

    Args:
        predictions: List of window predictions (must be time-ordered).
        window_size: Number of neighbors to average (must be odd).

    Returns:
        New list with smoothed probabilities.
    """
    if len(predictions) <= 1:
        return predictions

    if window_size < 1:
        return predictions

    # Ensure odd window size
    if window_size % 2 == 0:
        window_size += 1

    half_window = window_size // 2
    smoothed: list[WindowPrediction] = []

    for i, pred in enumerate(predictions):
        # Get neighbors
        start_idx = max(0, i - half_window)
        end_idx = min(len(predictions), i + half_window + 1)

        # Average probabilities
        neighbor_probs = [predictions[j].probs for j in range(start_idx, end_idx)]
        avg_probs = np.mean(neighbor_probs, axis=0)

        # Create smoothed prediction
        predicted_class = int(np.argmax(avg_probs))
        confidence = float(avg_probs[predicted_class])

        smoothed.append(
            WindowPrediction(
                window_start_s=pred.window_start_s,
                window_end_s=pred.window_end_s,
                window_center_s=pred.window_center_s,
                probs=avg_probs,
                predicted_class=predicted_class,
                confidence=confidence,
            )
        )

    return smoothed


# ============================================================
# PEAK DETECTION
# ============================================================


def detect_score_peaks(
    predictions: list[WindowPrediction],
    config: InferenceConfig,
) -> list[ScoreRegion]:
    """Find regions where class probability exceeds threshold.

    Processes pickup and putdown separately with their own thresholds.

    Args:
        predictions: Smoothed window predictions.
        config: Configuration with thresholds.

    Returns:
        List of ScoreRegion for detected events.
    """
    regions: list[ScoreRegion] = []

    # Detect pickup regions
    pickup_regions = _find_regions_above_threshold(
        predictions=predictions,
        class_idx=LABEL_PICKUP,
        threshold=config.pickup_threshold,
        event_type="pickup",
    )
    regions.extend(pickup_regions)

    # Detect putdown regions
    putdown_regions = _find_regions_above_threshold(
        predictions=predictions,
        class_idx=LABEL_PUTDOWN,
        threshold=config.putdown_threshold,
        event_type="putdown",
    )
    regions.extend(putdown_regions)

    return regions


def _find_regions_above_threshold(
    predictions: list[WindowPrediction],
    class_idx: int,
    threshold: float,
    event_type: str,
) -> list[ScoreRegion]:
    """Find contiguous regions where class score > threshold.

    Args:
        predictions: Window predictions (time-ordered).
        class_idx: Class index to check (1=pickup, 2=putdown).
        threshold: Score threshold.
        event_type: Event type string for output.

    Returns:
        List of detected ScoreRegion.
    """
    regions: list[ScoreRegion] = []

    if not predictions:
        return regions

    # Track current region
    in_region = False
    region_start = 0.0
    region_end = 0.0
    region_scores: list[float] = []

    for pred in predictions:
        score = float(pred.probs[class_idx])

        if score > threshold:
            if not in_region:
                # Start new region
                in_region = True
                region_start = pred.window_start_s
                region_scores = [score]
            else:
                # Continue region
                region_scores.append(score)
            region_end = pred.window_end_s
        else:
            if in_region:
                # End region
                regions.append(
                    ScoreRegion(
                        event_type=event_type,
                        start_s=region_start,
                        end_s=region_end,
                        peak_score=max(region_scores),
                        mean_score=np.mean(region_scores),
                    )
                )
                in_region = False
                region_scores = []

    # Handle region at end
    if in_region and region_scores:
        regions.append(
            ScoreRegion(
                event_type=event_type,
                start_s=region_start,
                end_s=region_end,
                peak_score=max(region_scores),
                mean_score=np.mean(region_scores),
            )
        )

    return regions


# ============================================================
# SAME-TYPE MERGING
# ============================================================


def merge_same_type_regions(
    regions: list[ScoreRegion],
    merge_gap_s: float,
    min_duration_s: float,
) -> list[ScoreRegion]:
    """Merge overlapping/adjacent regions of the SAME type.

    CRITICAL: Never merges pickup with putdown - they are different events!

    Args:
        regions: Detected score regions.
        merge_gap_s: Merge if gap between same-type regions < this.
        min_duration_s: Discard regions shorter than this.

    Returns:
        Merged and filtered regions.
    """
    if not regions:
        return []

    merged: list[ScoreRegion] = []

    # Process pickup and putdown separately
    pickup_regions = [r for r in regions if r.event_type == "pickup"]
    putdown_regions = [r for r in regions if r.event_type == "putdown"]

    merged.extend(_merge_regions_of_type(pickup_regions, merge_gap_s))
    merged.extend(_merge_regions_of_type(putdown_regions, merge_gap_s))

    # Filter by minimum duration
    filtered = [r for r in merged if (r.end_s - r.start_s) >= min_duration_s]

    # Sort by start time
    filtered.sort(key=lambda r: r.start_s)

    return filtered


def _merge_regions_of_type(
    regions: list[ScoreRegion],
    merge_gap_s: float,
) -> list[ScoreRegion]:
    """Merge overlapping/adjacent regions of the same type.

    Args:
        regions: Regions of a single type.
        merge_gap_s: Maximum gap to merge.

    Returns:
        Merged regions.
    """
    if not regions:
        return []

    # Sort by start time
    sorted_regions = sorted(regions, key=lambda r: r.start_s)

    merged: list[ScoreRegion] = []
    current = sorted_regions[0]

    for next_region in sorted_regions[1:]:
        gap = next_region.start_s - current.end_s

        if gap <= merge_gap_s:
            # Merge: extend current region
            current = ScoreRegion(
                event_type=current.event_type,
                start_s=current.start_s,
                end_s=max(current.end_s, next_region.end_s),
                peak_score=max(current.peak_score, next_region.peak_score),
                mean_score=(current.mean_score + next_region.mean_score) / 2,
            )
        else:
            # Gap too large: save current and start new
            merged.append(current)
            current = next_region

    # Don't forget last region
    merged.append(current)

    return merged


# ============================================================
# EVENT CREATION
# ============================================================


def create_event_predictions(
    regions: list[ScoreRegion],
    clip_id: str,
    candidate_id: str,
    actor_id: str,
    config: InferenceConfig,
) -> list[EventPrediction]:
    """Convert score regions to canonical event predictions.

    Args:
        regions: Merged score regions.
        clip_id: Clip identifier.
        candidate_id: Candidate identifier.
        actor_id: Actor identifier.
        config: Configuration with model name.

    Returns:
        List of EventPrediction in canonical format.
    """
    predictions: list[EventPrediction] = []

    for i, region in enumerate(regions):
        # Generate unique prediction ID
        pred_id = _generate_pred_id(clip_id, candidate_id, region, i)

        predictions.append(
            EventPrediction(
                pred_id=pred_id,
                clip_id=clip_id,
                candidate_id=candidate_id,
                actor_id=actor_id,
                event_type=region.event_type,
                t_start=region.start_s,
                t_end=region.end_s,
                score=region.peak_score,
                model=config.model_name,
            )
        )

    return predictions


def _generate_pred_id(
    clip_id: str,
    candidate_id: str,
    region: ScoreRegion,
    index: int,
) -> str:
    """Generate unique prediction ID."""
    payload = f"{clip_id}:{candidate_id}:{region.event_type}:{region.start_s:.3f}:{index}"
    hash_suffix = hashlib.sha256(payload.encode()).hexdigest()[:8]
    return f"b1_{hash_suffix}"


# ============================================================
# SINGLE CANDIDATE INFERENCE
# ============================================================


def infer_candidate(
    model: VideoMAEClassifier,
    candidate: pd.Series,
    video_path: Path,
    pose_track_df: pd.DataFrame,
    shelf_region: Optional[dict],
    config: InferenceConfig,
    device: torch.device,
) -> list[EventPrediction]:
    """Run full inference pipeline on one candidate.

    This is the main entry point for processing a single candidate.
    It orchestrates: window generation → prediction → smoothing →
    peak detection → merging → event creation.

    Args:
        model: Trained VideoMAE classifier.
        candidate: Candidate row from DataFrame.
        video_path: Path to video file.
        pose_track_df: Pose track for this actor.
        shelf_region: Shelf region config.
        config: Inference configuration.
        device: Device to run on.

    Returns:
        List of detected events (may be empty, one, or multiple).
    """
    clip_id = candidate["clip_id"]
    candidate_id = candidate["candidate_id"]
    actor_id = candidate["actor_id"]

    # 1. Generate windows (reuses dataset.py logic!)
    window_config = config.to_window_config()
    windows = generate_inference_windows_for_candidate(candidate, window_config)

    if not windows:
        logger.debug(f"No windows generated for candidate {candidate_id}")
        return []

    logger.debug(f"Generated {len(windows)} windows for candidate {candidate_id}")

    # 2. Run model predictions
    predictions = predict_windows(
        model=model,
        windows=windows,
        video_path=video_path,
        pose_track_df=pose_track_df,
        shelf_region=shelf_region,
        config=config,
        device=device,
    )

    if not predictions:
        return []

    # 3. Temporal smoothing
    smoothed = smooth_predictions(predictions, config.smoothing_window)

    # 4. Peak detection
    regions = detect_score_peaks(smoothed, config)

    # 5. Same-type merging
    merged = merge_same_type_regions(
        regions,
        config.same_type_merge_gap_s,
        config.min_event_duration_s,
    )

    # 6. Create event predictions
    events = create_event_predictions(
        merged,
        clip_id,
        candidate_id,
        actor_id,
        config,
    )

    logger.debug(
        f"Candidate {candidate_id}: {len(windows)} windows → "
        f"{len(regions)} peaks → {len(merged)} merged → {len(events)} events"
    )

    return events


# ============================================================
# BATCH INFERENCE
# ============================================================


def infer_all_candidates(
    model: VideoMAEClassifier,
    candidates_df: pd.DataFrame,
    video_dir: Path,
    pose_tracks_dir: Path,
    shelf_regions: dict[str, dict],
    clips_df: pd.DataFrame,
    config: InferenceConfig,
    device: torch.device,
) -> pd.DataFrame:
    """Run inference on all candidates.

    Args:
        model: Trained VideoMAE classifier.
        candidates_df: All candidates to process.
        video_dir: Directory with video files.
        pose_tracks_dir: Directory with pose tracks.
        shelf_regions: Dict mapping region_id to region config.
        clips_df: Clip metadata.
        config: Inference configuration.
        device: Device to run on.

    Returns:
        DataFrame with all predictions in canonical format.
    """
    logger.info(f"Running inference on {len(candidates_df)} candidates")

    all_predictions: list[dict] = []
    pose_cache: dict[str, pd.DataFrame] = {}

    for idx, candidate in candidates_df.iterrows():
        clip_id = candidate["clip_id"]
        actor_id = candidate["actor_id"]
        region_id = candidate.get("region_id")

        # Get video path
        video_path = _get_video_path(clip_id, video_dir, clips_df)
        if not video_path.exists():
            logger.warning(f"Video not found for clip {clip_id}: {video_path}")
            continue

        # Load pose track (with caching)
        cache_key = f"{clip_id}_{actor_id}"
        if cache_key not in pose_cache:
            pose_cache[cache_key] = _load_pose_track(clip_id, actor_id, pose_tracks_dir)
        pose_track_df = pose_cache[cache_key]

        # Get shelf region
        shelf_region = shelf_regions.get(region_id) if region_id else None

        # Run inference on this candidate
        events = infer_candidate(
            model=model,
            candidate=candidate,
            video_path=video_path,
            pose_track_df=pose_track_df,
            shelf_region=shelf_region,
            config=config,
            device=device,
        )

        # Collect predictions
        for event in events:
            all_predictions.append(event.to_dict())

        # Progress logging
        if (idx + 1) % 10 == 0:
            logger.info(f"Processed {idx + 1}/{len(candidates_df)} candidates")

    # Create DataFrame
    if all_predictions:
        predictions_df = pd.DataFrame(all_predictions)
        logger.info(
            f"Inference complete: {len(predictions_df)} predictions from "
            f"{len(candidates_df)} candidates"
        )
    else:
        predictions_df = pd.DataFrame(
            columns=["pred_id", "clip_id", "type", "t_start", "t_end", "score", "model"]
        )
        logger.info("Inference complete: no events detected")

    return predictions_df


def _get_video_path(clip_id: str, video_dir: Path, clips_df: pd.DataFrame) -> Path:
    """Resolve video path from clip_id."""
    # Try to get path from clips_df
    if "s3_key" in clips_df.columns:
        matches = clips_df[clips_df["clip_id"] == clip_id]
        if not matches.empty:
            s3_key = matches.iloc[0]["s3_key"]
            filename = Path(s3_key).name
            video_path = video_dir / filename
            if video_path.exists():
                return video_path

    # Fallback: try common patterns
    for ext in [".mp4", ".avi", ".mov", ".mkv"]:
        video_path = video_dir / f"{clip_id}{ext}"
        if video_path.exists():
            return video_path

    # Last resort
    return video_dir / f"{clip_id}.mp4"


def _load_pose_track(clip_id: str, actor_id: str, pose_tracks_dir: Path) -> pd.DataFrame:
    """Load pose track for actor."""
    pose_file = pose_tracks_dir / f"{clip_id}.parquet"

    if not pose_file.exists():
        logger.warning(f"Pose track not found: {pose_file}")
        return pd.DataFrame()

    try:
        pose_df = pd.read_parquet(pose_file)
        if "actor_id" in pose_df.columns:
            pose_df = pose_df[pose_df["actor_id"] == actor_id]
        return pose_df
    except Exception as e:
        logger.warning(f"Failed to load pose track {pose_file}: {e}")
        return pd.DataFrame()


# ============================================================
# OUTPUT
# ============================================================


def save_predictions(
    predictions_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Save predictions to canonical CSV format.

    Args:
        predictions_df: DataFrame with predictions.
        output_path: Output file path.

    Returns:
        Path where file was saved.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Select canonical columns (in order)
    canonical_columns = ["pred_id", "clip_id", "type", "t_start", "t_end", "score", "model"]
    available_columns = [col for col in canonical_columns if col in predictions_df.columns]

    predictions_df[available_columns].to_csv(output_path, index=False)
    logger.info(f"Saved {len(predictions_df)} predictions to {output_path}")

    return output_path


# ============================================================
# CLI ENTRY POINT
# ============================================================


def main(
    checkpoint_path: str,
    candidates_path: str,
    clips_path: str,
    video_dir: str,
    pose_tracks_dir: str,
    shelf_regions_path: str,
    output_path: str,
    config_path: Optional[str] = None,
    pickup_threshold: float = 0.5,
    putdown_threshold: float = 0.5,
) -> None:
    """CLI entry point for inference.

    Args:
        checkpoint_path: Path to trained model checkpoint.
        candidates_path: Path to candidates.parquet.
        clips_path: Path to clips.csv.
        video_dir: Directory with video files.
        pose_tracks_dir: Directory with pose tracks.
        shelf_regions_path: Path to shelf regions YAML.
        output_path: Output predictions CSV path.
        config_path: Optional config YAML path.
        pickup_threshold: Threshold for pickup detection.
        putdown_threshold: Threshold for putdown detection.
    """
    import yaml

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("=" * 60)
    logger.info("Track B1 Inference")
    logger.info("=" * 60)

    # Load config
    config = InferenceConfig(
        pickup_threshold=pickup_threshold,
        putdown_threshold=putdown_threshold,
    )

    if config_path is not None:
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    # Load data
    logger.info("Loading data...")
    candidates_df = pd.read_parquet(candidates_path)
    clips_df = pd.read_csv(clips_path)

    # Load shelf regions
    with open(shelf_regions_path) as f:
        shelf_regions_config = yaml.safe_load(f)
    shelf_regions = {r["region_id"]: r for r in shelf_regions_config.get("regions", [])}

    # Load model
    logger.info(f"Loading model from {checkpoint_path}")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    checkpoint_data = load_checkpoint(checkpoint_path, device=str(device))
    model = checkpoint_data["model"]
    model.eval()

    logger.info(f"Model loaded, device={device}")
    logger.info(f"Thresholds: pickup={config.pickup_threshold}, putdown={config.putdown_threshold}")

    # Run inference
    predictions_df = infer_all_candidates(
        model=model,
        candidates_df=candidates_df,
        video_dir=Path(video_dir),
        pose_tracks_dir=Path(pose_tracks_dir),
        shelf_regions=shelf_regions,
        clips_df=clips_df,
        config=config,
        device=device,
    )

    # Save predictions
    save_predictions(predictions_df, Path(output_path))

    logger.info("=" * 60)
    logger.info("Inference complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    import typer

    typer.run(main)
