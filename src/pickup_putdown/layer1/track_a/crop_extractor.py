"""Extract hand and shelf crops from video frames at sampled timestamps.

This module handles:
- Loading individual frames from video at specific timestamps
- Extracting hand crops centered on wrist coordinates
- Extracting shelf patches around the interaction area
- Clamping crops to stay within frame boundaries
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from pickup_putdown.common.schemas import Candidate, PoseObservation
    from pickup_putdown.config import TrackAFeaturesConfig
    from pickup_putdown.perception.shelf_regions import Polygon

from pickup_putdown.layer1.track_a.contracts import CropGeometry, CropRecord
from pickup_putdown.layer1.track_a.sampling import SamplePoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hand crop is roughly this fraction of the actor's bounding box height
HAND_CROP_BBOX_RATIO: float = 0.4

# Minimum and maximum crop size in pixels (to keep crops reasonable)
MIN_CROP_SIZE: int = 64
MAX_CROP_SIZE: int = 448

# Maximum time tolerance when finding nearest pose observation
DEFAULT_POSE_TOLERANCE_S: float = 0.2


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------


def load_frame_at_timestamp(
    video_path: Path | str,
    timestamp_s: float,
) -> np.ndarray | None:
    """Load a single frame from video at the specified timestamp.

    Args:
        video_path: Path to the video file.
        timestamp_s: Timestamp in seconds.

    Returns:
        Frame as numpy array (H, W, 3) in BGR format, or None if failed.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None

    try:
        # Seek to timestamp
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000)
        ret, frame = cap.read()

        if not ret or frame is None:
            logger.warning(f"Failed to read frame at {timestamp_s}s from {video_path}")
            return None

        return frame

    finally:
        cap.release()


def get_video_dimensions(video_path: Path | str) -> tuple[int, int] | None:
    """Get video frame dimensions (width, height).

    Args:
        video_path: Path to the video file.

    Returns:
        Tuple of (width, height) or None if failed.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Nearest pose observation
# ---------------------------------------------------------------------------


def find_nearest_pose_observation(
    timestamp_s: float,
    pose_observations: list[PoseObservation],
    max_tolerance_s: float = DEFAULT_POSE_TOLERANCE_S,
) -> PoseObservation | None:
    """Find the pose observation nearest to the given timestamp.

    Args:
        timestamp_s: Target timestamp in seconds.
        pose_observations: List of pose observations (should be pre-filtered
            for the relevant clip/actor/hand).
        max_tolerance_s: Maximum allowed time difference.

    Returns:
        Nearest PoseObservation within tolerance, or None if none found.
    """
    if not pose_observations:
        return None

    nearest = None
    min_diff = float("inf")

    for obs in pose_observations:
        diff = abs(obs.timestamp_s - timestamp_s)
        if diff < min_diff:
            min_diff = diff
            nearest = obs

    if nearest is not None and min_diff <= max_tolerance_s:
        return nearest

    return None


# ---------------------------------------------------------------------------
# Crop scale computation
# ---------------------------------------------------------------------------


def compute_crop_scale(
    pose_obs: PoseObservation,
    config: TrackAFeaturesConfig,
) -> int:
    """Compute the crop size based on actor bounding box.

    Uses the actor's bounding box height to scale the crop appropriately.
    Falls back to config.hand_crop_size if bbox not available.

    Args:
        pose_obs: Pose observation with optional bbox info.
        config: Track A configuration.

    Returns:
        Crop size in pixels (square crop).
    """
    # Try to use actor bbox for scale
    bbox_x1 = getattr(pose_obs, "person_bbox_x1", None)
    bbox_y1 = getattr(pose_obs, "person_bbox_y1", None)
    bbox_x2 = getattr(pose_obs, "person_bbox_x2", None)
    bbox_y2 = getattr(pose_obs, "person_bbox_y2", None)

    if all(v is not None for v in [bbox_x1, bbox_y1, bbox_x2, bbox_y2]):
        bbox_height = bbox_y2 - bbox_y1
        scale = int(bbox_height * HAND_CROP_BBOX_RATIO)
        scale = max(MIN_CROP_SIZE, min(scale, MAX_CROP_SIZE))
        return scale

    # Fallback to config default
    return config.hand_crop_size


# ---------------------------------------------------------------------------
# Crop extraction
# ---------------------------------------------------------------------------


def extract_hand_crop(
    frame: np.ndarray,
    wrist_x: float,
    wrist_y: float,
    crop_size: int,
) -> tuple[np.ndarray, CropGeometry]:
    """Extract a square crop centered on the wrist position.

    If the crop would extend outside the frame, it is shifted to stay
    within bounds (Option A clamping).

    Args:
        frame: Video frame (H, W, 3).
        wrist_x: Wrist x coordinate in pixels.
        wrist_y: Wrist y coordinate in pixels.
        crop_size: Size of the square crop.

    Returns:
        Tuple of (crop array, CropGeometry).
    """
    frame_h, frame_w = frame.shape[:2]
    half_size = crop_size // 2

    # Calculate initial crop bounds centered on wrist
    x1 = int(wrist_x) - half_size
    y1 = int(wrist_y) - half_size
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    # Shift to stay within frame (Option A clamping)
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > frame_w:
        x1 -= (x2 - frame_w)
        x2 = frame_w
    if y2 > frame_h:
        y1 -= (y2 - frame_h)
        y2 = frame_h

    # Final clamp (in case frame is smaller than crop_size)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w, x2)
    y2 = min(frame_h, y2)

    # Extract crop
    crop = frame[y1:y2, x1:x2].copy()

    # Resize to target size if needed (when frame smaller than crop_size)
    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        crop = cv2.resize(crop, (crop_size, crop_size))

    geometry = CropGeometry(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

    return crop, geometry


def extract_shelf_patch(
    frame: np.ndarray,
    shelf_region: Polygon,
    contact_point: tuple[float, float] | None,
    patch_size: int,
) -> tuple[np.ndarray, CropGeometry]:
    """Extract a square patch from the shelf region.

    Centers on the contact point if provided, otherwise uses the
    centroid of the shelf region polygon.

    Args:
        frame: Video frame (H, W, 3).
        shelf_region: Polygon defining the shelf region.
        contact_point: Optional (x, y) of wrist contact with shelf.
        patch_size: Size of the square patch.

    Returns:
        Tuple of (patch array, CropGeometry).
    """
    # Determine center point
    if contact_point is not None:
        center_x, center_y = contact_point
    else:
        # Use polygon centroid
        center_x = sum(p[0] for p in shelf_region) / len(shelf_region)
        center_y = sum(p[1] for p in shelf_region) / len(shelf_region)

    # Use same extraction logic as hand crop
    return extract_hand_crop(frame, center_x, center_y, patch_size)


# ---------------------------------------------------------------------------
# Crop ID generation
# ---------------------------------------------------------------------------


def generate_crop_id(
    clip_id: str,
    candidate_id: str,
    timestamp_s: float,
    crop_type: str,
) -> str:
    """Generate a unique crop ID.

    Args:
        clip_id: Clip identifier.
        candidate_id: Candidate identifier.
        timestamp_s: Timestamp in seconds.
        crop_type: "hand" or "shelf".

    Returns:
        Unique crop ID string.
    """
    payload = f"{clip_id}:{candidate_id}:{timestamp_s:.6f}:{crop_type}"
    hash_suffix = hashlib.sha256(payload.encode()).hexdigest()[:8]
    return f"crop_{crop_type}_{hash_suffix}"


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------


def extract_crops_for_candidate(
    video_path: Path | str,
    candidate: Candidate,
    sample_points: list[SamplePoint],
    pose_observations: list[PoseObservation],
    shelf_region: Polygon,
    config: TrackAFeaturesConfig,
) -> list[CropRecord]:
    """Extract all crops for a candidate at the sampled timestamps.

    For each sample point, extracts both a hand crop and a shelf patch.

    Args:
        video_path: Path to the source video.
        candidate: The candidate interval.
        sample_points: List of sample points (from sampling.py).
        pose_observations: Pose observations for this candidate's actor/hand.
        shelf_region: Polygon defining the shelf region.
        config: Track A configuration.

    Returns:
        List of CropRecords (2 per sample point: hand + shelf).
    """
    video_path = Path(video_path)
    crops: list[CropRecord] = []

    for sample in sample_points:
        # Find nearest pose observation for this timestamp
        nearest_pose = find_nearest_pose_observation(
            sample.timestamp_s,
            pose_observations,
            max_tolerance_s=0.2,
        )

        if nearest_pose is None:
            logger.warning(
                f"No pose observation near {sample.timestamp_s}s for candidate "
                f"{candidate.candidate_id}, skipping this sample point"
            )
            continue

        # Load frame at this timestamp
        frame = load_frame_at_timestamp(video_path, sample.timestamp_s)
        if frame is None:
            logger.warning(
                f"Failed to load frame at {sample.timestamp_s}s for candidate "
                f"{candidate.candidate_id}, skipping this sample point"
            )
            continue

        # Compute crop scale from actor bbox
        crop_scale = compute_crop_scale(nearest_pose, config)

        # Extract hand crop
        hand_crop, hand_geom = extract_hand_crop(
            frame,
            nearest_pose.wrist_x,
            nearest_pose.wrist_y,
            crop_scale,
        )

        hand_crop_id = generate_crop_id(
            candidate.clip_id,
            candidate.candidate_id,
            sample.timestamp_s,
            "hand",
        )

        crops.append(
            CropRecord(
                crop_id=hand_crop_id,
                clip_id=candidate.clip_id,
                candidate_id=candidate.candidate_id,
                timestamp_s=sample.timestamp_s,
                sample_position=sample.position,
                crop_type="hand",
                geometry=hand_geom,
                actor_id=candidate.actor_id,
                hand_side=candidate.hand_side,
                region_id=candidate.region_id,
            )
        )

        # Extract shelf patch (centered on wrist position as contact point)
        contact_point = (nearest_pose.wrist_x, nearest_pose.wrist_y)
        shelf_crop, shelf_geom = extract_shelf_patch(
            frame,
            shelf_region,
            contact_point,
            config.shelf_patch_size,
        )

        shelf_crop_id = generate_crop_id(
            candidate.clip_id,
            candidate.candidate_id,
            sample.timestamp_s,
            "shelf",
        )

        crops.append(
            CropRecord(
                crop_id=shelf_crop_id,
                clip_id=candidate.clip_id,
                candidate_id=candidate.candidate_id,
                timestamp_s=sample.timestamp_s,
                sample_position=sample.position,
                crop_type="shelf",
                geometry=shelf_geom,
                actor_id=candidate.actor_id,
                hand_side=candidate.hand_side,
                region_id=candidate.region_id,
            )
        )

    return crops


def save_crop_image(
    crop: np.ndarray,
    output_path: Path | str,
) -> Path:
    """Save a crop image to disk.

    Args:
        crop: Crop array (H, W, 3) in BGR format.
        output_path: Destination path.

    Returns:
        Path where the image was saved.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), crop)
    return output_path