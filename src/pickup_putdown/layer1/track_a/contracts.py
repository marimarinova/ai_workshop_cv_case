"""Data contracts for Track A feature extraction pipeline.

These dataclasses define the shape of data flowing through Task 9:
candidates → sample points → crops → embeddings → feature records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# Crop geometry and records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CropGeometry:
    """Pixel coordinates of a crop within the source frame."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Crop dimensions must be positive: {self.width}x{self.height}")
        if self.x < 0 or self.y < 0:
            raise ValueError(f"Crop coordinates must be non-negative: ({self.x}, {self.y})")


VALID_SAMPLE_POSITIONS: tuple[str, ...] = ("pre", "mid", "contact", "post")


@dataclass
class CropRecord:
    """One extracted image crop from a video frame.

    A crop can exist independently of its embedding (for QA inspection
    or if encoding fails).
    """

    crop_id: str
    clip_id: str
    candidate_id: str
    timestamp_s: float
    sample_position: str  # "pre", "mid", "contact", "post" — required
    crop_type: Literal["hand", "shelf"]
    geometry: CropGeometry
    crop_path: Path | None = None  # Path to saved crop image (if saved)

    # Optional metadata
    actor_id: str | None = None
    hand_side: str | None = None
    region_id: str | None = None

    def __post_init__(self) -> None:
        if self.timestamp_s < 0:
            raise ValueError(f"timestamp_s must be non-negative: {self.timestamp_s}")
        if self.sample_position not in VALID_SAMPLE_POSITIONS:
            raise ValueError(
                f"sample_position must be one of {VALID_SAMPLE_POSITIONS}: {self.sample_position}"
            )
        if self.crop_type not in ("hand", "shelf"):
            raise ValueError(f"crop_type must be 'hand' or 'shelf': {self.crop_type}")


# ---------------------------------------------------------------------------
# Embedding records
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingRecord:
    """One embedding vector extracted from a crop.

    Tied to a specific encoder name and version for cache invalidation.
    """

    crop_id: str
    embedding_path: Path  # Path to saved .npy file
    encoder_name: str
    encoder_version: str
    embedding_dim: int

    def __post_init__(self) -> None:
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive: {self.embedding_dim}")


# ---------------------------------------------------------------------------
# Feature records (training-ready)
# ---------------------------------------------------------------------------


VALID_LABELS: tuple[str, ...] = ("pickup", "putdown", "negative")
VALID_SPLITS: tuple[str, ...] = ("train", "val", "test")


@dataclass
class FeatureRecord:
    """Complete feature record ready for classifier training.

    Combines crop info, embedding reference, label, and split assignment.
    """

    # Identifiers
    crop_id: str
    clip_id: str
    candidate_id: str

    # Temporal info
    timestamp_s: float
    sample_position: str  # "pre", "mid", "contact", "post"

    # Crop info
    crop_type: Literal["hand", "shelf"]
    geometry: CropGeometry

    # Embedding reference
    embedding_path: Path
    encoder_name: str
    encoder_version: str

    # Label and split
    label: str  # "pickup", "putdown", "negative"
    split: str  # "train", "val", "test"

    # Optional metadata
    actor_id: str | None = None
    hand_side: str | None = None
    region_id: str | None = None
    confidence: str | None = None  # from ground truth: "high", "med", "low"
    hard_case: bool = False
    event_id: str | None = None  # reference to ground truth event if positive

    def __post_init__(self) -> None:
        if self.timestamp_s < 0:
            raise ValueError(f"timestamp_s must be non-negative: {self.timestamp_s}")
        if self.label not in VALID_LABELS:
            raise ValueError(f"label must be one of {VALID_LABELS}: {self.label}")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}: {self.split}")


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------


@dataclass
class FeatureDataset:
    """Collection of feature records with metadata."""

    records: list[FeatureRecord] = field(default_factory=list)
    encoder_name: str = ""
    encoder_version: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    # Statistics (populated after building)
    n_pickup: int = 0
    n_putdown: int = 0
    n_negative: int = 0
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0

    def compute_stats(self) -> None:
        """Recompute statistics from records."""
        self.n_pickup = sum(1 for r in self.records if r.label == "pickup")
        self.n_putdown = sum(1 for r in self.records if r.label == "putdown")
        self.n_negative = sum(1 for r in self.records if r.label == "negative")
        self.n_train = sum(1 for r in self.records if r.split == "train")
        self.n_val = sum(1 for r in self.records if r.split == "val")
        self.n_test = sum(1 for r in self.records if r.split == "test")

    def filter_by_split(self, split: str) -> list[FeatureRecord]:
        """Return records for a specific split."""
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}: {split}")
        return [r for r in self.records if r.split == split]

    def filter_by_label(self, label: str) -> list[FeatureRecord]:
        """Return records for a specific label."""
        if label not in VALID_LABELS:
            raise ValueError(f"label must be one of {VALID_LABELS}: {label}")
        return [r for r in self.records if r.label == label]
