"""Hand-state classifier for Track A.

Derives hand-state labels from verified event type and sample position:
  - pickup  + pre  → empty
  - pickup  + post → carrying
  - putdown + pre  → carrying
  - putdown + post → empty

Negative candidates and contact/mid positions are excluded because they
do not provide a reliable hand-state label.

Output states: empty, carrying, uncertain
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pickup_putdown.layer1.track_a.classifier import (
    TrackAClassifier,
)

if TYPE_CHECKING:
    from pickup_putdown.layer1.track_a.contracts import FeatureRecord

logger = logging.getLogger(__name__)

HAND_STATE_CLASS_NAMES = ["empty", "carrying"]
HAND_STATE_ALL_OUTPUTS = ["empty", "carrying", "uncertain"]


# ---------------------------------------------------------------------------
# Label derivation
# ---------------------------------------------------------------------------


def derive_hand_state_label(record: FeatureRecord) -> str | None:
    """Derive a hand-state label from a feature record.

    Returns None for records that cannot be labeled (negatives, contact/mid
    positions, or non-hand crops).

    Args:
        record: Feature record from the Phase 1 dataset.

    Returns:
        Hand-state label or None.
    """
    if record.crop_type != "hand":
        return None

    if record.label == "negative":
        return None

    if record.sample_position not in ("pre", "post"):
        return None

    if record.label == "pickup" and record.sample_position == "pre":
        return "empty"
    if record.label == "pickup" and record.sample_position == "post":
        return "carrying"
    if record.label == "putdown" and record.sample_position == "pre":
        return "carrying"
    if record.label == "putdown" and record.sample_position == "post":
        return "empty"

    return None


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def extract_hand_training_data(
    records: list[FeatureRecord],
    split: str = "train",
) -> tuple[np.ndarray, list[str], list[FeatureRecord]]:
    """Extract embeddings and labels for hand-state training.

    Loads embeddings for supervised hand records in the given split.

    Args:
        records: All feature records.
        split: Split to use ("train" or "val").

    Returns:
        Tuple of (embeddings array, labels, matched records).
    """
    embeddings: list[np.ndarray] = []
    labels: list[str] = []
    matched: list[FeatureRecord] = []

    for record in records:
        if record.split != split:
            continue
        if record.crop_type != "hand":
            continue

        label = derive_hand_state_label(record)
        if label is None:
            continue

        emb_path = Path(record.embedding_path)
        if not emb_path.exists():
            logger.warning("Missing embedding: %s", emb_path)
            continue

        embedding = np.load(emb_path)
        embeddings.append(embedding)
        labels.append(label)
        matched.append(record)

    if not embeddings:
        return np.empty((0, 0)), [], []

    return np.array(embeddings), labels, matched


# ---------------------------------------------------------------------------
# Classifier factory and training
# ---------------------------------------------------------------------------


def create_hand_classifier(
    confidence_threshold: float = 0.60,
    margin_threshold: float = 0.15,
    random_seed: int = 42,
    class_weight: str = "balanced",
    max_iter: int = 1000,
) -> TrackAClassifier:
    """Create a configured hand-state classifier.

    Args:
        confidence_threshold: Min probability for confident prediction.
        margin_threshold: Min margin between top-2 probabilities.
        random_seed: Random seed for LogisticRegression.
        class_weight: Class weight strategy.
        max_iter: Max iterations for solver.

    Returns:
        Configured TrackAClassifier instance.
    """
    return TrackAClassifier(
        classifier_name="hand_state",
        crop_type="hand",
        class_names=HAND_STATE_CLASS_NAMES,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
        random_seed=random_seed,
        class_weight=class_weight,
        max_iter=max_iter,
    )


def train_hand_classifier(
    records: list[FeatureRecord],
    confidence_threshold: float = 0.60,
    margin_threshold: float = 0.15,
    random_seed: int = 42,
    class_weight: str = "balanced",
    max_iter: int = 1000,
) -> tuple[TrackAClassifier, dict[str, any]]:
    """Train the hand-state classifier on the train split, evaluate on val.

    Args:
        records: All feature records from the Phase 1 dataset.
        confidence_threshold: Min probability for confident prediction.
        margin_threshold: Min margin between top-2 probabilities.
        random_seed: Random seed for LogisticRegression.
        class_weight: Class weight strategy.
        max_iter: Max iterations for solver.

    Returns:
        Tuple of (trained classifier, training report dict).
    """
    classifier = create_hand_classifier(
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
        random_seed=random_seed,
        class_weight=class_weight,
        max_iter=max_iter,
    )

    # Extract training data
    train_embeddings, train_labels, train_records = extract_hand_training_data(
        records, split="train"
    )

    if len(train_embeddings) == 0:
        raise ValueError("No supervised hand-state training records found")

    # Validate we have both classes
    unique_labels = set(train_labels)
    if len(unique_labels) < 2:
        raise ValueError(
            f"Need at least 2 hand-state classes for training, got {len(unique_labels)}: "
            f"{sorted(unique_labels)}"
        )

    # Get encoder info from first record
    encoder_name = train_records[0].encoder_name
    encoder_version = train_records[0].encoder_version

    # Train
    classifier.train(train_embeddings, train_labels)

    # Extract validation data
    val_embeddings, val_labels, val_records = extract_hand_training_data(records, split="val")

    # Build report
    from collections import Counter

    train_class_counts = dict(Counter(train_labels))
    report: dict[str, any] = {
        "classifier": "hand_state",
        "train_records": len(train_records),
        "val_records": len(val_records),
        "train_class_counts": train_class_counts,
        "embedding_dim": train_embeddings.shape[1] if len(train_embeddings) > 0 else 0,
        "encoder_name": encoder_name,
        "encoder_version": encoder_version,
        "confidence_threshold": confidence_threshold,
        "margin_threshold": margin_threshold,
        "random_seed": random_seed,
        "class_weight": class_weight,
        "max_iter": max_iter,
    }

    # Validation metrics
    if len(val_embeddings) > 0:
        val_metrics = classifier.evaluate(val_embeddings, val_labels)
        report["validation"] = val_metrics
        val_class_counts = dict(Counter(val_labels))
        report["val_class_counts"] = val_class_counts
    else:
        report["validation"] = {"error": "No validation records"}
        report["val_class_counts"] = {}

    logger.info(
        "Hand-state classifier: train=%d (%s), val=%d",
        len(train_records),
        train_class_counts,
        len(val_records),
    )

    return classifier, report
