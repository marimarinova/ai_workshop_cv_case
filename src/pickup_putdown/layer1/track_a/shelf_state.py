"""Shelf-transition classifier for Track A.

Derives shelf-transition labels from verified event type and sample position:
  - pickup  + post → object_removed
  - putdown + post → object_placed
  - negative + post → no_change

Only post-position samples are used because they represent the completed
state change (or confirmed no-change). Pre-event and contact/mid samples
are excluded to avoid ambiguous labeling.

Output states: object_removed, object_placed, no_change, uncertain
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pickup_putdown.layer1.track_a.classifier import (
    TrackAClassifier,
)

if TYPE_CHECKING:
    from pickup_putdown.layer1.track_a.contracts import FeatureRecord

logger = logging.getLogger(__name__)

SHELF_STATE_CLASS_NAMES = ["object_removed", "object_placed", "no_change"]
SHELF_STATE_ALL_OUTPUTS = ["object_removed", "object_placed", "no_change", "uncertain"]


# ---------------------------------------------------------------------------
# Label derivation
# ---------------------------------------------------------------------------


def derive_shelf_state_label(record: FeatureRecord) -> str | None:
    """Derive a shelf-transition label from a feature record.

    Returns None for records that cannot be labeled (non-shelf crops,
    non-post positions, or ambiguous labels).

    Args:
        record: Feature record from the Phase 1 dataset.

    Returns:
        Shelf-state label or None.
    """
    if record.crop_type != "shelf":
        return None

    if record.sample_position != "post":
        return None

    if record.label == "pickup":
        return "object_removed"
    if record.label == "putdown":
        return "object_placed"
    if record.label == "negative":
        return "no_change"

    return None


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def extract_shelf_training_data(
    records: list[FeatureRecord],
    split: str = "train",
) -> tuple[np.ndarray, list[str], list[FeatureRecord]]:
    """Extract embeddings and labels for shelf-state training.

    Loads embeddings for supervised shelf records in the given split.

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
        if record.crop_type != "shelf":
            continue

        label = derive_shelf_state_label(record)
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


def create_shelf_classifier(
    confidence_threshold: float = 0.60,
    margin_threshold: float = 0.15,
    random_seed: int = 42,
    class_weight: str = "balanced",
    max_iter: int = 1000,
) -> TrackAClassifier:
    """Create a configured shelf-transition classifier.

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
        classifier_name="shelf_state",
        crop_type="shelf",
        class_names=SHELF_STATE_CLASS_NAMES,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
        random_seed=random_seed,
        class_weight=class_weight,
        max_iter=max_iter,
    )


def train_shelf_classifier(
    records: list[FeatureRecord],
    confidence_threshold: float = 0.60,
    margin_threshold: float = 0.15,
    random_seed: int = 42,
    class_weight: str = "balanced",
    max_iter: int = 1000,
) -> tuple[TrackAClassifier, dict[str, Any]]:
    """Train the shelf-state classifier on the train split, evaluate on val.

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
    classifier = create_shelf_classifier(
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
        random_seed=random_seed,
        class_weight=class_weight,
        max_iter=max_iter,
    )

    # Extract training data
    train_embeddings, train_labels, train_records = extract_shelf_training_data(
        records, split="train"
    )

    if len(train_embeddings) == 0:
        raise ValueError("No supervised shelf-state training records found")

    # Validate we have at least 2 classes
    unique_labels = set(train_labels)
    if len(unique_labels) < 2:
        raise ValueError(
            f"Need at least 2 shelf-state classes for training, got {len(unique_labels)}: "
            f"{sorted(unique_labels)}"
        )

    # Get encoder info from first record
    encoder_name = train_records[0].encoder_name
    encoder_version = train_records[0].encoder_version

    # Train
    classifier.train(train_embeddings, train_labels)

    # Extract validation data
    val_embeddings, val_labels, val_records = extract_shelf_training_data(records, split="val")

    # Build report
    from collections import Counter

    train_class_counts = dict(Counter(train_labels))
    report: dict[str, Any] = {
        "classifier": "shelf_state",
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
        "Shelf-state classifier: train=%d (%s), val=%d",
        len(train_records),
        train_class_counts,
        len(val_records),
    )

    return classifier, report
