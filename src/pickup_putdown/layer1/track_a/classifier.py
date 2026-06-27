"""Shared sklearn classifier base for Track A state classifiers.

Provides a StandardScaler → LogisticRegression pipeline with:
- Deterministic training
- Confidence-aware prediction with uncertain thresholding
- Artifact persistence via joblib + JSON metadata
- Validation metrics and reporting
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prediction result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Prediction:
    """Single prediction result with confidence and uncertainty."""

    state: str
    confidence: float
    probabilities: dict[str, float]
    is_uncertain: bool


# ---------------------------------------------------------------------------
# Training metadata
# ---------------------------------------------------------------------------


@dataclass
class ClassifierMetadata:
    """Metadata persisted alongside the classifier artifact."""

    classifier_name: str
    class_names: list[str]
    embedding_dim: int
    crop_type: str
    encoder_name: str
    encoder_version: str
    training_timestamp: str
    training_record_counts: dict[str, int]
    train_split_count: int
    val_split_count: int
    confidence_threshold: float
    margin_threshold: float
    random_seed: int
    class_weight: str
    max_iter: int
    artifact_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_version": self.artifact_version,
            "classifier_name": self.classifier_name,
            "class_names": self.class_names,
            "embedding_dim": self.embedding_dim,
            "crop_type": self.crop_type,
            "encoder_name": self.encoder_name,
            "encoder_version": self.encoder_version,
            "training_timestamp": self.training_timestamp,
            "training_record_counts": self.training_record_counts,
            "train_split_count": self.train_split_count,
            "val_split_count": self.val_split_count,
            "confidence_threshold": self.confidence_threshold,
            "margin_threshold": self.margin_threshold,
            "random_seed": self.random_seed,
            "class_weight": self.class_weight,
            "max_iter": self.max_iter,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassifierMetadata:
        return cls(**data)


# ---------------------------------------------------------------------------
# Metrics report
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    class_names: list[str],
) -> dict[str, Any]:
    """Compute classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: Ordered list of class names.

    Returns:
        Dict with accuracy, balanced accuracy, per-class metrics, and confusion matrix.
    """
    report: dict[str, Any] = {}

    if not y_true or not y_pred:
        report["error"] = "Empty predictions"
        report["n_true"] = len(y_true)
        report["n_pred"] = len(y_pred)
        return report

    report["n_samples"] = len(y_true)
    report["accuracy"] = float(accuracy_score(y_true, y_pred))
    report["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    report["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", labels=class_names))

    # Per-class metrics
    per_class: dict[str, dict[str, float]] = {}
    present_classes = list(set(y_true))
    for cls in class_names:
        if cls in present_classes:
            prec = float(precision_score(y_true, y_pred, labels=[cls], average="micro"))
            rec = float(recall_score(y_true, y_pred, labels=[cls], average="micro"))
            f1 = float(f1_score(y_true, y_pred, labels=[cls], average="micro"))
            support = sum(1 for y in y_true if y == cls)
            per_class[cls] = {
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "support": support,
            }
        else:
            per_class[cls] = {
                "precision": None,
                "recall": None,
                "f1": None,
                "support": 0,
                "missing_in_validation": True,
            }
    report["per_class"] = per_class

    # Confusion matrix
    try:
        cm = confusion_matrix(y_true, y_pred, labels=class_names)
        report["confusion_matrix"] = cm.tolist()
        report["confusion_matrix_labels"] = class_names
    except ValueError:
        report["confusion_matrix"] = None
        report["confusion_matrix_error"] = "Could not compute confusion matrix"

    # Classification report text
    report["classification_report"] = classification_report(
        y_true, y_pred, labels=class_names, zero_division=0
    )

    return report


# ---------------------------------------------------------------------------
# Base classifier
# ---------------------------------------------------------------------------


class TrackAClassifier:
    """Base classifier using StandardScaler → LogisticRegression pipeline.

    Supports training, prediction with confidence/uncertainty, and
    artifact persistence.
    """

    def __init__(
        self,
        classifier_name: str,
        crop_type: str,
        class_names: list[str],
        confidence_threshold: float = 0.60,
        margin_threshold: float = 0.15,
        random_seed: int = 42,
        class_weight: str = "balanced",
        max_iter: int = 1000,
    ):
        self.classifier_name = classifier_name
        self.crop_type = crop_type
        self.class_names = list(class_names)
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.random_seed = random_seed
        self.class_weight = class_weight
        self.max_iter = max_iter

        self._pipeline: Any | None = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        embeddings: np.ndarray,
        labels: list[str],
    ) -> None:
        """Train the classifier pipeline.

        Args:
            embeddings: (n_samples, embedding_dim) numpy array.
            labels: List of label strings matching class_names.

        Raises:
            ValueError: If no records, single class, or wrong dimensions.
        """
        if len(embeddings) == 0:
            raise ValueError(f"No training records for {self.classifier_name}")

        unique_labels = set(labels)
        if len(unique_labels) < 2:
            raise ValueError(
                f"Need at least 2 classes for training, got {len(unique_labels)}: "
                f"{sorted(unique_labels)}"
            )

        valid_labels = set(self.class_names)
        invalid = unique_labels - valid_labels
        if invalid:
            raise ValueError(
                f"Unknown labels {sorted(invalid)} for {self.classifier_name}. "
                f"Expected: {self.class_names}"
            )

        n_samples, emb_dim = embeddings.shape
        logger.info(
            "Training %s: %d samples, dim=%d, classes=%s",
            self.classifier_name,
            n_samples,
            emb_dim,
            sorted(unique_labels),
        )

        self._embedding_dim = emb_dim

        self._pipeline = [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight=self.class_weight,
                    max_iter=self.max_iter,
                    random_state=self.random_seed,
                    solver="lbfgs",
                ),
            ),
        ]

        from sklearn.pipeline import Pipeline

        pipe = Pipeline(self._pipeline)
        pipe.fit(embeddings, labels)
        self._is_fitted = True
        self.class_names = list(pipe.named_steps["classifier"].classes_)

        logger.info("Trained %s with classes %s", self.classifier_name, self.class_names)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, embedding: np.ndarray) -> Prediction:
        """Predict state for a single embedding.

        Args:
            embedding: 1D numpy array of shape (embedding_dim,).

        Returns:
            Prediction with state, confidence, probabilities, and uncertainty flag.
        """
        if embedding.ndim == 2:
            if embedding.shape[0] == 1:
                return self.predict_batch(embedding)[0]
            raise ValueError(
                f"Expected 1D array, got 2D shape {embedding.shape}. "
                f"Use predict_batch() for multiple samples."
            )
        return self.predict_batch(embedding.reshape(1, -1))[0]

    def predict_batch(self, embeddings: np.ndarray) -> list[Prediction]:
        """Predict states for a batch of embeddings.

        Args:
            embeddings: (n_samples, embedding_dim) numpy array.

        Returns:
            List of Prediction objects.
        """
        if not self._is_fitted:
            raise RuntimeError(f"{self.classifier_name} has not been trained")

        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D array, got {embeddings.ndim}D")

        if embeddings.shape[1] != self._embedding_dim:
            raise ValueError(
                f"Expected embedding dim {self._embedding_dim}, got {embeddings.shape[1]}"
            )

        from sklearn.pipeline import Pipeline

        pipe = Pipeline(self._pipeline)
        proba = pipe.predict_proba(embeddings)

        results: list[Prediction] = []
        for row in proba:
            probs = dict(zip(self.class_names, row, strict=True))
            max_prob = float(max(row))
            sorted_probs = sorted(row, reverse=True)
            margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0

            is_uncertain = max_prob < self.confidence_threshold or margin < self.margin_threshold

            pred_idx = int(np.argmax(row))
            state = self.class_names[pred_idx] if not is_uncertain else "uncertain"

            results.append(
                Prediction(
                    state=state,
                    confidence=max_prob,
                    probabilities={k: float(v) for k, v in probs.items()},
                    is_uncertain=is_uncertain,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        embeddings: np.ndarray,
        labels: list[str],
    ) -> dict[str, Any]:
        """Evaluate on a held-out set.

        Args:
            embeddings: (n_samples, embedding_dim) numpy array.
            labels: True labels.

        Returns:
            Metrics dict.
        """
        preds = self.predict_batch(embeddings)
        y_true = labels
        y_pred = [p.state for p in preds]

        metrics = compute_metrics(y_true, y_pred, self.class_names)

        # Add uncertain stats
        n_uncertain = sum(1 for p in preds if p.is_uncertain)
        metrics["n_uncertain"] = n_uncertain
        metrics["proportion_uncertain"] = n_uncertain / len(preds) if preds else 0.0

        # Class counts
        from collections import Counter

        metrics["true_class_counts"] = dict(Counter(y_true))
        metrics["predicted_class_counts"] = dict(Counter(y_pred))

        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_pipeline(self, path: Path | str) -> Path:
        """Save the trained pipeline to disk.

        Args:
            path: Output path for the .joblib file.

        Returns:
            Path where the pipeline was saved.
        """
        if not self._is_fitted:
            raise RuntimeError(f"{self.classifier_name} has not been trained")

        from sklearn.pipeline import Pipeline

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        pipe = Pipeline(self._pipeline)
        joblib.dump(pipe, path)
        logger.info("Saved %s pipeline to %s", self.classifier_name, path)
        return path

    @classmethod
    def load_pipeline(
        cls,
        path: Path | str,
        metadata_path: Path | str | None = None,
    ) -> tuple[TrackAClassifier, ClassifierMetadata | None]:
        """Load a trained classifier from disk.

        Args:
            path: Path to the .joblib file.
            metadata_path: Optional path to metadata JSON.

        Returns:
            Tuple of (classifier instance, metadata or None).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pipeline not found: {path}")

        pipe = joblib.load(path)
        classifier_obj = pipe.named_steps["classifier"]

        class_names = list(classifier_obj.classes_)
        embedding_dim = classifier_obj.coef_.shape[1]
        assert len(class_names) > 0

        # Extract classifier config
        random_seed = int(classifier_obj.random_state) if classifier_obj.random_state else 42
        max_iter = int(classifier_obj.max_iter)
        class_weight = str(classifier_obj.class_weight)

        # Default values — will be overridden by metadata if available
        crop_type = "unknown"
        confidence_threshold = 0.60
        margin_threshold = 0.15

        metadata = None
        if metadata_path:
            metadata_path = Path(metadata_path)
            if metadata_path.exists():
                metadata = ClassifierMetadata.from_dict(json.loads(metadata_path.read_text()))
                crop_type = metadata.crop_type
                confidence_threshold = metadata.confidence_threshold
                margin_threshold = metadata.margin_threshold
                random_seed = metadata.random_seed
                max_iter = metadata.max_iter
                class_weight = metadata.class_weight

        instance = cls(
            classifier_name=class_names[0] if len(class_names) == 1 else "loaded",
            crop_type=crop_type,
            class_names=class_names,
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
            random_seed=random_seed,
            class_weight=class_weight,
            max_iter=max_iter,
        )
        instance._pipeline = [
            ("scaler", pipe.named_steps["scaler"]),
            ("classifier", pipe.named_steps["classifier"]),
        ]
        instance._is_fitted = True
        instance._embedding_dim = embedding_dim

        logger.info(
            "Loaded %s classifier: classes=%s, dim=%d",
            instance.classifier_name,
            class_names,
            embedding_dim,
        )
        return instance, metadata

    def build_metadata(
        self,
        encoder_name: str,
        encoder_version: str,
        training_record_counts: dict[str, int],
        train_split_count: int,
        val_split_count: int,
    ) -> ClassifierMetadata:
        """Build metadata for the current classifier state.

        Args:
            encoder_name: Name of the feature encoder.
            encoder_version: Version of the feature encoder.
            training_record_counts: Per-class training record counts.
            train_split_count: Total training records.
            val_split_count: Total validation records.

        Returns:
            ClassifierMetadata instance.
        """
        return ClassifierMetadata(
            classifier_name=self.classifier_name,
            class_names=self.class_names,
            embedding_dim=self._embedding_dim,
            crop_type=self.crop_type,
            encoder_name=encoder_name,
            encoder_version=encoder_version,
            training_timestamp=datetime.now().isoformat(),
            training_record_counts=training_record_counts,
            train_split_count=train_split_count,
            val_split_count=val_split_count,
            confidence_threshold=self.confidence_threshold,
            margin_threshold=self.margin_threshold,
            random_seed=self.random_seed,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
        )
