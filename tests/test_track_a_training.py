"""Integration tests for Track A classifier training pipeline.

Covers end-to-end training, artifact persistence, and metrics reporting
using synthetic embeddings.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pickup_putdown.layer1.track_a.classifier import (
    TrackAClassifier,
    compute_metrics,
)
from pickup_putdown.layer1.track_a.contracts import (
    CropGeometry,
    FeatureRecord,
)
from pickup_putdown.layer1.track_a.hand_state import (
    HAND_STATE_CLASS_NAMES,
    derive_hand_state_label,
    train_hand_classifier,
)
from pickup_putdown.layer1.track_a.shelf_state import (
    derive_shelf_state_label,
    train_shelf_classifier,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hand_record(
    label: str,
    sample_position: str,
    split: str,
    emb: np.ndarray,
    idx: int,
) -> FeatureRecord:
    emb_path = Path(f"/tmp/test_integration_hand_{idx}.npy")
    np.save(emb_path, emb)
    return FeatureRecord(
        crop_id=f"hand_{idx}",
        clip_id=f"clip_{split}_{idx}",
        candidate_id=f"cand_{idx}",
        timestamp_s=float(idx),
        sample_position=sample_position,
        crop_type="hand",
        geometry=CropGeometry(x=0, y=0, width=224, height=224),
        embedding_path=emb_path,
        encoder_name="mobilenet_v3_small",
        encoder_version="v1",
        label=label,
        split=split,
    )


def _make_shelf_record(
    label: str,
    sample_position: str,
    split: str,
    emb: np.ndarray,
    idx: int,
) -> FeatureRecord:
    emb_path = Path(f"/tmp/test_integration_shelf_{idx}.npy")
    np.save(emb_path, emb)
    return FeatureRecord(
        crop_id=f"shelf_{idx}",
        clip_id=f"clip_{split}_{idx}",
        candidate_id=f"cand_{idx}",
        timestamp_s=float(idx),
        sample_position=sample_position,
        crop_type="shelf",
        geometry=CropGeometry(x=0, y=0, width=224, height=224),
        embedding_path=emb_path,
        encoder_name="mobilenet_v3_small",
        encoder_version="v1",
        label=label,
        split=split,
    )


# ---------------------------------------------------------------------------
# Label derivation integration
# ---------------------------------------------------------------------------


class TestLabelDerivationIntegration:
    def test_all_hand_labels_derived(self) -> None:
        dim = 16
        np.random.seed(42)

        records = [
            _make_hand_record("pickup", "pre", "train", np.random.randn(dim), 0),
            _make_hand_record("pickup", "post", "train", np.random.randn(dim), 1),
            _make_hand_record("putdown", "pre", "train", np.random.randn(dim), 2),
            _make_hand_record("putdown", "post", "train", np.random.randn(dim), 3),
        ]

        labels = [derive_hand_state_label(r) for r in records]
        assert labels == ["empty", "carrying", "carrying", "empty"]

    def test_all_shelf_labels_derived(self) -> None:
        dim = 16
        np.random.seed(42)

        records = [
            _make_shelf_record("pickup", "post", "train", np.random.randn(dim), 0),
            _make_shelf_record("putdown", "post", "train", np.random.randn(dim), 1),
            _make_shelf_record("negative", "post", "train", np.random.randn(dim), 2),
        ]

        labels = [derive_shelf_state_label(r) for r in records]
        assert labels == ["object_removed", "object_placed", "no_change"]

    def test_unsupported_records_excluded(self) -> None:
        dim = 16
        np.random.seed(42)

        records = [
            _make_hand_record("negative", "pre", "train", np.random.randn(dim), 0),
            _make_hand_record("pickup", "contact", "train", np.random.randn(dim), 1),
            _make_hand_record("pickup", "mid", "train", np.random.randn(dim), 2),
            _make_shelf_record("pickup", "pre", "train", np.random.randn(dim), 3),
        ]

        hand_labels = [derive_hand_state_label(r) for r in records[:3]]
        shelf_labels = [derive_shelf_state_label(r) for r in records[3:]]
        assert all(label is None for label in hand_labels)
        assert all(label is None for label in shelf_labels)


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------


class TestFullTrainingPipeline:
    def _build_hand_records(self) -> list[FeatureRecord]:
        dim = 32
        np.random.seed(42)
        records: list[FeatureRecord] = []

        # Train: balanced classes
        train_configs = [
            ("pickup", "pre", "train"),
            ("pickup", "post", "train"),
            ("putdown", "pre", "train"),
            ("putdown", "post", "train"),
        ]
        for i, (lbl, pos, spl) in enumerate(train_configs * 5):
            emb = np.random.randn(dim)
            records.append(_make_hand_record(lbl, pos, spl, emb, i))

        # Val
        val_configs = [
            ("pickup", "pre", "val"),
            ("pickup", "post", "val"),
            ("putdown", "pre", "val"),
            ("putdown", "post", "val"),
        ]
        for i, (lbl, pos, spl) in enumerate(val_configs * 2):
            emb = np.random.randn(dim)
            records.append(_make_hand_record(lbl, pos, spl, emb, 100 + i))

        return records

    def _build_shelf_records(self) -> list[FeatureRecord]:
        dim = 32
        np.random.seed(42)
        records: list[FeatureRecord] = []

        train_configs = [
            ("pickup", "post", "train"),
            ("putdown", "post", "train"),
            ("negative", "post", "train"),
        ]
        for i, (lbl, pos, spl) in enumerate(train_configs * 5):
            emb = np.random.randn(dim)
            records.append(_make_shelf_record(lbl, pos, spl, emb, i))

        val_configs = [
            ("pickup", "post", "val"),
            ("putdown", "post", "val"),
            ("negative", "post", "val"),
        ]
        for i, (lbl, pos, spl) in enumerate(val_configs * 2):
            emb = np.random.randn(dim)
            records.append(_make_shelf_record(lbl, pos, spl, emb, 100 + i))

        return records

    def test_hand_classifier_trains_and_evaluates(self) -> None:
        records = self._build_hand_records()
        classifier, report = train_hand_classifier(records, random_seed=42)

        assert report["train_records"] > 0
        assert report["val_records"] > 0
        assert "empty" in report["train_class_counts"]
        assert "carrying" in report["train_class_counts"]
        assert "validation" in report
        assert "accuracy" in report["validation"]

    def test_shelf_classifier_trains_and_evaluates(self) -> None:
        records = self._build_shelf_records()
        classifier, report = train_shelf_classifier(records, random_seed=42)

        assert report["train_records"] > 0
        assert report["val_records"] > 0
        assert "object_removed" in report["train_class_counts"]
        assert "object_placed" in report["train_class_counts"]
        assert "no_change" in report["train_class_counts"]
        assert "validation" in report
        assert "accuracy" in report["validation"]

    def test_train_val_separation(self) -> None:
        records = self._build_hand_records()
        classifier, report = train_hand_classifier(records, random_seed=42)

        train_count = report["train_records"]
        val_count = report["val_records"]
        assert train_count > 0
        assert val_count > 0
        assert train_count > val_count


# ---------------------------------------------------------------------------
# Metrics and reporting
# ---------------------------------------------------------------------------


class TestMetricsReporting:
    def test_metrics_contains_class_counts(self) -> None:
        y_true = ["a", "a", "b", "b", "b"]
        y_pred = ["a", "b", "b", "b", "a"]
        metrics = compute_metrics(y_true, y_pred, ["a", "b"])

        assert "accuracy" in metrics
        assert "balanced_accuracy" in metrics
        assert "macro_f1" in metrics
        assert "per_class" in metrics
        assert "confusion_matrix" in metrics

    def test_metrics_handles_missing_class(self) -> None:
        y_true = ["a", "a", "a"]
        y_pred = ["a", "a", "a"]
        metrics = compute_metrics(y_true, y_pred, ["a", "b"])

        assert "per_class" in metrics
        assert metrics["per_class"]["b"].get("missing_in_validation") is True
        assert metrics["per_class"]["b"]["support"] == 0

    def test_metrics_empty_input(self) -> None:
        metrics = compute_metrics([], [], ["a", "b"])
        assert "error" in metrics

    def test_confusion_matrix_in_metrics(self) -> None:
        y_true = ["a", "a", "b", "b"]
        y_pred = ["a", "a", "b", "b"]
        metrics = compute_metrics(y_true, y_pred, ["a", "b"])

        cm = metrics["confusion_matrix"]
        assert cm == [[2, 0], [0, 2]]
        assert metrics["confusion_matrix_labels"] == ["a", "b"]


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------


class TestArtifactPersistence:
    def test_full_artifact_cycle(self, tmp_path: Path) -> None:
        dim = 32
        np.random.seed(42)
        records = []

        train_configs = [
            ("pickup", "pre", "train"),
            ("pickup", "post", "train"),
            ("putdown", "pre", "train"),
            ("putdown", "post", "train"),
        ]
        for i, (lbl, pos, spl) in enumerate(train_configs * 5):
            emb = np.random.randn(dim)
            records.append(_make_hand_record(lbl, pos, spl, emb, i))

        val_configs = [
            ("pickup", "pre", "val"),
            ("pickup", "post", "val"),
            ("putdown", "pre", "val"),
            ("putdown", "post", "val"),
        ]
        for i, (lbl, pos, spl) in enumerate(val_configs * 2):
            emb = np.random.randn(dim)
            records.append(_make_hand_record(lbl, pos, spl, emb, 100 + i))

        classifier, report = train_hand_classifier(records, random_seed=42)

        # Save
        joblib_path = tmp_path / "hand_state.joblib"
        classifier.save_pipeline(joblib_path)

        metadata = classifier.build_metadata(
            encoder_name="mobilenet_v3_small",
            encoder_version="v1",
            training_record_counts=report["train_class_counts"],
            train_split_count=report["train_records"],
            val_split_count=report["val_records"],
        )
        meta_path = tmp_path / "hand_state_metadata.json"
        meta_path.write_text(json.dumps(metadata.to_dict(), indent=2))

        metrics_path = tmp_path / "hand_state_metrics.json"
        metrics_path.write_text(json.dumps(report, indent=2, default=str))

        # Verify files
        assert joblib_path.exists()
        assert meta_path.exists()
        assert metrics_path.exists()

        # Load and verify
        loaded, loaded_meta = TrackAClassifier.load_pipeline(joblib_path, meta_path)
        assert loaded_meta is not None
        assert set(loaded_meta.class_names) == set(HAND_STATE_CLASS_NAMES)
        assert loaded_meta.embedding_dim == dim

        # Metrics contain expected fields
        loaded_metrics = json.loads(metrics_path.read_text())
        assert "train_records" in loaded_metrics
        assert "train_class_counts" in loaded_metrics
        assert "validation" in loaded_metrics
        val = loaded_metrics["validation"]
        assert "confusion_matrix" in val

    def test_missing_val_class_reported(self, tmp_path: Path) -> None:
        dim = 32
        np.random.seed(42)

        # Only "a" in val, "b" missing
        train_X = np.random.randn(40, dim)
        train_y = ["a"] * 20 + ["b"] * 20

        val_X = np.random.randn(10, dim)
        val_y = ["a"] * 10

        c = TrackAClassifier(
            classifier_name="test",
            crop_type="hand",
            class_names=["a", "b"],
            random_seed=42,
        )
        c.train(train_X, train_y)
        metrics = c.evaluate(val_X, val_y)

        assert metrics["per_class"]["b"].get("missing_in_validation") is True
