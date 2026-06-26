"""Tests for shelf-transition classifier label derivation and training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pickup_putdown.layer1.track_a.contracts import (
    CropGeometry,
    FeatureRecord,
)
from pickup_putdown.layer1.track_a.shelf_state import (
    SHELF_STATE_CLASS_NAMES,
    create_shelf_classifier,
    derive_shelf_state_label,
    extract_shelf_training_data,
    train_shelf_classifier,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_record(
    label: str = "pickup",
    sample_position: str = "post",
    crop_type: str = "shelf",
    split: str = "train",
    embedding_dim: int = 64,
) -> FeatureRecord:
    emb_path = Path(f"/tmp/test_emb_{label}_{sample_position}_{crop_type}.npy")
    np.save(emb_path, np.random.randn(embedding_dim))
    return FeatureRecord(
        crop_id=f"crop_{label}_{sample_position}",
        clip_id="clip_1",
        candidate_id="cand_1",
        timestamp_s=1.0,
        sample_position=sample_position,
        crop_type=crop_type,
        geometry=CropGeometry(x=0, y=0, width=224, height=224),
        embedding_path=emb_path,
        encoder_name="mobilenet_v3_small",
        encoder_version="v1",
        label=label,
        split=split,
    )


# ---------------------------------------------------------------------------
# Label derivation tests
# ---------------------------------------------------------------------------


class TestDeriveShelfStateLabel:
    def test_pickup_post_is_object_removed(self) -> None:
        rec = _make_record(label="pickup", sample_position="post")
        assert derive_shelf_state_label(rec) == "object_removed"

    def test_putdown_post_is_object_placed(self) -> None:
        rec = _make_record(label="putdown", sample_position="post")
        assert derive_shelf_state_label(rec) == "object_placed"

    def test_negative_post_is_no_change(self) -> None:
        rec = _make_record(label="negative", sample_position="post")
        assert derive_shelf_state_label(rec) == "no_change"

    def test_pickup_pre_excluded(self) -> None:
        rec = _make_record(label="pickup", sample_position="pre")
        assert derive_shelf_state_label(rec) is None

    def test_pickup_contact_excluded(self) -> None:
        rec = _make_record(label="pickup", sample_position="contact")
        assert derive_shelf_state_label(rec) is None

    def test_hand_crop_excluded(self) -> None:
        rec = _make_record(label="pickup", sample_position="post", crop_type="hand")
        assert derive_shelf_state_label(rec) is None


# ---------------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------------


class TestShelfClassifierTraining:
    def test_train_deterministic(self) -> None:
        dim = 32
        np.random.seed(42)
        X = np.random.randn(30, dim)
        y = ["object_removed"] * 10 + ["object_placed"] * 10 + ["no_change"] * 10

        c1 = create_shelf_classifier(random_seed=42)
        c1.train(X, y)

        c2 = create_shelf_classifier(random_seed=42)
        c2.train(X, y)

        p1 = c1.predict_batch(X[:5])
        p2 = c2.predict_batch(X[:5])
        for a, b in zip(p1, p2):
            assert a.state == b.state

    def test_train_fails_zero_records(self) -> None:
        records: list[FeatureRecord] = []
        with pytest.raises(ValueError, match="No supervised"):
            train_shelf_classifier(records)

    def test_train_fails_single_class(self) -> None:
        dim = 32
        X = np.random.randn(10, dim)
        y = ["object_removed"] * 10

        c = create_shelf_classifier(random_seed=42)
        with pytest.raises(ValueError, match="at least 2"):
            c.train(X, y)

    def test_wrong_embedding_dim_rejected(self) -> None:
        dim = 32
        X = np.random.randn(30, dim)
        y = ["object_removed"] * 10 + ["object_placed"] * 10 + ["no_change"] * 10

        c = create_shelf_classifier(random_seed=42)
        c.train(X, y)

        bad_emb = np.random.randn(64)
        with pytest.raises(ValueError, match="dim"):
            c.predict(bad_emb)

    def test_crop_type_mismatch(self) -> None:
        records = [
            _make_record(label="pickup", sample_position="post", crop_type="hand"),
        ]
        embs, labels, matched = extract_shelf_training_data(records)
        assert len(matched) == 0


# ---------------------------------------------------------------------------
# Prediction tests
# ---------------------------------------------------------------------------


class TestShelfClassifierPrediction:
    @pytest.fixture()
    def trained_classifier(self):
        dim = 32
        np.random.seed(42)
        X = np.random.randn(60, dim)
        y = ["object_removed"] * 20 + ["object_placed"] * 20 + ["no_change"] * 20

        c = create_shelf_classifier(random_seed=42)
        c.train(X, y)
        return c, X, dim

    def test_prediction_includes_probabilities(self, trained_classifier) -> None:
        c, X, dim = trained_classifier
        pred = c.predict(X[0])
        assert isinstance(pred.probabilities, dict)
        for cls in SHELF_STATE_CLASS_NAMES:
            assert cls in pred.probabilities
        assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-6

    def test_confident_prediction_returns_class(self, trained_classifier) -> None:
        c, X, dim = trained_classifier
        pred = c.predict(X[0])
        if not pred.is_uncertain:
            assert pred.state in SHELF_STATE_CLASS_NAMES

    def test_low_confidence_returns_uncertain(self) -> None:
        dim = 32
        np.random.seed(42)
        X = np.random.randn(60, dim)
        y = ["object_removed"] * 20 + ["object_placed"] * 20 + ["no_change"] * 20

        c = create_shelf_classifier(
            confidence_threshold=0.99,
            margin_threshold=0.5,
            random_seed=42,
        )
        c.train(X, y)
        pred = c.predict(X[0])
        assert pred.is_uncertain or pred.state == "uncertain"

    def test_low_margin_returns_uncertain(self) -> None:
        dim = 32
        np.random.seed(42)
        X = np.random.randn(60, dim)
        y = ["object_removed"] * 20 + ["object_placed"] * 20 + ["no_change"] * 20

        c = create_shelf_classifier(
            confidence_threshold=0.1,
            margin_threshold=0.99,
            random_seed=42,
        )
        c.train(X, y)
        pred = c.predict(X[0])
        assert pred.is_uncertain or pred.state == "uncertain"


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


class TestShelfClassifierPersistence:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        dim = 32
        np.random.seed(42)
        X = np.random.randn(60, dim)
        y = ["object_removed"] * 20 + ["object_placed"] * 20 + ["no_change"] * 20

        c = create_shelf_classifier(random_seed=42)
        c.train(X, y)
        preds_before = c.predict_batch(X[:3])

        joblib_path = tmp_path / "shelf_state.joblib"
        c.save_pipeline(joblib_path)

        loaded, metadata = type(c).load_pipeline(joblib_path)
        preds_after = loaded.predict_batch(X[:3])

        for a, b in zip(preds_before, preds_after):
            assert a.state == b.state

    def test_metadata_contains_required_fields(self, tmp_path: Path) -> None:
        dim = 32
        np.random.seed(42)
        X = np.random.randn(60, dim)
        y = ["object_removed"] * 20 + ["object_placed"] * 20 + ["no_change"] * 20

        c = create_shelf_classifier(random_seed=42)
        c.train(X, y)

        meta = c.build_metadata(
            encoder_name="mobilenet_v3_small",
            encoder_version="v1",
            training_record_counts={
                "object_removed": 20,
                "object_placed": 20,
                "no_change": 20,
            },
            train_split_count=60,
            val_split_count=15,
        )
        d = meta.to_dict()
        assert "class_names" in d
        assert "embedding_dim" in d
        assert d["embedding_dim"] == dim
        assert d["crop_type"] == "shelf"
