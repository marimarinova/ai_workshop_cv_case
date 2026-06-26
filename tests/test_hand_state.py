"""Tests for hand-state classifier label derivation and training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pickup_putdown.layer1.track_a.contracts import (
    CropGeometry,
    FeatureRecord,
)
from pickup_putdown.layer1.track_a.hand_state import (
    HAND_STATE_CLASS_NAMES,
    create_hand_classifier,
    derive_hand_state_label,
    extract_hand_training_data,
    train_hand_classifier,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_record(
    label: str = "pickup",
    sample_position: str = "pre",
    crop_type: str = "hand",
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


class TestDeriveHandStateLabel:
    def test_pickup_pre_is_empty(self) -> None:
        rec = _make_record(label="pickup", sample_position="pre")
        assert derive_hand_state_label(rec) == "empty"

    def test_pickup_post_is_carrying(self) -> None:
        rec = _make_record(label="pickup", sample_position="post")
        assert derive_hand_state_label(rec) == "carrying"

    def test_putdown_pre_is_carrying(self) -> None:
        rec = _make_record(label="putdown", sample_position="pre")
        assert derive_hand_state_label(rec) == "carrying"

    def test_putdown_post_is_empty(self) -> None:
        rec = _make_record(label="putdown", sample_position="post")
        assert derive_hand_state_label(rec) == "empty"

    def test_negative_excluded(self) -> None:
        rec = _make_record(label="negative", sample_position="pre")
        assert derive_hand_state_label(rec) is None

    def test_contact_excluded(self) -> None:
        rec = _make_record(label="pickup", sample_position="contact")
        assert derive_hand_state_label(rec) is None

    def test_mid_excluded(self) -> None:
        rec = _make_record(label="pickup", sample_position="mid")
        assert derive_hand_state_label(rec) is None

    def test_shelf_crop_excluded(self) -> None:
        rec = _make_record(label="pickup", sample_position="pre", crop_type="shelf")
        assert derive_hand_state_label(rec) is None


# ---------------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------------


class TestHandClassifierTraining:
    def test_train_deterministic(self) -> None:
        dim = 32
        np.random.seed(42)
        X = np.random.randn(20, dim)
        y = ["empty"] * 10 + ["carrying"] * 10

        c1 = create_hand_classifier(random_seed=42)
        c1.train(X, y)

        c2 = create_hand_classifier(random_seed=42)
        c2.train(X, y)

        p1 = c1.predict_batch(X[:5])
        p2 = c2.predict_batch(X[:5])
        for a, b in zip(p1, p2):
            assert a.state == b.state
            assert abs(a.confidence - b.confidence) < 1e-6

    def test_train_fails_zero_records(self) -> None:
        records: list[FeatureRecord] = []
        with pytest.raises(ValueError, match="No supervised"):
            train_hand_classifier(records)

    def test_train_fails_single_class(self) -> None:
        dim = 32
        X = np.random.randn(10, dim)
        y = ["empty"] * 10

        c = create_hand_classifier(random_seed=42)
        with pytest.raises(ValueError, match="at least 2"):
            c.train(X, y)

    def test_wrong_embedding_dim_rejected(self) -> None:
        dim = 32
        X = np.random.randn(20, dim)
        y = ["empty"] * 10 + ["carrying"] * 10

        c = create_hand_classifier(random_seed=42)
        c.train(X, y)

        bad_emb = np.random.randn(64)
        with pytest.raises(ValueError, match="dim"):
            c.predict(bad_emb)

    def test_crop_type_mismatch(self) -> None:
        records = [
            _make_record(label="pickup", sample_position="pre", crop_type="shelf"),
        ]
        embs, labels, matched = extract_hand_training_data(records)
        assert len(matched) == 0
        assert embs.shape[0] == 0


# ---------------------------------------------------------------------------
# Prediction tests
# ---------------------------------------------------------------------------


class TestHandClassifierPrediction:
    @pytest.fixture()
    def trained_classifier(self):
        dim = 32
        np.random.seed(42)
        X = np.random.randn(40, dim)
        y = ["empty"] * 20 + ["carrying"] * 20

        c = create_hand_classifier(random_seed=42)
        c.train(X, y)
        return c, X, dim

    def test_prediction_includes_probabilities(self, trained_classifier) -> None:
        c, X, dim = trained_classifier
        pred = c.predict(X[0])
        assert isinstance(pred.probabilities, dict)
        assert "empty" in pred.probabilities
        assert "carrying" in pred.probabilities
        assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-6

    def test_confident_prediction(self, trained_classifier) -> None:
        c, X, dim = trained_classifier
        pred = c.predict(X[0])
        assert not pred.is_uncertain or pred.state == "uncertain"

    def test_low_confidence_returns_uncertain(self) -> None:
        dim = 32
        np.random.seed(42)
        X = np.random.randn(40, dim)
        y = ["empty"] * 20 + ["carrying"] * 20

        c = create_hand_classifier(
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
        X = np.random.randn(40, dim)
        y = ["empty"] * 20 + ["carrying"] * 20

        c = create_hand_classifier(
            confidence_threshold=0.1,
            margin_threshold=0.99,
            random_seed=42,
        )
        c.train(X, y)
        pred = c.predict(X[0])
        assert pred.is_uncertain or pred.state == "uncertain"

    def test_batch_prediction(self, trained_classifier) -> None:
        c, X, dim = trained_classifier
        preds = c.predict_batch(X[:5])
        assert len(preds) == 5
        for p in preds:
            assert p.state in HAND_STATE_CLASS_NAMES or p.state == "uncertain"


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


class TestHandClassifierPersistence:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        dim = 32
        np.random.seed(42)
        X = np.random.randn(40, dim)
        y = ["empty"] * 20 + ["carrying"] * 20

        c = create_hand_classifier(random_seed=42)
        c.train(X, y)
        preds_before = c.predict_batch(X[:3])

        joblib_path = tmp_path / "hand_state.joblib"
        c.save_pipeline(joblib_path)

        loaded, metadata = type(c).load_pipeline(joblib_path)
        preds_after = loaded.predict_batch(X[:3])

        for a, b in zip(preds_before, preds_after):
            assert a.state == b.state
            assert abs(a.confidence - b.confidence) < 1e-6

    def test_metadata_contains_required_fields(self, tmp_path: Path) -> None:
        dim = 32
        np.random.seed(42)
        X = np.random.randn(40, dim)
        y = ["empty"] * 20 + ["carrying"] * 20

        c = create_hand_classifier(random_seed=42)
        c.train(X, y)

        meta = c.build_metadata(
            encoder_name="mobilenet_v3_small",
            encoder_version="v1",
            training_record_counts={"empty": 20, "carrying": 20},
            train_split_count=40,
            val_split_count=10,
        )
        d = meta.to_dict()
        assert "class_names" in d
        assert "embedding_dim" in d
        assert d["embedding_dim"] == dim
        assert d["crop_type"] == "hand"
