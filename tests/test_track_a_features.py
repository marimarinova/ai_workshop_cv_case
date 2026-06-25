"""Tests for the TEMPORARY label-free feature wrapper (task_10).

Task 9's crop extraction and the encoder are mocked here — no real video is
decoded — so these exercise only the wrapper's own pairing/orchestration logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pytest

from pickup_putdown.common.schemas import Candidate, PoseObservation
from pickup_putdown.config import TrackAFeaturesConfig
from pickup_putdown.layer1.track_a import features
from pickup_putdown.layer1.track_a.contracts import CropGeometry, CropRecord
from pickup_putdown.layer1.track_a.features import (
    extract_crops_for_inference,
    pair_and_embed,
)
from pickup_putdown.layer1.track_a.sampling import SamplePoint
from pickup_putdown.layer1.track_a.state_types import Embedding

if TYPE_CHECKING:
    from pickup_putdown.perception.shelf_regions import Polygon


def _crop(ts: float, position: str, crop_type: Literal["hand", "shelf"]) -> CropRecord:
    return CropRecord(
        crop_id=f"{crop_type}-{ts}",
        clip_id="clip_x",
        candidate_id="cand1",
        timestamp_s=ts,
        sample_position=position,
        crop_type=crop_type,
        geometry=CropGeometry(0, 0, 4, 4),
    )


def _fake_embed(crop: CropRecord) -> Embedding:
    # Encode the timestamp so the test can tell hand/shelf embeddings apart.
    base = crop.timestamp_s + (0.5 if crop.crop_type == "shelf" else 0.0)
    return np.full(4, base, dtype=np.float32)


def test_pair_and_embed_pairs_hand_and_shelf_in_time_order() -> None:
    crops = [
        _crop(2.0, "contact", "shelf"),
        _crop(1.0, "pre", "hand"),
        _crop(2.0, "contact", "hand"),
        _crop(1.0, "pre", "shelf"),
    ]
    feats = pair_and_embed(crops, _fake_embed)

    assert [(f.timestamp_s, f.sample_position) for f in feats] == [(1.0, "pre"), (2.0, "contact")]
    assert feats[0].hand_embedding[0] == 1.0
    assert feats[0].shelf_embedding[0] == 1.5  # shelf offset applied


def test_pair_and_embed_drops_unpaired_samples() -> None:
    crops = [_crop(1.0, "pre", "hand")]  # no shelf crop at this sample
    assert pair_and_embed(crops, _fake_embed) == []


def _candidate() -> Candidate:
    return Candidate(
        candidate_id="cand1",
        clip_id="clip_x",
        actor_id="a1",
        hand_side="left",
        region_id="shelf1",
        raw_start_s=0.0,
        raw_end_s=4.0,
        window_start_s=0.0,
        window_end_s=4.0,
    )


def _pose(ts: float) -> PoseObservation:
    return PoseObservation(
        clip_id="clip_x",
        timestamp_s=ts,
        sample_index=0,
        actor_id="a1",
        hand_side="left",
        wrist_x=1.0,
        wrist_y=1.0,
        wrist_confidence=0.9,
    )


def test_extract_crops_for_inference_composes_task9_mocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    # Mock the Task 9 sampling + crop extraction — no real geometry or decoding.
    monkeypatch.setattr(features, "get_wrist_trajectory_for_candidate", lambda c, p: list(p))
    monkeypatch.setattr(features, "get_contact_time", lambda c, t, s: 2.0)
    monkeypatch.setattr(
        features, "compute_sample_times", lambda a, b, c, cfg: [SamplePoint(2.0, "contact", 0)]
    )

    def fake_extractor(video, candidate, samples, pose, shelf, cfg):  # type: ignore[no-untyped-def]
        captured["samples"] = samples
        return [_crop(2.0, "contact", "hand"), _crop(2.0, "contact", "shelf")]

    shelf = cast("Polygon", object())
    crops = extract_crops_for_inference(
        tmp_path / "clip.mp4",
        _candidate(),
        [_pose(2.0)],
        shelf,
        TrackAFeaturesConfig(),
        crop_extract_fn=fake_extractor,
    )

    assert [c.crop_type for c in crops] == ["hand", "shelf"]
    assert captured["samples"] == [SamplePoint(2.0, "contact", 0)]


def test_extract_crops_for_inference_returns_empty_without_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(features, "get_wrist_trajectory_for_candidate", lambda c, p: [])
    shelf = cast("Polygon", object())
    crops = extract_crops_for_inference(
        tmp_path / "clip.mp4",
        _candidate(),
        [],
        shelf,
        TrackAFeaturesConfig(),
        crop_extract_fn=lambda *a: [],
    )
    assert crops == []
