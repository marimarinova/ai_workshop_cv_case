"""DEFERRED Task 16 orchestrator adapter for Track A (task_10) — NOT WIRED.

The Task 16 orchestrator (``pickup_putdown.pipeline``) is not in master yet, so
this adapter is intentionally standalone and imported by nobody. It deliberately
does **not** import the missing orchestrator; instead it depends on a structural
``StageContextLike`` Protocol describing only the fields Track A needs.

When Task 16 lands, wire it in two steps with no change here:
1. Register ``TrackAStage`` in ``build_default_registry`` (replacing
   ``ComponentStub("track_a")``). ``pipeline.StageContext`` structurally
   satisfies ``StageContextLike``.
2. Wrap :meth:`TrackAStage.run`'s dict into a ``pipeline.StageResult`` — the
   ``predictions`` rows feed the canonical ``events.csv`` aggregator.

The default candidate loader / feature function are deferred (they depend on the
final propose-output contract); inject them — as the tests do — until then.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pickup_putdown.config import AppConfig
from pickup_putdown.layer1.track_a import hand_state, shelf_state
from pickup_putdown.layer1.track_a.hand_state import load_hand_state_classifier
from pickup_putdown.layer1.track_a.inference import CandidateInput, FeatureFn, infer_track_a
from pickup_putdown.layer1.track_a.shelf_state import load_shelf_state_classifier
from pickup_putdown.layer1.track_a.state_types import HandStateClassifier, ShelfStateClassifier

#: Builds Track A candidate inputs from a stage context (deferred default).
CandidateLoader = Callable[["StageContextLike"], "list[CandidateInput]"]


@runtime_checkable
class StageContextLike(Protocol):
    """The subset of Task 16's StageContext that Track A consumes."""

    clip_id: str
    cache_dir: Path
    config: dict[str, Any]
    upstream: Mapping[str, Any]


class TrackAStage:
    """Track A as a Task 16 stage. Deferred: not registered anywhere yet."""

    name = "track_a"
    inputs: tuple[str, ...] = ("propose",)
    outputs: tuple[str, ...] = ("events.csv",)

    def __init__(
        self,
        config: AppConfig,
        *,
        candidate_loader: CandidateLoader | None = None,
        hand_classifier: HandStateClassifier | None = None,
        shelf_classifier: ShelfStateClassifier | None = None,
        feature_fn: FeatureFn | None = None,
    ) -> None:
        self._config = config
        self._candidate_loader = candidate_loader
        self._hand_classifier = hand_classifier
        self._shelf_classifier = shelf_classifier
        self._feature_fn = feature_fn

    def is_available(self) -> bool:
        """True only when both trained classifier checkpoints exist (task_7)."""
        return hand_state.is_available(self._config.track_a) and shelf_state.is_available(
            self._config.track_a
        )

    def run(self, ctx: StageContextLike) -> dict[str, Any]:
        """Run Track A and return a JSON-able summary with canonical rows."""
        if self._candidate_loader is None or self._feature_fn is None:
            raise NotImplementedError(
                "TrackAStage default wiring is deferred until Task 16 lands; inject "
                "candidate_loader and feature_fn (see module docstring)."
            )
        # Defensive availability gate: if a classifier was not explicitly
        # injected and no trained checkpoint exists, do NOT silently fall back to
        # placeholder estimators — their output must never surface as a real
        # prediction. Explicit injection (e.g. tests) bypasses this.
        needs_load = self._hand_classifier is None or self._shelf_classifier is None
        if needs_load and not self.is_available():
            return {
                "status": "unavailable",
                "n_events": 0,
                "predictions": [],
                "reason": (
                    "no trained Track A checkpoints (task_7 pending); refusing to "
                    "emit placeholder predictions"
                ),
            }

        candidate_inputs = self._candidate_loader(ctx)
        hand = self._hand_classifier or load_hand_state_classifier(self._config.track_a)
        shelf = self._shelf_classifier or load_shelf_state_classifier(self._config.track_a)
        predictions = infer_track_a(
            ctx.clip_id,
            candidate_inputs,
            self._config.track_a,
            hand_classifier=hand,
            shelf_classifier=shelf,
            feature_fn=self._feature_fn,
        )
        return {
            "status": "ok",
            "n_events": len(predictions),
            "predictions": [p.to_canonical_row() for p in predictions],
        }
