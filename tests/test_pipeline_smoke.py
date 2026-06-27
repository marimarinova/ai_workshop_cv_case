"""Smoke tests for the task_16 pipeline orchestrator.

These exercise the *orchestration* paths (no-person early completion, canonical
output, resumability, and the atomic meta-last invariant) using lightweight
injected stages and tiny synthetic fixtures, so they run in CI without models.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pickup_putdown.common.run_metadata import RunMetadata
from pickup_putdown.config import AppConfig, load_config
from pickup_putdown.pipeline import (
    CANONICAL_EVENT_COLUMNS,
    EvaluateStage,
    Stage,
    StageContext,
    StageResult,
    TrackAStage,
    TriageStage,
    _stage_input_hash,
    run_pipeline,
    validate_events_csv,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures (tiny stand-in "videos"; injected stages do not decode)
# ---------------------------------------------------------------------------
@pytest.fixture
def app_config() -> AppConfig:
    return load_config(None)


@pytest.fixture
def video_person(tmp_path: Path) -> Path:
    path = tmp_path / "clip_person.mp4"
    path.write_bytes(b"SYNTHETIC-PERSON-0001")
    return path


@pytest.fixture
def video_no_person(tmp_path: Path) -> Path:
    path = tmp_path / "clip_no_person.mp4"
    path.write_bytes(b"SYNTHETIC-NOPERSON-0001")
    return path


# ---------------------------------------------------------------------------
# Injected fake stages
# ---------------------------------------------------------------------------
class FakeTriage:
    name = "triage"
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ("clips.txt",)

    def __init__(self, calls: dict[str, int]) -> None:
        self._calls = calls

    def is_available(self) -> bool:
        return True

    def run(self, ctx: StageContext) -> StageResult:
        self._calls["triage"] = self._calls.get("triage", 0) + 1
        (ctx.stage_dir / "clips.txt").write_text("ok", encoding="utf-8")
        has_person = "no_person" not in ctx.video_path.stem
        return StageResult(
            name=self.name,
            status="ok" if has_person else "no_person",
            outputs={"clips.txt": str(ctx.stage_dir / "clips.txt")},
            summary={"has_person": has_person},
        )


class FakePropose:
    name = "propose"
    inputs: tuple[str, ...] = ("triage",)
    outputs: tuple[str, ...] = ("candidates.txt",)

    def __init__(self, calls: dict[str, int]) -> None:
        self._calls = calls

    def is_available(self) -> bool:
        return True

    def run(self, ctx: StageContext) -> StageResult:
        self._calls["propose"] = self._calls.get("propose", 0) + 1
        (ctx.stage_dir / "candidates.txt").write_text("c", encoding="utf-8")
        return StageResult(name=self.name, status="ok")


class FakeDetector:
    name = "track_a"
    inputs: tuple[str, ...] = ("propose",)
    outputs: tuple[str, ...] = ()

    def is_available(self) -> bool:
        return True

    def run(self, ctx: StageContext) -> StageResult:
        prediction = {
            "clip_id": ctx.clip_id,
            "pred_id": "p1",
            "type": "pickup",
            "t_start": 1.0,
            "t_end": 2.0,
            "score": 0.9,
            "model": "fake-track-a",
        }
        return StageResult(name=self.name, status="ok", summary={"predictions": [prediction]})


class FakeEvaluate:
    name = "evaluate"
    inputs: tuple[str, ...] = ("track_a",)
    outputs: tuple[str, ...] = ()

    def is_available(self) -> bool:
        return True

    def run(self, ctx: StageContext) -> StageResult:
        n = sum(len(r.summary.get("predictions", [])) for r in ctx.upstream.values())
        return StageResult(name=self.name, status="ok", summary={"n_predictions": n})


class FailingPropose:
    name = "propose"
    inputs: tuple[str, ...] = ("triage",)
    outputs: tuple[str, ...] = ()

    def is_available(self) -> bool:
        return True

    def run(self, ctx: StageContext) -> StageResult:
        (ctx.stage_dir / "half-written.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("boom")


class UnavailablePropose:
    name = "propose"
    inputs: tuple[str, ...] = ("triage",)
    outputs: tuple[str, ...] = ()

    def is_available(self) -> bool:
        return False

    def run(self, ctx: StageContext) -> StageResult:  # pragma: no cover - must not run
        raise AssertionError("unavailable stage must not run")


class FailedStatusPropose:
    """A stage that completes but reports failure (without raising)."""

    name = "propose"
    inputs: tuple[str, ...] = ("triage",)
    outputs: tuple[str, ...] = ()

    def is_available(self) -> bool:
        return True

    def run(self, ctx: StageContext) -> StageResult:
        return StageResult(name=self.name, status="failed")


def _registry(calls: dict[str, int]) -> list[Stage]:
    return [FakeTriage(calls), FakePropose(calls), FakeDetector(), FakeEvaluate()]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_person_clip_runs_full_pipeline(
    tmp_path: Path, app_config: AppConfig, video_person: Path
) -> None:
    out = tmp_path / "out"
    summary = run_pipeline(video_person, out, app_config, registry=_registry({}))

    assert summary["status"] == "ok"
    assert summary["n_events"] == 1
    assert summary["events_valid"] is True
    assert summary["stages"]["evaluate"]["status"] == "ok"

    events = out / video_person.stem / "events.csv"
    assert validate_events_csv(events)
    assert events.read_text(encoding="utf-8").strip().count("\n") == 1  # header + 1 row


def test_no_person_clip_completes_early(
    tmp_path: Path, app_config: AppConfig, video_no_person: Path
) -> None:
    out = tmp_path / "out"
    summary = run_pipeline(video_no_person, out, app_config, registry=_registry({}))

    assert summary["status"] == "no_person"
    assert summary["n_events"] == 0
    assert summary["events_valid"] is True
    # Downstream stages must not have run after early completion.
    assert "propose" not in summary["stages"]
    assert "evaluate" not in summary["stages"]

    events = out / video_no_person.stem / "events.csv"
    assert validate_events_csv(events)  # header only, still valid
    assert events.read_text(encoding="utf-8").strip().count("\n") == 0


def test_no_person_completes_early_on_resume(
    tmp_path: Path, app_config: AppConfig, video_no_person: Path
) -> None:
    out = tmp_path / "out"
    calls: dict[str, int] = {}

    first = run_pipeline(video_no_person, out, app_config, registry=_registry(calls))
    assert first["status"] == "no_person"
    assert calls["triage"] == 1

    # Second run resumes triage; early completion is status-driven, so the
    # cached "no_person" outcome must still short-circuit the pipeline.
    second = run_pipeline(video_no_person, out, app_config, registry=_registry(calls))
    assert calls["triage"] == 1  # resumed, not re-run
    assert second["status"] == "no_person"
    assert second["stages"]["triage"]["status"] == "no_person"
    assert "propose" not in second["stages"]


def test_resume_skips_unchanged_then_reprocesses_on_invalidation(
    tmp_path: Path, app_config: AppConfig, video_person: Path
) -> None:
    out = tmp_path / "out"
    calls: dict[str, int] = {}

    run_pipeline(video_person, out, app_config, registry=_registry(calls))
    assert calls["triage"] == 1

    # Second run with resume=True: nothing changed -> stage is not re-executed.
    summary = run_pipeline(video_person, out, app_config, registry=_registry(calls))
    assert calls["triage"] == 1
    assert summary["stages"]["triage"]["status"] == "resumed"

    # Delete the marker (simulating a crash before the marker was written):
    # the stage must be reprocessed, not skipped.
    (out / video_person.stem / "triage" / ".stage_meta.json").unlink()
    run_pipeline(video_person, out, app_config, registry=_registry(calls))
    assert calls["triage"] == 2


def test_failed_stage_leaves_no_marker_or_partial(
    tmp_path: Path, app_config: AppConfig, video_person: Path
) -> None:
    out = tmp_path / "out"
    registry: list[Stage] = [FakeTriage({}), FailingPropose()]

    with pytest.raises(RuntimeError, match="boom"):
        run_pipeline(video_person, out, app_config, registry=registry)

    clip_dir = out / video_person.stem
    # No promoted propose dir and no leftover partial dirs.
    assert not (clip_dir / "propose").exists()
    assert not any(p.name.startswith("propose.partial") for p in clip_dir.iterdir())
    # The successful upstream stage kept its marker.
    assert (clip_dir / "triage" / ".stage_meta.json").is_file()


def test_unavailable_upstream_cascades_to_downstream(
    tmp_path: Path, app_config: AppConfig, video_person: Path
) -> None:
    out = tmp_path / "out"
    # FakeDetector declares inputs=("propose",); propose is unavailable here.
    registry: list[Stage] = [FakeTriage({}), UnavailablePropose(), FakeDetector()]

    summary = run_pipeline(video_person, out, app_config, registry=registry)

    # Downstream must be gated (not run -> no KeyError), labelled unavailable,
    # and an expected-unavailable subtree must not fail the whole run.
    assert summary["stages"]["propose"]["status"] == "unavailable"
    assert summary["stages"]["track_a"]["status"] == "unavailable"
    assert summary["status"] == "ok"


def test_failed_upstream_blocks_downstream_and_top_status(
    tmp_path: Path, app_config: AppConfig, video_person: Path
) -> None:
    out = tmp_path / "out"
    registry: list[Stage] = [FakeTriage({}), FailedStatusPropose(), FakeDetector()]

    summary = run_pipeline(video_person, out, app_config, registry=registry)

    assert summary["stages"]["propose"]["status"] == "failed"
    assert summary["stages"]["track_a"]["status"] == "blocked"
    assert summary["stages"]["track_a"]["summary"]["blocked_on"] == "propose"
    # A failed/blocked stage must not let the run report success.
    assert summary["status"] == "failed"


def test_missing_video_raises_before_creating_outputs(
    tmp_path: Path, app_config: AppConfig
) -> None:
    out = tmp_path / "out"
    missing = tmp_path / "does_not_exist.mp4"

    with pytest.raises(FileNotFoundError, match="video not found"):
        run_pipeline(missing, out, app_config, registry=_registry({}))

    # The bad path must not leave an empty output directory behind.
    assert not (out / missing.stem).exists()


def test_cache_dir_is_shared_across_clips(
    tmp_path: Path,
    app_config: AppConfig,
    video_person: Path,
    video_no_person: Path,
) -> None:
    cache_root = tmp_path / "shared_cache"
    config = app_config.model_copy(update={"cache_dir": str(cache_root)})
    seen: list[Path] = []

    class CaptureTriage(FakeTriage):
        def run(self, ctx: StageContext) -> StageResult:
            seen.append(ctx.cache_dir)
            return super().run(ctx)

    out = tmp_path / "out"
    run_pipeline(video_person, out, config, registry=[CaptureTriage({})])
    run_pipeline(video_no_person, out, config, registry=[CaptureTriage({})])

    # The cache root is derived from config and shared across clips/runs.
    assert cache_root.is_dir()
    assert seen == [cache_root, cache_root]


def test_run_pipeline_materialises_resolved_config(
    tmp_path: Path, app_config: AppConfig, video_person: Path
) -> None:
    out = tmp_path / "out"
    run_pipeline(video_person, out, app_config, registry=_registry({}))

    resolved = out / video_person.stem / "resolved_config.yaml"
    assert resolved.is_file()
    # The materialised file must round-trip back into an AppConfig so CLI-based
    # stages can load it and run under the exact same configuration.
    assert load_config(resolved).model_dump() == app_config.model_dump()


def test_cli_stage_invoke_forwards_resolved_config(
    monkeypatch: pytest.MonkeyPatch, app_config: AppConfig
) -> None:
    captured: dict[str, list[str]] = {}

    class _FakeCompleted:
        returncode = 0

    def _fake_run(argv: list[str], **kwargs: object) -> _FakeCompleted:
        captured["argv"] = argv
        return _FakeCompleted()

    monkeypatch.setattr("pickup_putdown.pipeline.subprocess.run", _fake_run)

    stage = TriageStage(app_config)
    stage._invoke(["triage", "clip.mp4"], Path("cfg.yaml"))

    argv = captured["argv"]
    assert "--config" in argv
    assert argv[argv.index("--config") + 1] == str(Path("cfg.yaml"))


def test_events_csv_schema_rejects_bad_rows(tmp_path: Path) -> None:
    good = tmp_path / "good.csv"
    good.write_text(
        "clip_id,pred_id,type,t_start,t_end,score,model\nclip_x,p1,pickup,1.0,2.0,0.5,m\n",
        encoding="utf-8",
    )
    assert validate_events_csv(good)

    bad_score = tmp_path / "bad.csv"
    bad_score.write_text(
        "clip_id,pred_id,type,t_start,t_end,score,model\nclip_x,p1,pickup,1.0,2.0,9.9,m\n",
        encoding="utf-8",
    )
    assert not validate_events_csv(bad_score)


def test_events_csv_missing_file_is_invalid_not_raising(tmp_path: Path) -> None:
    # A missing/unreadable file must read as invalid rather than raise OSError.
    assert validate_events_csv(tmp_path / "nope.csv") is False


# ---------------------------------------------------------------------------
# Track A stage (task_10 wiring, gated on trained checkpoints)
# ---------------------------------------------------------------------------
def _config_with_artifacts(app_config: AppConfig, artifact_dir: Path) -> AppConfig:
    return app_config.model_copy(
        update={
            "track_a_stage": app_config.track_a_stage.model_copy(
                update={"artifact_dir": str(artifact_dir)}
            )
        }
    )


def _track_a_ctx(
    clip_id: str, video: Path, stage_dir: Path, cache_dir: Path, propose: StageResult
) -> StageContext:
    return StageContext(
        clip_id=clip_id,
        video_path=video,
        output_dir=stage_dir.parent,
        stage_dir=stage_dir,
        cache_dir=cache_dir,
        config={},
        config_path=stage_dir.parent / "resolved_config.yaml",
        run_metadata=RunMetadata(),
        upstream={"propose": propose},
    )


def test_track_a_unavailable_without_checkpoints(
    tmp_path: Path, app_config: AppConfig, video_person: Path
) -> None:
    config = _config_with_artifacts(app_config, tmp_path / "no_artifacts")
    stage = TrackAStage(config)
    assert stage.is_available() is False

    out = tmp_path / "out"
    registry: list[Stage] = [FakeTriage({}), FakePropose({}), stage, FakeEvaluate()]
    summary = run_pipeline(video_person, out, config, registry=registry)

    # Gated off cleanly: the missing checkpoints make the stage *unavailable*
    # (not failed/blocked), so the run still reports success with an empty file.
    assert summary["stages"]["track_a"]["status"] == "unavailable"
    assert summary["status"] == "ok"
    events = out / video_person.stem / "events.csv"
    assert validate_events_csv(events)
    assert events.read_text(encoding="utf-8").strip().count("\n") == 0  # header only


def test_track_a_runs_and_adapts_predictions(
    tmp_path: Path, app_config: AppConfig, video_person: Path
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "hand_state.joblib").write_bytes(b"hand")
    (artifacts / "shelf_state.joblib").write_bytes(b"shelf")
    config = _config_with_artifacts(app_config, artifacts)
    stage = TrackAStage(config)
    assert stage.is_available() is True

    captured: dict[str, object] = {}

    def fake_invoke(args: list[str], config_path: Path) -> None:
        captured["args"] = args
        captured["config_path"] = config_path
        out_dir = Path(args[args.index("--output-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        # infer-track-a writes a superset predictions.csv: 7 canonical + 3 extra.
        (out_dir / "predictions.csv").write_text(
            "clip_id,pred_id,type,t_start,t_end,score,model,actor_id,hand_side,region_id\n"
            "clip_x,c0-trackA-0,pickup,1.0,2.0,0.8,track_a,a1,left,shelf_1\n",
            encoding="utf-8",
        )

    stage._invoke = fake_invoke  # type: ignore[method-assign]

    stage_dir = tmp_path / "out" / "clip_x" / "track_a"
    propose = StageResult(
        name="propose",
        status="ok",
        outputs={"candidates.parquet": "C.parquet", "tracks_pose.parquet": "P.parquet"},
    )
    ctx = _track_a_ctx("clip_x", video_person, stage_dir, tmp_path / "cache", propose)

    result = stage.run(ctx)

    args = captured["args"]
    assert isinstance(args, list)
    assert args[0] == "infer-track-a"
    assert args[args.index("--candidates") + 1] == "C.parquet"
    assert args[args.index("--pose-observations") + 1] == "P.parquet"
    assert args[args.index("--source-video-dir") + 1] == str(video_person.parent)
    assert args[args.index("--artifact-dir") + 1] == str(artifacts)
    assert args[args.index("--output-dir") + 1] == str(stage_dir)
    assert "--clip-id" not in args  # omitted: per-clip parquet needs no filter
    # The Track A YAML (not the resolved AppConfig) is forwarded as --config.
    assert captured["config_path"] == Path(config.track_a_stage.config_path)

    preds = result.summary["predictions"]
    assert len(preds) == 1
    row = preds[0]
    assert row["type"] in ("pickup", "putdown")
    assert 0.0 <= float(row["score"]) <= 1.0
    assert row["actor_id"] == "a1"  # extra column carried through
    # The canonical projection keeps exactly the 7 columns; extras are dropped.
    canonical = {key: row.get(key, "") for key in CANONICAL_EVENT_COLUMNS}
    assert tuple(canonical) == CANONICAL_EVENT_COLUMNS


def test_track_a_resume_key_changes_with_checkpoint(
    tmp_path: Path, app_config: AppConfig, video_person: Path
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "hand_state.joblib").write_bytes(b"v1")
    (artifacts / "shelf_state.joblib").write_bytes(b"v1")
    config = _config_with_artifacts(app_config, artifacts)
    stage = TrackAStage(config)

    propose = StageResult(name="propose", status="ok", input_hash="abc")
    ctx = _track_a_ctx("clip_x", video_person, tmp_path / "s", tmp_path / "c", propose)

    before = _stage_input_hash(stage, ctx)
    # Retraining (Task 7) replaces the checkpoint content: the resume key must
    # change so the cached, stale prediction is not served.
    (artifacts / "hand_state.joblib").write_bytes(b"v2-retrained")
    after = _stage_input_hash(stage, ctx)

    assert before != after


def test_track_a_propagates_invoke_failure(
    tmp_path: Path,
    app_config: AppConfig,
    video_person: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "hand_state.joblib").write_bytes(b"hand")
    (artifacts / "shelf_state.joblib").write_bytes(b"shelf")
    config = _config_with_artifacts(app_config, artifacts)
    stage = TrackAStage(config)

    class _NonZero:
        returncode = 1

    monkeypatch.setattr(
        "pickup_putdown.pipeline.subprocess.run",
        lambda argv, **kwargs: _NonZero(),
    )

    stage_dir = tmp_path / "out" / "clip_x" / "track_a"
    stage_dir.mkdir(parents=True)
    propose = StageResult(
        name="propose",
        status="ok",
        outputs={"candidates.parquet": "C.parquet", "tracks_pose.parquet": "P.parquet"},
    )
    ctx = _track_a_ctx("clip_x", video_person, stage_dir, tmp_path / "cache", propose)

    # A non-zero infer-track-a exit (e.g. no candidates) surfaces as a
    # RuntimeError, which the orchestrator turns into a failed clip — the current
    # degrade-to-failure semantics, to be revisited at activation.
    with pytest.raises(RuntimeError, match="track_a"):
        stage.run(ctx)


# ---------------------------------------------------------------------------
# EvaluateStage -> Task 8 evaluator (gated on ground truth)
# ---------------------------------------------------------------------------
def _config_with_ground_truth(app_config: AppConfig, ground_truth_dir: Path) -> AppConfig:
    return app_config.model_copy(
        update={
            "evaluation": app_config.evaluation.model_copy(
                update={"ground_truth_dir": str(ground_truth_dir)}
            )
        }
    )


def _evaluate_ctx(clip_id: str, stage_dir: Path, upstream: dict[str, StageResult]) -> StageContext:
    # EvaluateStage reads only clip_id + upstream; video_path/cache_dir are unused.
    return StageContext(
        clip_id=clip_id,
        video_path=stage_dir,
        output_dir=stage_dir.parent,
        stage_dir=stage_dir,
        cache_dir=stage_dir.parent,
        config={},
        config_path=stage_dir.parent / "resolved_config.yaml",
        run_metadata=RunMetadata(),
        upstream=upstream,
    )


def _detector_result(clip_id: str) -> StageResult:
    return StageResult(
        name="track_a",
        status="ok",
        summary={
            "predictions": [
                {
                    "clip_id": clip_id,
                    "pred_id": "p1",
                    "type": "pickup",
                    "t_start": "1.0",
                    "t_end": "2.0",
                    "score": "0.9",
                    "model": "m",
                }
            ]
        },
    )


def test_evaluate_scores_against_ground_truth(tmp_path: Path, app_config: AppConfig) -> None:
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    (gt_dir / "clipX.csv").write_text(
        "clip_id,type,t_start,t_end\nclipX,pickup,1.0,2.0\n", encoding="utf-8"
    )
    stage = EvaluateStage(_config_with_ground_truth(app_config, gt_dir))
    stage_dir = tmp_path / "out" / "clipX" / "evaluate"
    stage_dir.mkdir(parents=True)
    ctx = _evaluate_ctx("clipX", stage_dir, {"track_a": _detector_result("clipX")})

    result = stage.run(ctx)

    assert result.summary["evaluated"] is True
    assert result.summary["n_predictions"] == 1
    # A perfectly overlapping pickup is a true positive at tIoU 0.5.
    assert result.summary["tiou@0.5"]["tp"] == 1
    assert result.summary["tiou@0.5"]["fp"] == 0
    assert result.summary["tiou@0.5"]["fn"] == 0
    written = json.loads((stage_dir / "metrics.json").read_text(encoding="utf-8"))
    assert written["evaluated"] is True
    assert written["tiou@0.5"]["tp"] == 1


def test_evaluate_without_ground_truth_stays_stub(tmp_path: Path, app_config: AppConfig) -> None:
    stage = EvaluateStage(app_config)  # default ground_truth_dir == ""
    stage_dir = tmp_path / "out" / "clipX" / "evaluate"
    stage_dir.mkdir(parents=True)
    ctx = _evaluate_ctx("clipX", stage_dir, {"track_a": _detector_result("clipX")})

    result = stage.run(ctx)

    # The pre-Task-7 contract is unchanged: a stable stub, no scoring keys.
    assert result.summary == {"n_predictions": 1, "evaluated": False}
    assert "tiou@0.5" not in result.summary
    written = json.loads((stage_dir / "metrics.json").read_text(encoding="utf-8"))
    assert written == {"n_predictions": 1, "evaluated": False}


def test_evaluate_ground_truth_dir_without_clip_file_stays_stub(
    tmp_path: Path, app_config: AppConfig
) -> None:
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()  # configured but holds no <clip_id>.csv
    stage = EvaluateStage(_config_with_ground_truth(app_config, gt_dir))
    stage_dir = tmp_path / "out" / "clipX" / "evaluate"
    stage_dir.mkdir(parents=True)
    ctx = _evaluate_ctx("clipX", stage_dir, {})

    result = stage.run(ctx)

    assert result.summary == {"n_predictions": 0, "evaluated": False}
