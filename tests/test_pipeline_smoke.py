"""Smoke tests for the task_16 pipeline orchestrator.

These exercise the *orchestration* paths (no-person early completion, canonical
output, resumability, and the atomic meta-last invariant) using lightweight
injected stages and tiny synthetic fixtures, so they run in CI without models.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pickup_putdown.config import AppConfig, load_config
from pickup_putdown.pipeline import (
    Stage,
    StageContext,
    StageResult,
    TriageStage,
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
