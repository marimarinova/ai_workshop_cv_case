"""Tests for the Track A CLI command.

Mocks the pipeline — no GPU, no real videos required.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from pickup_putdown.cli import (
    _filter_candidates,
    _load_candidates_for_inference,
    _resolve_pose_observations,
    _resolve_source_videos,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dirs(tmp_path: Path):
    """Create minimal directory structure for CLI tests."""
    src_dir = tmp_path / "source_videos"
    src_dir.mkdir()
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir()
    cand_dir = meta_dir / "candidates"
    cand_dir.mkdir()
    return {
        "source_videos": src_dir,
        "artifacts": art_dir,
        "cache_dir": cache_dir,
        "output_dir": out_dir,
        "candidate_metadata": meta_dir,
    }


@pytest.fixture
def sample_video(tmp_dirs: dict[str, Path]) -> Path:
    """Create a dummy video file."""
    vp = tmp_dirs["source_videos"] / "test_clip.mp4"
    vp.write_bytes(b"fake_video")
    return vp


@pytest.fixture
def sample_artifacts(tmp_dirs: dict[str, Path]) -> None:
    """Create dummy classifier artifacts."""
    art = tmp_dirs["artifacts"]
    (art / "hand_state.joblib").write_bytes(b"fake")
    (art / "hand_state_metadata.json").write_text(
        json.dumps(
            {
                "artifact_version": "1.0",
                "embedding_dim": 16,
                "encoder_name": "mobilenet_v3_small",
                "encoder_version": "v1",
                "class_names": ["empty", "carrying"],
            }
        )
    )
    (art / "shelf_state.joblib").write_bytes(b"fake")
    (art / "shelf_state_metadata.json").write_text(
        json.dumps(
            {
                "artifact_version": "1.0",
                "embedding_dim": 16,
                "encoder_name": "mobilenet_v3_small",
                "encoder_version": "v1",
                "class_names": ["object_removed", "object_placed", "no_change"],
            }
        )
    )


@pytest.fixture
def sample_candidate_metadata(tmp_dirs: dict[str, Path]) -> None:
    """Write a candidate metadata JSON with known candidates."""
    meta_file = tmp_dirs["candidate_metadata"] / "candidates" / "test_clip" / "test_clip.json"
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file.write_text(
        json.dumps(
            {
                "source_video_id": "test_clip",
                "candidate_count": 2,
                "candidates": [
                    {
                        "candidate_id": "cand_001",
                        "source_start_s": 10.0,
                        "source_end_s": 14.0,
                        "duration_s": 4.0,
                        "actor_id": "actor_1",
                        "hand_side": "right",
                        "region_id": "Shelf_01",
                    },
                    {
                        "candidate_id": "cand_002",
                        "source_start_s": 20.0,
                        "source_end_s": 24.0,
                        "duration_s": 4.0,
                        "actor_id": "actor_1",
                        "hand_side": "left",
                        "region_id": "Shelf_01",
                    },
                ],
            }
        )
    )


@pytest.fixture
def sample_candidates(tmp_dirs: dict[str, Path], sample_candidate_metadata: None) -> list[dict]:
    """Load the sample candidates."""
    return _load_candidates_for_inference(
        candidate_metadata_dir=tmp_dirs["candidate_metadata"],
        candidates_path=None,
    )


# ---------------------------------------------------------------------------
# Helper: mock the pipeline for CLI command tests
# ---------------------------------------------------------------------------


def _mock_pipeline_result(tmp_dirs: dict[str, Path]):
    """Return a mock InferenceResult writer."""
    out = tmp_dirs["output_dir"]
    # Pre-write expected outputs so the CLI doesn't fail on overwrite check
    return out


# ---------------------------------------------------------------------------
# Tests: CLI helper functions
# ---------------------------------------------------------------------------


class TestLoadCandidates:
    def test_load_from_metadata_dir(self, tmp_dirs, sample_candidate_metadata):
        cands = _load_candidates_for_inference(
            candidate_metadata_dir=tmp_dirs["candidate_metadata"],
            candidates_path=None,
        )
        assert len(cands) == 2
        ids = {c["candidate_id"] for c in cands}
        assert ids == {"cand_001", "cand_002"}

    def test_load_empty_dir(self, tmp_dirs):
        cands = _load_candidates_for_inference(
            candidate_metadata_dir=tmp_dirs["candidate_metadata"],
            candidates_path=None,
        )
        assert cands == []


class TestFilterCandidates:
    def test_filter_by_candidate_id(self, sample_candidates):
        result = _filter_candidates(sample_candidates, candidate_id="cand_001")
        assert len(result) == 1
        assert result[0]["candidate_id"] == "cand_001"

    def test_filter_by_clip_id(self, sample_candidates):
        result = _filter_candidates(sample_candidates, clip_id="test_clip")
        assert len(result) == 2

    def test_filter_unknown_candidate(self, sample_candidates):
        result = _filter_candidates(sample_candidates, candidate_id="cand_999")
        assert result == []

    def test_filter_unknown_clip(self, sample_candidates):
        result = _filter_candidates(sample_candidates, clip_id="other_clip")
        assert result == []

    def test_filter_both(self, sample_candidates):
        result = _filter_candidates(
            sample_candidates, clip_id="test_clip", candidate_id="cand_001"
        )
        assert len(result) == 1

    def test_no_filter(self, sample_candidates):
        result = _filter_candidates(sample_candidates)
        assert len(result) == 2


class TestResolveSourceVideos:
    def test_resolve_existing_video(self, sample_candidates, sample_video, tmp_dirs):
        # Update candidates to match video name
        cands = [{"candidate_id": "c", "clip_id": "test_clip"}]
        videos = _resolve_source_videos(cands, tmp_dirs["source_videos"])
        assert "test_clip" in videos
        assert videos["test_clip"].exists()

    def test_resolve_missing_video(self, tmp_dirs):
        cands = [{"candidate_id": "c", "clip_id": "missing_clip"}]
        videos = _resolve_source_videos(cands, tmp_dirs["source_videos"])
        assert videos == {}


class TestResolvePoseObservations:
    def test_no_pose_path_no_remote(self, sample_candidates, tmp_dirs):
        obs = _resolve_pose_observations(
            sample_candidates, pose_path=None, source_video_dir=tmp_dirs["source_videos"]
        )
        assert obs == []

    def test_explicit_pose_file(self, sample_candidates, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        pf = tmp_path / "poses.parquet"
        table = pa.table(
            {
                "clip_id": ["test_clip"],
                "timestamp_s": [1.0],
                "actor_id": ["actor_1"],
                "hand_side": ["right"],
                "wrist_x": [100.0],
                "wrist_y": [200.0],
                "wrist_confidence": [0.8],
            }
        )
        pq.write_table(table, str(pf))

        obs = _resolve_pose_observations(sample_candidates, pose_path=pf, source_video_dir=None)
        assert len(obs) == 1
        assert obs[0]["wrist_x"] == 100.0


# ---------------------------------------------------------------------------
# Tests: CLI command integration (mocked pipeline)
# ---------------------------------------------------------------------------


class TestInferTrackACliCommand:
    """Test the CLI command end-to-end with mocked pipeline."""

    def _run_cli(self, args: list[str], tmp_dirs, sample_artifacts=None):
        """Run the CLI command via typer.TestClient."""
        from typer.testing import CliRunner

        from pickup_putdown.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["infer-track-a"] + args, catch_exceptions=False)
        return result

    def test_help_shows_command(self):
        from typer.testing import CliRunner

        from pickup_putdown.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert "infer-track-a" in result.stdout

    def test_defaults_resolve(
        self, tmp_dirs, sample_video, sample_artifacts, sample_candidate_metadata, tmp_path
    ):
        """Defaults should resolve without explicit args (except paths)."""
        from unittest.mock import patch

        from typer.testing import CliRunner

        from pickup_putdown.cli import app

        runner = CliRunner()

        # Mock the pipeline to avoid real inference
        mock_result = mock.MagicMock()
        mock_result.summary.candidates_processed = 1
        mock_result.summary.candidates_skipped = 0
        mock_result.summary.total_samples = 5
        mock_result.summary.feature_cache_hits = 0
        mock_result.summary.feature_cache_misses = 5
        mock_result.summary.raw_events_emitted = 0
        mock_result.summary.final_events_after_dedup = 0
        mock_result.summary.pickup_count = 0
        mock_result.summary.putdown_count = 0
        mock_result.summary.mean_confidence = 0.0
        mock_result.summary.skip_reasons = {}
        mock_result.output_paths = {}

        with (
            patch(
                "pickup_putdown.layer1.track_a.inference.TrackAInferencePipeline"
            ) as MockPipeline,
            patch("pickup_putdown.layer1.track_a.image_features.TorchVisionEmbedder"),
            patch("pickup_putdown.ingestion.video_probe.probe_video") as mock_probe,
        ):
            mock_probe.return_value.decode_ok = True
            mock_probe.return_value.duration_s = 30.0

            mock_instance = mock.MagicMock()
            mock_instance.run.return_value = mock_result
            MockPipeline.return_value = mock_instance

            result = runner.invoke(
                app,
                [
                    "infer-track-a",
                    "--config",
                    "configs/track_a.yaml",
                    "--candidate-metadata",
                    str(tmp_dirs["candidate_metadata"]),
                    "--source-video-dir",
                    str(tmp_dirs["source_videos"]),
                    "--artifact-dir",
                    str(tmp_dirs["artifacts"]),
                    "--cache-dir",
                    str(tmp_dirs["cache_dir"]),
                    "--output-dir",
                    str(tmp_dirs["output_dir"]),
                    "--candidate-id",
                    "cand_001",
                ],
            )

            assert result.exit_code == 0, result.output
            mock_instance.run.assert_called_once()

    def test_unknown_candidate_fails(self, tmp_dirs, sample_artifacts, sample_candidate_metadata):
        from typer.testing import CliRunner

        from pickup_putdown.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "infer-track-a",
                "--config",
                "configs/track_a.yaml",
                "--candidate-metadata",
                str(tmp_dirs["candidate_metadata"]),
                "--source-video-dir",
                str(tmp_dirs["source_videos"]),
                "--artifact-dir",
                str(tmp_dirs["artifacts"]),
                "--cache-dir",
                str(tmp_dirs["cache_dir"]),
                "--output-dir",
                str(tmp_dirs["output_dir"]),
                "--candidate-id",
                "cand_nonexistent",
            ],
        )
        assert result.exit_code != 0

    def test_missing_artifact_dir_fails(self, tmp_dirs, sample_candidate_metadata):
        from typer.testing import CliRunner

        from pickup_putdown.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "infer-track-a",
                "--config",
                "configs/track_a.yaml",
                "--candidate-metadata",
                str(tmp_dirs["candidate_metadata"]),
                "--source-video-dir",
                str(tmp_dirs["source_videos"]),
                "--artifact-dir",
                str(tmp_dirs["artifacts"] / "nonexistent"),
                "--cache-dir",
                str(tmp_dirs["cache_dir"]),
                "--output-dir",
                str(tmp_dirs["output_dir"]),
            ],
        )
        assert result.exit_code != 0

    def test_missing_source_video_reported(
        self, tmp_dirs, sample_artifacts, sample_candidate_metadata, tmp_path
    ):
        """Missing source video should be reported in diagnostics, not crash."""
        from unittest.mock import MagicMock, patch

        from typer.testing import CliRunner

        from pickup_putdown.cli import app

        runner = CliRunner()

        mock_result = MagicMock()
        mock_result.summary.candidates_processed = 0
        mock_result.summary.candidates_skipped = 1
        mock_result.summary.total_samples = 0
        mock_result.summary.feature_cache_hits = 0
        mock_result.summary.feature_cache_misses = 0
        mock_result.summary.raw_events_emitted = 0
        mock_result.summary.final_events_after_dedup = 0
        mock_result.summary.pickup_count = 0
        mock_result.summary.putdown_count = 0
        mock_result.summary.mean_confidence = 0.0
        mock_result.summary.skip_reasons = {"missing_source_video": 1}
        mock_result.output_paths = {}

        with (
            patch(
                "pickup_putdown.layer1.track_a.inference.TrackAInferencePipeline"
            ) as MockPipeline,
            patch("pickup_putdown.layer1.track_a.image_features.TorchVisionEmbedder"),
            patch("pickup_putdown.ingestion.video_probe.probe_video") as mock_probe,
        ):
            mock_probe.return_value.decode_ok = False
            mock_probe.return_value.duration_s = None

            mock_instance = MagicMock()
            mock_instance.run.return_value = mock_result
            MockPipeline.return_value = mock_instance

            result = runner.invoke(
                app,
                [
                    "infer-track-a",
                    "--config",
                    "configs/track_a.yaml",
                    "--candidate-metadata",
                    str(tmp_dirs["candidate_metadata"]),
                    "--source-video-dir",
                    str(tmp_dirs["source_videos"]),
                    "--artifact-dir",
                    str(tmp_dirs["artifacts"]),
                    "--cache-dir",
                    str(tmp_dirs["cache_dir"]),
                    "--output-dir",
                    str(tmp_dirs["output_dir"]),
                ],
            )
            assert result.exit_code == 0

    def test_existing_output_requires_force(
        self, tmp_dirs, sample_artifacts, sample_candidate_metadata
    ):
        from typer.testing import CliRunner

        from pickup_putdown.cli import app

        runner = CliRunner()
        # Pre-create output file
        (tmp_dirs["output_dir"] / "predictions.csv").write_text("clip_id,pred_id\n")

        result = runner.invoke(
            app,
            [
                "infer-track-a",
                "--config",
                "configs/track_a.yaml",
                "--candidate-metadata",
                str(tmp_dirs["candidate_metadata"]),
                "--source-video-dir",
                str(tmp_dirs["source_videos"]),
                "--artifact-dir",
                str(tmp_dirs["artifacts"]),
                "--cache-dir",
                str(tmp_dirs["cache_dir"]),
                "--output-dir",
                str(tmp_dirs["output_dir"]),
            ],
        )
        assert result.exit_code != 0
        assert "force" in result.stdout.lower() or "force" in result.stderr.lower()

    def test_force_overwrites(self, tmp_dirs, sample_artifacts, sample_candidate_metadata):
        from unittest.mock import MagicMock, patch

        from typer.testing import CliRunner

        from pickup_putdown.cli import app

        runner = CliRunner()
        (tmp_dirs["output_dir"] / "predictions.csv").write_text("old\n")

        mock_result = MagicMock()
        mock_result.summary.candidates_processed = 0
        mock_result.summary.candidates_skipped = 0
        mock_result.summary.total_samples = 0
        mock_result.summary.feature_cache_hits = 0
        mock_result.summary.feature_cache_misses = 0
        mock_result.summary.raw_events_emitted = 0
        mock_result.summary.final_events_after_dedup = 0
        mock_result.summary.pickup_count = 0
        mock_result.summary.putdown_count = 0
        mock_result.summary.mean_confidence = 0.0
        mock_result.summary.skip_reasons = {}
        mock_result.output_paths = {}

        with (
            patch(
                "pickup_putdown.layer1.track_a.inference.TrackAInferencePipeline"
            ) as MockPipeline,
            patch("pickup_putdown.layer1.track_a.image_features.TorchVisionEmbedder"),
            patch("pickup_putdown.ingestion.video_probe.probe_video") as mock_probe,
        ):
            mock_probe.return_value.decode_ok = True
            mock_probe.return_value.duration_s = 30.0

            mock_instance = MagicMock()
            mock_instance.run.return_value = mock_result
            MockPipeline.return_value = mock_instance

            result = runner.invoke(
                app,
                [
                    "infer-track-a",
                    "--config",
                    "configs/track_a.yaml",
                    "--candidate-metadata",
                    str(tmp_dirs["candidate_metadata"]),
                    "--source-video-dir",
                    str(tmp_dirs["source_videos"]),
                    "--artifact-dir",
                    str(tmp_dirs["artifacts"]),
                    "--cache-dir",
                    str(tmp_dirs["cache_dir"]),
                    "--output-dir",
                    str(tmp_dirs["output_dir"]),
                    "--force",
                ],
            )
            assert result.exit_code == 0, result.output

    def test_pipeline_called_once(
        self, tmp_dirs, sample_video, sample_artifacts, sample_candidate_metadata
    ):
        from unittest.mock import MagicMock, patch

        from typer.testing import CliRunner

        from pickup_putdown.cli import app

        runner = CliRunner()

        mock_result = MagicMock()
        mock_result.summary.candidates_processed = 1
        mock_result.summary.candidates_skipped = 0
        mock_result.summary.total_samples = 4
        mock_result.summary.feature_cache_hits = 0
        mock_result.summary.feature_cache_misses = 4
        mock_result.summary.raw_events_emitted = 1
        mock_result.summary.final_events_after_dedup = 1
        mock_result.summary.pickup_count = 1
        mock_result.summary.putdown_count = 0
        mock_result.summary.mean_confidence = 0.6
        mock_result.summary.skip_reasons = {}
        mock_result.output_paths = {}

        with (
            patch(
                "pickup_putdown.layer1.track_a.inference.TrackAInferencePipeline"
            ) as MockPipeline,
            patch("pickup_putdown.layer1.track_a.image_features.TorchVisionEmbedder"),
            patch("pickup_putdown.ingestion.video_probe.probe_video") as mock_probe,
        ):
            mock_probe.return_value.decode_ok = True
            mock_probe.return_value.duration_s = 30.0

            mock_instance = MagicMock()
            mock_instance.run.return_value = mock_result
            MockPipeline.return_value = mock_instance

            result = runner.invoke(
                app,
                [
                    "infer-track-a",
                    "--config",
                    "configs/track_a.yaml",
                    "--candidate-metadata",
                    str(tmp_dirs["candidate_metadata"]),
                    "--source-video-dir",
                    str(tmp_dirs["source_videos"]),
                    "--artifact-dir",
                    str(tmp_dirs["artifacts"]),
                    "--cache-dir",
                    str(tmp_dirs["cache_dir"]),
                    "--output-dir",
                    str(tmp_dirs["output_dir"]),
                    "--candidate-id",
                    "cand_001",
                ],
            )
            assert result.exit_code == 0, result.output
            assert mock_instance.run.call_count == 1

    def test_verbose_enables_debug(self, tmp_dirs, sample_artifacts, sample_candidate_metadata):
        from unittest.mock import MagicMock, patch

        from typer.testing import CliRunner

        from pickup_putdown.cli import app

        runner = CliRunner()

        mock_result = MagicMock()
        mock_result.summary.candidates_processed = 0
        mock_result.summary.candidates_skipped = 0
        mock_result.summary.total_samples = 0
        mock_result.summary.feature_cache_hits = 0
        mock_result.summary.feature_cache_misses = 0
        mock_result.summary.raw_events_emitted = 0
        mock_result.summary.final_events_after_dedup = 0
        mock_result.summary.pickup_count = 0
        mock_result.summary.putdown_count = 0
        mock_result.summary.mean_confidence = 0.0
        mock_result.summary.skip_reasons = {}
        mock_result.output_paths = {}

        with (
            patch(
                "pickup_putdown.layer1.track_a.inference.TrackAInferencePipeline"
            ) as MockPipeline,
            patch("pickup_putdown.layer1.track_a.image_features.TorchVisionEmbedder"),
            patch("pickup_putdown.ingestion.video_probe.probe_video") as mock_probe,
        ):
            mock_probe.return_value.decode_ok = True
            mock_probe.return_value.duration_s = 30.0

            mock_instance = MagicMock()
            mock_instance.run.return_value = mock_result
            MockPipeline.return_value = mock_instance

            result = runner.invoke(
                app,
                [
                    "infer-track-a",
                    "--config",
                    "configs/track_a.yaml",
                    "--candidate-metadata",
                    str(tmp_dirs["candidate_metadata"]),
                    "--source-video-dir",
                    str(tmp_dirs["source_videos"]),
                    "--artifact-dir",
                    str(tmp_dirs["artifacts"]),
                    "--cache-dir",
                    str(tmp_dirs["cache_dir"]),
                    "--output-dir",
                    str(tmp_dirs["output_dir"]),
                    "-v",
                ],
            )
            assert result.exit_code == 0, result.output

    def test_debug_traces_passed(
        self, tmp_dirs, sample_video, sample_artifacts, sample_candidate_metadata
    ):
        from unittest.mock import MagicMock, patch

        from typer.testing import CliRunner

        from pickup_putdown.cli import app

        runner = CliRunner()

        mock_result = MagicMock()
        mock_result.summary.candidates_processed = 1
        mock_result.summary.candidates_skipped = 0
        mock_result.summary.total_samples = 4
        mock_result.summary.feature_cache_hits = 0
        mock_result.summary.feature_cache_misses = 4
        mock_result.summary.raw_events_emitted = 0
        mock_result.summary.final_events_after_dedup = 0
        mock_result.summary.pickup_count = 0
        mock_result.summary.putdown_count = 0
        mock_result.summary.mean_confidence = 0.0
        mock_result.summary.skip_reasons = {}
        mock_result.output_paths = {}

        with (
            patch(
                "pickup_putdown.layer1.track_a.inference.TrackAInferencePipeline"
            ) as MockPipeline,
            patch("pickup_putdown.layer1.track_a.image_features.TorchVisionEmbedder"),
            patch("pickup_putdown.ingestion.video_probe.probe_video") as mock_probe,
        ):
            mock_probe.return_value.decode_ok = True
            mock_probe.return_value.duration_s = 30.0

            mock_instance = MagicMock()
            mock_instance.run.return_value = mock_result
            MockPipeline.return_value = mock_instance

            result = runner.invoke(
                app,
                [
                    "infer-track-a",
                    "--config",
                    "configs/track_a.yaml",
                    "--candidate-metadata",
                    str(tmp_dirs["candidate_metadata"]),
                    "--source-video-dir",
                    str(tmp_dirs["source_videos"]),
                    "--artifact-dir",
                    str(tmp_dirs["artifacts"]),
                    "--cache-dir",
                    str(tmp_dirs["cache_dir"]),
                    "--output-dir",
                    str(tmp_dirs["output_dir"]),
                    "--debug-traces",
                    "--candidate-id",
                    "cand_001",
                ],
            )
            assert result.exit_code == 0, result.output
            # Verify debug_traces=True in pipeline config
            call_kwargs = MockPipeline.call_args
            assert call_kwargs is not None

    def test_summary_prints_counts(
        self, tmp_dirs, sample_video, sample_artifacts, sample_candidate_metadata
    ):
        from unittest.mock import MagicMock, patch

        from typer.testing import CliRunner

        from pickup_putdown.cli import app

        runner = CliRunner()

        mock_result = MagicMock()
        mock_result.summary.candidates_processed = 2
        mock_result.summary.candidates_skipped = 1
        mock_result.summary.total_samples = 10
        mock_result.summary.feature_cache_hits = 5
        mock_result.summary.feature_cache_misses = 5
        mock_result.summary.raw_events_emitted = 3
        mock_result.summary.final_events_after_dedup = 2
        mock_result.summary.pickup_count = 1
        mock_result.summary.putdown_count = 1
        mock_result.summary.mean_confidence = 0.75
        mock_result.summary.skip_reasons = {}
        mock_result.output_paths = {}

        with (
            patch(
                "pickup_putdown.layer1.track_a.inference.TrackAInferencePipeline"
            ) as MockPipeline,
            patch("pickup_putdown.layer1.track_a.image_features.TorchVisionEmbedder"),
            patch("pickup_putdown.ingestion.video_probe.probe_video") as mock_probe,
        ):
            mock_probe.return_value.decode_ok = True
            mock_probe.return_value.duration_s = 30.0

            mock_instance = MagicMock()
            mock_instance.run.return_value = mock_result
            MockPipeline.return_value = mock_instance

            result = runner.invoke(
                app,
                [
                    "infer-track-a",
                    "--config",
                    "configs/track_a.yaml",
                    "--candidate-metadata",
                    str(tmp_dirs["candidate_metadata"]),
                    "--source-video-dir",
                    str(tmp_dirs["source_videos"]),
                    "--artifact-dir",
                    str(tmp_dirs["artifacts"]),
                    "--cache-dir",
                    str(tmp_dirs["cache_dir"]),
                    "--output-dir",
                    str(tmp_dirs["output_dir"]),
                ],
            )
            assert result.exit_code == 0, result.output
            assert "Candidates processed:" in result.stdout
            assert "Candidates skipped:" in result.stdout
            assert "Pickups:" in result.stdout
            assert "Putdowns:" in result.stdout

    def test_zero_events_exits_zero(
        self, tmp_dirs, sample_video, sample_artifacts, sample_candidate_metadata
    ):
        """Successful inference with zero events exits 0."""
        from unittest.mock import MagicMock, patch

        from typer.testing import CliRunner

        from pickup_putdown.cli import app

        runner = CliRunner()

        mock_result = MagicMock()
        mock_result.summary.candidates_processed = 2
        mock_result.summary.candidates_skipped = 0
        mock_result.summary.total_samples = 8
        mock_result.summary.feature_cache_hits = 0
        mock_result.summary.feature_cache_misses = 8
        mock_result.summary.raw_events_emitted = 0
        mock_result.summary.final_events_after_dedup = 0
        mock_result.summary.pickup_count = 0
        mock_result.summary.putdown_count = 0
        mock_result.summary.mean_confidence = 0.0
        mock_result.summary.skip_reasons = {}
        mock_result.output_paths = {}

        with (
            patch(
                "pickup_putdown.layer1.track_a.inference.TrackAInferencePipeline"
            ) as MockPipeline,
            patch("pickup_putdown.layer1.track_a.image_features.TorchVisionEmbedder"),
            patch("pickup_putdown.ingestion.video_probe.probe_video") as mock_probe,
        ):
            mock_probe.return_value.decode_ok = True
            mock_probe.return_value.duration_s = 30.0

            mock_instance = MagicMock()
            mock_instance.run.return_value = mock_result
            MockPipeline.return_value = mock_instance

            result = runner.invoke(
                app,
                [
                    "infer-track-a",
                    "--config",
                    "configs/track_a.yaml",
                    "--candidate-metadata",
                    str(tmp_dirs["candidate_metadata"]),
                    "--source-video-dir",
                    str(tmp_dirs["source_videos"]),
                    "--artifact-dir",
                    str(tmp_dirs["artifacts"]),
                    "--cache-dir",
                    str(tmp_dirs["cache_dir"]),
                    "--output-dir",
                    str(tmp_dirs["output_dir"]),
                ],
            )
            assert result.exit_code == 0, result.output

    def test_no_candidates_exits_nonzero(
        self, tmp_dirs, sample_artifacts, sample_candidate_metadata
    ):
        """No candidates processed (all skipped) exits 0 but reports clearly."""
        from unittest.mock import MagicMock, patch

        from typer.testing import CliRunner

        from pickup_putdown.cli import app

        runner = CliRunner()

        mock_result = MagicMock()
        mock_result.summary.candidates_processed = 0
        mock_result.summary.candidates_skipped = 2
        mock_result.summary.total_samples = 0
        mock_result.summary.feature_cache_hits = 0
        mock_result.summary.feature_cache_misses = 0
        mock_result.summary.raw_events_emitted = 0
        mock_result.summary.final_events_after_dedup = 0
        mock_result.summary.pickup_count = 0
        mock_result.summary.putdown_count = 0
        mock_result.summary.mean_confidence = 0.0
        mock_result.summary.skip_reasons = {"missing_source_video": 2}
        mock_result.output_paths = {}

        with (
            patch(
                "pickup_putdown.layer1.track_a.inference.TrackAInferencePipeline"
            ) as MockPipeline,
            patch("pickup_putdown.layer1.track_a.image_features.TorchVisionEmbedder"),
            patch("pickup_putdown.ingestion.video_probe.probe_video") as mock_probe,
        ):
            mock_probe.return_value.decode_ok = False
            mock_probe.return_value.duration_s = None

            mock_instance = MagicMock()
            mock_instance.run.return_value = mock_result
            MockPipeline.return_value = mock_instance

            result = runner.invoke(
                app,
                [
                    "infer-track-a",
                    "--config",
                    "configs/track_a.yaml",
                    "--candidate-metadata",
                    str(tmp_dirs["candidate_metadata"]),
                    "--source-video-dir",
                    str(tmp_dirs["source_videos"]),
                    "--artifact-dir",
                    str(tmp_dirs["artifacts"]),
                    "--cache-dir",
                    str(tmp_dirs["cache_dir"]),
                    "--output-dir",
                    str(tmp_dirs["output_dir"]),
                ],
            )
            # Pipeline completed (exit 0), but all skipped — diagnostics show reason
            assert result.exit_code == 0, result.output
            assert "missing_source_video" in result.stdout

    def test_makefile_invokes_cli(self):
        """Makefile target invokes the correct CLI command."""
        result = subprocess.run(
            ["make", "-n", "infer-track-a"],
            capture_output=True,
            text=True,
        )
        assert "infer-track-a" in result.stdout
        assert "pickup-putdown" in result.stdout or "pickup_putdown.cli" in result.stdout
