"""Tests for remote S3 candidate generation modules."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pickup_putdown.remote.coordinator import _make_source_video_id
from pickup_putdown.remote.discovery import discover_source_videos
from pickup_putdown.remote.encoding import (
    EncodingConfig,
    encode_candidate,
    validate_encoding,
)
from pickup_putdown.remote.ledger import (
    LedgerEntry,
    ProcessingLedger,
)
from pickup_putdown.remote.worker import (
    WorkerConfig,
    WorkerResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_work(tmp_path: Path) -> Path:
    return tmp_path / "work"


@pytest.fixture
def mock_storage() -> MagicMock:
    storage = MagicMock()
    storage.bucket = "chillinbite-cameras"
    storage.prefix = "annon"
    storage.endpoint_url = None
    storage.region = "eu-central-1"
    storage.anonymous = False
    return storage


@pytest.fixture
def mock_ledger(mock_storage: MagicMock) -> ProcessingLedger:
    return ProcessingLedger(mock_storage)


@pytest.fixture
def sample_candidates_parquet(tmp_path: Path) -> Path:
    """Create a minimal candidates.parquet for testing."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    data = [
        {
            "candidate_id": "test_candidate_0001",
            "clip_id": "test_video",
            "actor_id": "actor_1",
            "hand_side": "right",
            "region_id": "Shelf_01",
            "raw_start_s": 10.0,
            "raw_end_s": 15.0,
            "window_start_s": 8.0,
            "window_end_s": 17.0,
            "n_raw_interactions": 1,
            "min_region_distance": 0.5,
            "max_wrist_confidence": 0.8,
            "total_dwell_duration_s": 3.0,
            "config_fingerprint": "fp1",
            "proposal_reason": "wrist_in_region",
            "proposal_score": 0.75,
            "review_status": "pending",
        },
    ]
    table = pa.Table.from_pylist(data)
    out = tmp_path / "candidates.parquet"
    pq.write_table(table, str(out))
    return out


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """Create a minimal test video using ffmpeg."""
    video_path = tmp_path / "test_source.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg not available")

    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=20:size=320x240:rate=30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-t",
            "20",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0 or not video_path.exists():
        pytest.skip(f"Could not create test video: {result.stderr[-200:]}")
    return video_path


# ---------------------------------------------------------------------------
# Ledger tests
# ---------------------------------------------------------------------------


class TestProcessingLedger:
    def test_load_creates_empty(
        self, mock_storage: MagicMock, mock_ledger: ProcessingLedger
    ) -> None:
        mock_storage.download.side_effect = FileNotFoundError("no ledger")
        mock_ledger.load()
        assert len(mock_ledger.entries) == 0

    def test_load_parses_existing(
        self, mock_storage: MagicMock, mock_ledger: ProcessingLedger, tmp_path: Path
    ) -> None:
        csv_content = (
            "file_name,processed\ncamera_01/video_001.mp4,false\ncamera_01/video_002.mp4,true\n"
        )
        tmp_file = tmp_path / "ledger.csv"
        tmp_file.write_text(csv_content)

        def fake_download(key: str, path: Path) -> None:
            path.write_text(csv_content)

        mock_storage.download = fake_download
        mock_ledger.load()

        assert len(mock_ledger.entries) == 2
        assert mock_ledger.entries["camera_01/video_001.mp4"].processed is False
        assert mock_ledger.entries["camera_01/video_002.mp4"].processed is True

    def test_sync_adds_new_preserves_existing(self, mock_ledger: ProcessingLedger) -> None:
        mock_ledger.entries["existing.mp4"] = LedgerEntry("existing.mp4", processed=True)
        mock_ledger.sync_with_discovery(["existing.mp4", "new.mp4"])
        assert len(mock_ledger.entries) == 2
        assert mock_ledger.entries["existing.mp4"].processed is True
        assert mock_ledger.entries["new.mp4"].processed is False

    def test_select_unprocessed_deterministic(self, mock_ledger: ProcessingLedger) -> None:
        mock_ledger.entries["b.mp4"] = LedgerEntry("b.mp4", False)
        mock_ledger.entries["a.mp4"] = LedgerEntry("a.mp4", False)
        mock_ledger.entries["c.mp4"] = LedgerEntry("c.mp4", True)
        selected = mock_ledger.select_unprocessed(2)
        assert len(selected) == 2
        assert selected[0].file_name == "a.mp4"
        assert selected[1].file_name == "b.mp4"

    def test_select_respects_target_count(self, mock_ledger: ProcessingLedger) -> None:
        for i in range(10):
            mock_ledger.entries[f"v{i:03d}.mp4"] = LedgerEntry(f"v{i:03d}.mp4", False)
        selected = mock_ledger.select_unprocessed(3)
        assert len(selected) == 3

    def test_mark_processed(self, mock_ledger: ProcessingLedger) -> None:
        mock_ledger.entries["v.mp4"] = LedgerEntry("v.mp4", False)
        mock_ledger.mark_processed("v.mp4")
        assert mock_ledger.entries["v.mp4"].processed is True

    def test_save_writes_csv(
        self, mock_storage: MagicMock, mock_ledger: ProcessingLedger, tmp_path: Path
    ) -> None:
        mock_ledger.entries["a.mp4"] = LedgerEntry("a.mp4", False)
        mock_ledger.entries["b.mp4"] = LedgerEntry("b.mp4", True)

        uploaded: dict[str, Path] = {}

        def fake_upload(local: Path, key: str) -> None:
            uploaded[key] = local

        mock_storage.upload = fake_upload
        mock_storage.full_key = lambda k: f"annon/{k}"
        mock_ledger.save()

        assert "annon/process_for_candidates.csv" in uploaded

    def test_preserves_true_never_resets(self, mock_ledger: ProcessingLedger) -> None:
        mock_ledger.entries["v.mp4"] = LedgerEntry("v.mp4", True)
        mock_ledger.sync_with_discovery(["v.mp4"])
        assert mock_ledger.entries["v.mp4"].processed is True

    def test_processed_count(self, mock_ledger: ProcessingLedger) -> None:
        mock_ledger.entries["a.mp4"] = LedgerEntry("a.mp4", True)
        mock_ledger.entries["b.mp4"] = LedgerEntry("b.mp4", False)
        assert mock_ledger.processed_count == 1
        assert mock_ledger.unprocessed_count == 1


# ---------------------------------------------------------------------------
# Discovery tests
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_discovers_videos_excludes_candidates(self, mock_storage: MagicMock) -> None:
        mock_storage.list_objects.return_value = [
            {"key": "annon/camera_01/video_001.mp4"},
            {"key": "annon/camera_01/video_002.mp4"},
            {"key": "annon/candidates/videos/x.mp4"},
            {"key": "annon/process_for_candidates.csv"},
            {"key": "annon/readme.txt"},
        ]
        mock_storage.relative_key = lambda k: k.replace("annon/", "")
        from pickup_putdown.remote.s3_storage import S3Storage

        mock_storage.is_video = S3Storage.is_video
        mock_storage.is_excluded = S3Storage.is_excluded
        discovered = discover_source_videos(mock_storage)
        assert len(discovered) == 2
        assert "camera_01/video_001.mp4" in discovered
        assert "camera_01/video_002.mp4" in discovered
        assert "candidates/videos/x.mp4" not in discovered
        assert "process_for_candidates.csv" not in discovered
        assert "readme.txt" not in discovered

    def test_duplicate_basenames_different_dirs(self, mock_storage: MagicMock) -> None:
        mock_storage.list_objects.return_value = [
            {"key": "annon/cam_a/video.mp4"},
            {"key": "annon/cam_b/video.mp4"},
        ]
        mock_storage.relative_key = lambda k: k.replace("annon/", "")
        from pickup_putdown.remote.s3_storage import S3Storage

        mock_storage.is_video = S3Storage.is_video
        mock_storage.is_excluded = S3Storage.is_excluded
        discovered = discover_source_videos(mock_storage)
        assert len(discovered) == 2
        assert "cam_a/video.mp4" in discovered
        assert "cam_b/video.mp4" in discovered

    def test_sorted_order(self, mock_storage: MagicMock) -> None:
        mock_storage.list_objects.return_value = [
            {"key": "annon/z.mp4"},
            {"key": "annon/a.mp4"},
            {"key": "annon/m.mp4"},
        ]
        mock_storage.relative_key = lambda k: k.replace("annon/", "")
        from pickup_putdown.remote.s3_storage import S3Storage

        mock_storage.is_video = S3Storage.is_video
        mock_storage.is_excluded = S3Storage.is_excluded
        discovered = discover_source_videos(mock_storage)
        assert discovered == ["a.mp4", "m.mp4", "z.mp4"]


# ---------------------------------------------------------------------------
# Encoding tests
# ---------------------------------------------------------------------------


class TestEncoding:
    def test_encode_produces_h264(self, sample_video: Path, tmp_path: Path) -> None:
        out = tmp_path / "encoded.mp4"
        encode_candidate(sample_video, out, EncodingConfig())
        assert out.exists()
        assert out.stat().st_size > 0

    def test_validate_h264_passes(self, sample_video: Path, tmp_path: Path) -> None:
        out = tmp_path / "valid.mp4"
        encode_candidate(sample_video, out, EncodingConfig())
        result = validate_encoding(out)
        assert result.valid is True
        assert result.codec_name == "h264"
        assert result.pix_fmt == "yuv420p"
        assert result.duration_s > 0

    def test_validate_empty_file_fails(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.mp4"
        empty.write_bytes(b"")
        result = validate_encoding(empty)
        assert result.valid is False

    def test_validate_no_video_stream_fails(self, tmp_path: Path) -> None:
        text = tmp_path / "not_video.mp4"
        text.write_text("not a video")
        result = validate_encoding(text)
        assert result.valid is False

    def test_encoding_config_defaults(self) -> None:
        cfg = EncodingConfig()
        assert cfg.codec == "libx264"
        assert cfg.pixel_format == "yuv420p"
        assert cfg.faststart is True
        assert cfg.retain_audio is False


# ---------------------------------------------------------------------------
# Worker tests
# ---------------------------------------------------------------------------


class TestWorkerHelpers:
    def test_make_source_video_id_simple(self) -> None:
        assert _make_source_video_id("video.mp4") == "video"

    def test_make_source_video_id_nested(self) -> None:
        assert _make_source_video_id("camera_01/video_001.mp4") == "camera_01_video_001"

    def test_make_source_video_id_deep(self) -> None:
        assert _make_source_video_id("a/b/c/video.mp4") == "a_b_c_video"

    def test_worker_result_defaults(self) -> None:
        r = WorkerResult(source_video_id="v1", source_key="k1", success=True)
        assert r.candidate_count == 0
        assert r.candidates == []
        assert r.error == ""

    def test_worker_config_defaults(self) -> None:
        cfg = WorkerConfig(
            storage_config=Path("s.yaml"),
            pipeline_config=Path("p.yaml"),
            work_dir=Path("/tmp"),
        )
        assert cfg.triage_config == "configs/triage.yaml"
        assert cfg.camera_id == "store_camera_01"


# ---------------------------------------------------------------------------
# S3 Storage tests
# ---------------------------------------------------------------------------


class TestS3Storage:
    def test_parse_bucket_uri(self) -> None:
        from pickup_putdown.remote.s3_storage import _parse_bucket_uri

        bucket, prefix = _parse_bucket_uri("s3://my-bucket/anon/")
        assert bucket == "my-bucket"
        assert prefix == "anon"

    def test_relative_key(self) -> None:
        from pickup_putdown.remote.s3_storage import S3Storage

        with patch("boto3.client"):
            s = S3Storage("s3://b/p/")
        assert s.relative_key("p/video.mp4") == "video.mp4"

    def test_full_key(self) -> None:
        from pickup_putdown.remote.s3_storage import S3Storage

        with patch("boto3.client"):
            s = S3Storage("s3://b/p/")
        assert s.full_key("video.mp4") == "p/video.mp4"

    def test_is_video(self) -> None:
        from pickup_putdown.remote.s3_storage import S3Storage

        assert S3Storage.is_video("test.mp4") is True
        assert S3Storage.is_video("test.MP4") is True
        assert S3Storage.is_video("test.txt") is False

    def test_is_excluded(self) -> None:
        from pickup_putdown.remote.s3_storage import S3Storage

        assert S3Storage.is_excluded("candidates/x.mp4") is True
        assert S3Storage.is_excluded("process_for_candidates.csv") is True
        assert S3Storage.is_excluded("camera_01/v.mp4") is False


# ---------------------------------------------------------------------------
# Coordinator tests
# ---------------------------------------------------------------------------


class TestCoordinator:
    def test_dry_run_no_processing(self) -> None:
        from pickup_putdown.remote.coordinator import CoordinationConfig

        cfg = CoordinationConfig(
            target_count=5,
            dry_run=True,
        )
        assert cfg.dry_run is True
        assert cfg.target_count == 5

    def test_fail_fast_config(self) -> None:
        from pickup_putdown.remote.coordinator import CoordinationConfig

        cfg = CoordinationConfig(
            target_count=100,
            fail_fast=True,
            workers=8,
            gpu_workers=1,
        )
        assert cfg.fail_fast is True
        assert cfg.workers == 8
        assert cfg.gpu_workers == 1
