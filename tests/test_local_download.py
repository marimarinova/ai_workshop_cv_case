"""Tests for local processing ledger and download batching."""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pickup_putdown.remote.download_coordinator import (
    DownloadConfig,
    DownloadResult,
    _build_download_report,
    _save_local_run_report,
    run_source_download,
)
from pickup_putdown.remote.local_ledger import (
    LocalLedgerEntry,
    LocalProcessingLedger,
    _is_legacy_header,
    _migrate_legacy_row,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_ledger_dir(tmp_path: Path) -> Path:
    return tmp_path / "ledger"


@pytest.fixture
def ledger_path(tmp_ledger_dir: Path) -> Path:
    tmp_ledger_dir.mkdir(parents=True, exist_ok=True)
    return tmp_ledger_dir / "local_processing.csv"


@pytest.fixture
def mock_storage() -> MagicMock:
    storage = MagicMock()
    storage.bucket = "test-bucket"
    storage.prefix = "anon"
    storage.endpoint_url = None
    storage.region = "us-east-1"
    storage.anonymous = False
    return storage


@pytest.fixture
def sample_s3_objects() -> list[dict]:
    return [
        {"key": "anon/camera_01/video_001.mp4", "size": 1000, "etag": "abc001"},
        {"key": "anon/camera_01/video_002.mp4", "size": 2000, "etag": "abc002"},
        {"key": "anon/camera_01/video_003.mp4", "size": 3000, "etag": "abc003"},
        {"key": "anon/camera_02/video_001.mp4", "size": 4000, "etag": "def001"},
        {"key": "anon/camera_02/video_002.mp4", "size": 5000, "etag": "def002"},
        {"key": "anon/candidates/videos/x.mp4", "size": 100, "etag": "cand001"},
        {"key": "anon/process_for_candidates.csv", "size": 50, "etag": "ledger001"},
    ]


# ---------------------------------------------------------------------------
# LocalLedgerEntry tests
# ---------------------------------------------------------------------------


class TestLocalLedgerEntry:
    def test_defaults(self) -> None:
        e = LocalLedgerEntry(file_name="test.mp4")
        assert e.file_name == "test.mp4"
        assert e.downloaded is False
        assert e.generated is False
        assert e.uploaded is False
        assert e.local_source_path == ""
        assert e.source_etag == ""
        assert e.source_size_bytes == ""
        assert e.last_error == ""

    def test_full_entry(self) -> None:
        e = LocalLedgerEntry(
            file_name="cam/v.mp4",
            downloaded=True,
            generated=True,
            uploaded=True,
            local_source_path="/tmp/v.mp4",
            source_etag="abc123",
            source_size_bytes="12345",
        )
        assert e.downloaded is True
        assert e.generated is True
        assert e.uploaded is True


# ---------------------------------------------------------------------------
# LocalProcessingLedger — load / save
# ---------------------------------------------------------------------------


class TestLocalLedgerLoadSave:
    def test_load_empty_creates_nothing(self, ledger_path: Path) -> None:
        ll = LocalProcessingLedger(ledger_path)
        ll.load()
        assert len(ll.entries) == 0

    def test_save_creates_file(self, ledger_path: Path) -> None:
        ll = LocalProcessingLedger(ledger_path)
        ll.entries["a.mp4"] = LocalLedgerEntry(file_name="a.mp4")
        ll.save()
        assert ledger_path.exists()
        text = ledger_path.read_text()
        assert "file_name" in text
        assert "downloaded" in text
        assert "generated" in text
        assert "uploaded" in text

    def test_save_roundtrip(self, ledger_path: Path) -> None:
        ll = LocalProcessingLedger(ledger_path)
        ll.entries["b.mp4"] = LocalLedgerEntry(
            file_name="b.mp4",
            downloaded=True,
            generated=False,
            uploaded=False,
            local_source_path="/tmp/b.mp4",
            source_etag="etag1",
            source_size_bytes="999",
        )
        ll.save()

        ll2 = LocalProcessingLedger(ledger_path)
        ll2.load()
        assert len(ll2.entries) == 1
        e = ll2.entries["b.mp4"]
        assert e.downloaded is True
        assert e.generated is False
        assert e.uploaded is False
        assert e.local_source_path == "/tmp/b.mp4"
        assert e.source_etag == "etag1"
        assert e.source_size_bytes == "999"

    def test_save_sorted_order(self, ledger_path: Path) -> None:
        ll = LocalProcessingLedger(ledger_path)
        ll.entries["z.mp4"] = LocalLedgerEntry(file_name="z.mp4")
        ll.entries["a.mp4"] = LocalLedgerEntry(file_name="a.mp4")
        ll.entries["m.mp4"] = LocalLedgerEntry(file_name="m.mp4")
        ll.save()

        reader = csv.DictReader(StringIO(ledger_path.read_text()))
        names = [row["file_name"] for row in reader]
        assert names == ["a.mp4", "m.mp4", "z.mp4"]

    def test_save_atomic_no_tmp_left(self, ledger_path: Path) -> None:
        ll = LocalProcessingLedger(ledger_path)
        ll.entries["x.mp4"] = LocalLedgerEntry(file_name="x.mp4")
        ll.save()
        tmp = ledger_path.with_suffix(".tmp")
        assert not tmp.exists()


# ---------------------------------------------------------------------------
# LocalProcessingLedger — legacy migration
# ---------------------------------------------------------------------------


class TestLocalLedgerMigration:
    def test_is_legacy_header_true(self) -> None:
        assert _is_legacy_header(["file_name", "processed"]) is True

    def test_is_legacy_header_false(self) -> None:
        assert (
            _is_legacy_header(
                [
                    "file_name",
                    "downloaded",
                    "generated",
                    "uploaded",
                    "local_source_path",
                    "source_etag",
                    "source_size_bytes",
                    "last_error",
                ]
            )
            is False
        )

    def test_migrate_legacy_row_unprocessed(self) -> None:
        row = {"file_name": "v.mp4", "processed": "false"}
        entry = _migrate_legacy_row(row, "v.mp4")
        assert entry.downloaded is False
        assert entry.generated is False
        assert entry.uploaded is False

    def test_migrate_legacy_row_processed(self) -> None:
        row = {"file_name": "v.mp4", "processed": "true"}
        entry = _migrate_legacy_row(row, "v.mp4")
        assert entry.downloaded is False
        assert entry.generated is True
        assert entry.uploaded is True

    def test_load_legacy_file(self, ledger_path: Path) -> None:
        ledger_path.write_text(
            "file_name,processed\ncamera_01/a.mp4,true\ncamera_01/b.mp4,false\n"
        )
        ll = LocalProcessingLedger(ledger_path)
        ll.load()
        assert len(ll.entries) == 2
        a = ll.entries["camera_01/a.mp4"]
        assert a.downloaded is False
        assert a.generated is True
        assert a.uploaded is True
        b = ll.entries["camera_01/b.mp4"]
        assert b.downloaded is False
        assert b.generated is False
        assert b.uploaded is False

    def test_load_legacy_rewrites_new_schema(self, ledger_path: Path) -> None:
        ledger_path.write_text("file_name,processed\nv.mp4,true\n")
        ll = LocalProcessingLedger(ledger_path)
        ll.load()
        ll.save()
        text = ledger_path.read_text()
        assert "downloaded" in text
        assert "generated" in text
        assert "uploaded" in text
        assert "processed" not in text


# ---------------------------------------------------------------------------
# LocalProcessingLedger — selection helpers
# ---------------------------------------------------------------------------


class TestLocalLedgerSelection:
    def _make_ledger(self, ledger_path: Path) -> LocalProcessingLedger:
        ll = LocalProcessingLedger(ledger_path)
        for i in range(25):
            fn = f"cam/video_{i:03d}.mp4"
            ll.entries[fn] = LocalLedgerEntry(file_name=fn)
        return ll

    def test_select_not_downloaded_deterministic(self, ledger_path: Path) -> None:
        ll = self._make_ledger(ledger_path)
        selected = ll.select_not_downloaded(10)
        assert len(selected) == 10
        assert selected[0].file_name == "cam/video_000.mp4"
        assert selected[9].file_name == "cam/video_009.mp4"

    def test_select_not_downloaded_skips_downloaded(self, ledger_path: Path) -> None:
        ll = self._make_ledger(ledger_path)
        for i in range(5):
            fn = f"cam/video_{i:03d}.mp4"
            ll.entries[fn].downloaded = True
        selected = ll.select_not_downloaded(10)
        assert len(selected) == 10
        assert selected[0].file_name == "cam/video_005.mp4"
        assert selected[9].file_name == "cam/video_014.mp4"

    def test_second_batch_different(self, ledger_path: Path) -> None:
        ll = self._make_ledger(ledger_path)
        batch1 = ll.select_not_downloaded(10)
        for e in batch1:
            ll.entries[e.file_name].downloaded = True
        batch2 = ll.select_not_downloaded(10)
        assert len(batch2) == 10
        assert batch2[0].file_name == "cam/video_010.mp4"
        first_batch_names = {e.file_name for e in batch1}
        second_batch_names = {e.file_name for e in batch2}
        assert first_batch_names.isdisjoint(second_batch_names)

    def test_select_ready_for_generation(self, ledger_path: Path) -> None:
        ll = LocalProcessingLedger(ledger_path)
        ll.entries["a.mp4"] = LocalLedgerEntry("a.mp4", downloaded=True, generated=False)
        ll.entries["b.mp4"] = LocalLedgerEntry("b.mp4", downloaded=True, generated=True)
        ll.entries["c.mp4"] = LocalLedgerEntry("c.mp4", downloaded=False, generated=False)
        selected = ll.select_ready_for_generation(10)
        assert len(selected) == 1
        assert selected[0].file_name == "a.mp4"

    def test_select_ready_for_upload(self, ledger_path: Path) -> None:
        ll = LocalProcessingLedger(ledger_path)
        ll.entries["a.mp4"] = LocalLedgerEntry("a.mp4", True, True, False)
        ll.entries["b.mp4"] = LocalLedgerEntry("b.mp4", True, True, True)
        ll.entries["c.mp4"] = LocalLedgerEntry("c.mp4", True, False, False)
        selected = ll.select_ready_for_upload(10)
        assert len(selected) == 1
        assert selected[0].file_name == "a.mp4"

    def test_generation_skips_generated_true(self, ledger_path: Path) -> None:
        ll = LocalProcessingLedger(ledger_path)
        ll.entries["a.mp4"] = LocalLedgerEntry("a.mp4", True, True, False)
        selected = ll.select_ready_for_generation(10)
        assert len(selected) == 0


# ---------------------------------------------------------------------------
# LocalProcessingLedger — state transitions
# ---------------------------------------------------------------------------


class TestLocalLedgerStateTransitions:
    def test_mark_downloaded(self, ledger_path: Path) -> None:
        ll = LocalProcessingLedger(ledger_path)
        ll.entries["v.mp4"] = LocalLedgerEntry("v.mp4")
        ll.mark_downloaded("v.mp4", "/tmp/v.mp4", "etag1", "100")
        e = ll.entries["v.mp4"]
        assert e.downloaded is True
        assert e.local_source_path == "/tmp/v.mp4"
        assert e.source_etag == "etag1"
        assert e.source_size_bytes == "100"
        assert e.last_error == ""

    def test_mark_generated_requires_downloaded(self, ledger_path: Path) -> None:
        ll = LocalProcessingLedger(ledger_path)
        ll.entries["v.mp4"] = LocalLedgerEntry("v.mp4", downloaded=True)
        ll.mark_generated("v.mp4")
        assert ll.entries["v.mp4"].generated is True

    def test_mark_generated_fails_without_downloaded(self, ledger_path: Path) -> None:
        ll = LocalProcessingLedger(ledger_path)
        ll.entries["v.mp4"] = LocalLedgerEntry("v.mp4", downloaded=False)
        with pytest.raises(AssertionError):
            ll.mark_generated("v.mp4")

    def test_mark_uploaded_requires_generated(self, ledger_path: Path) -> None:
        ll = LocalProcessingLedger(ledger_path)
        ll.entries["v.mp4"] = LocalLedgerEntry("v.mp4", True, True, False)
        ll.mark_uploaded("v.mp4")
        assert ll.entries["v.mp4"].uploaded is True

    def test_mark_uploaded_fails_without_generated(self, ledger_path: Path) -> None:
        ll = LocalProcessingLedger(ledger_path)
        ll.entries["v.mp4"] = LocalLedgerEntry("v.mp4", True, False, False)
        with pytest.raises(AssertionError):
            ll.mark_uploaded("v.mp4")

    def test_set_error(self, ledger_path: Path) -> None:
        ll = LocalProcessingLedger(ledger_path)
        ll.entries["v.mp4"] = LocalLedgerEntry("v.mp4")
        ll.set_error("v.mp4", "network timeout")
        assert ll.entries["v.mp4"].last_error == "network timeout"

    def test_full_progression(self, ledger_path: Path) -> None:
        ll = LocalProcessingLedger(ledger_path)
        ll.entries["v.mp4"] = LocalLedgerEntry("v.mp4")
        ll.mark_downloaded("v.mp4", "/tmp/v.mp4", "e1", "100")
        assert ll.entries["v.mp4"].downloaded is True
        assert ll.entries["v.mp4"].generated is False
        assert ll.entries["v.mp4"].uploaded is False
        ll.mark_generated("v.mp4")
        assert ll.entries["v.mp4"].generated is True
        assert ll.entries["v.mp4"].uploaded is False
        ll.mark_uploaded("v.mp4")
        assert ll.entries["v.mp4"].uploaded is True


# ---------------------------------------------------------------------------
# LocalProcessingLedger — reconciliation
# ---------------------------------------------------------------------------


class TestLocalLedgerReconciliation:
    def test_missing_file_resets_downloaded(self, ledger_path: Path, tmp_path: Path) -> None:
        ll = LocalProcessingLedger(ledger_path)
        ll.entries["v.mp4"] = LocalLedgerEntry(
            "v.mp4", downloaded=True, local_source_path=str(tmp_path / "v.mp4")
        )
        warnings = ll.reconcile_with_disk(tmp_path)
        assert ll.entries["v.mp4"].downloaded is False
        assert any("missing" in w for w in warnings)

    def test_existing_file_keeps_downloaded(self, ledger_path: Path, tmp_path: Path) -> None:
        f = tmp_path / "v.mp4"
        f.write_bytes(b"fake video")
        ll = LocalProcessingLedger(ledger_path)
        ll.entries["v.mp4"] = LocalLedgerEntry("v.mp4", downloaded=True, local_source_path=str(f))
        warnings = ll.reconcile_with_disk(tmp_path)
        assert ll.entries["v.mp4"].downloaded is True
        assert not any("missing" in w for w in warnings)

    def test_s3_etag_change_detected(self, ledger_path: Path, tmp_path: Path) -> None:
        f = tmp_path / "v.mp4"
        f.write_bytes(b"fake video")
        ll = LocalProcessingLedger(ledger_path)
        ll.entries["v.mp4"] = LocalLedgerEntry(
            "v.mp4",
            downloaded=True,
            local_source_path=str(f),
            source_etag="old_etag",
            source_size_bytes="100",
        )
        s3_info = {"v.mp4": {"etag": "new_etag", "size": "100"}}
        warnings = ll.reconcile_with_disk(tmp_path, s3_info, refresh_changed=False)
        assert ll.entries["v.mp4"].downloaded is False
        assert any("changed" in w for w in warnings)

    def test_s3_etag_change_with_refresh(self, ledger_path: Path, tmp_path: Path) -> None:
        f = tmp_path / "v.mp4"
        f.write_bytes(b"fake video")
        ll = LocalProcessingLedger(ledger_path)
        ll.entries["v.mp4"] = LocalLedgerEntry(
            "v.mp4",
            downloaded=True,
            local_source_path=str(f),
            source_etag="old_etag",
            source_size_bytes="100",
        )
        s3_info = {"v.mp4": {"etag": "new_etag", "size": "100"}}
        ll.reconcile_with_disk(tmp_path, s3_info, refresh_changed=True)
        assert ll.entries["v.mp4"].downloaded is True

    def test_size_change_detected(self, ledger_path: Path, tmp_path: Path) -> None:
        f = tmp_path / "v.mp4"
        f.write_bytes(b"fake video")
        ll = LocalProcessingLedger(ledger_path)
        ll.entries["v.mp4"] = LocalLedgerEntry(
            "v.mp4",
            downloaded=True,
            local_source_path=str(f),
            source_etag="same",
            source_size_bytes="100",
        )
        s3_info = {"v.mp4": {"etag": "same", "size": "200"}}
        ll.reconcile_with_disk(tmp_path, s3_info, refresh_changed=False)
        assert ll.entries["v.mp4"].downloaded is False

    def test_adopt_existing_file(self, ledger_path: Path, tmp_path: Path) -> None:
        f = tmp_path / "v.mp4"
        f.write_bytes(b"fake video" * 10)
        actual_size = str(f.stat().st_size)
        ll = LocalProcessingLedger(ledger_path)
        ll.entries["v.mp4"] = LocalLedgerEntry(
            "v.mp4",
            downloaded=False,
            local_source_path=str(f),
            source_size_bytes=actual_size,
        )
        s3_info = {"v.mp4": {"etag": "abc", "size": actual_size}}
        ll.reconcile_with_disk(tmp_path, s3_info)
        assert ll.entries["v.mp4"].downloaded is True


# ---------------------------------------------------------------------------
# LocalProcessingLedger — duplicate basenames
# ---------------------------------------------------------------------------


class TestLocalLedgerDuplicateBasenames:
    def test_different_dirs_preserved(self, ledger_path: Path) -> None:
        ll = LocalProcessingLedger(ledger_path)
        ll.sync_with_discovery(["cam_a/video.mp4", "cam_b/video.mp4"])
        assert len(ll.entries) == 2
        assert "cam_a/video.mp4" in ll.entries
        assert "cam_b/video.mp4" in ll.entries


# ---------------------------------------------------------------------------
# Download coordinator — report building
# ---------------------------------------------------------------------------


class TestDownloadReport:
    def test_build_report(self) -> None:
        from datetime import UTC, datetime

        t = datetime.now(UTC)
        report = _build_download_report(
            run_id="test_001",
            mode="download",
            requested=10,
            selected=10,
            downloaded=8,
            failed=2,
            skipped=0,
            t_start=t,
            t_end=t,
            transfer_workers=4,
            selected_keys=["a.mp4", "b.mp4"],
            failed_keys=["b.mp4"],
            errors=["b.mp4: timeout"],
        )
        assert report.run_id == "test_001"
        assert report.mode == "download"
        assert report.requested_count == 10
        assert report.selected_count == 10
        assert report.downloaded_count == 8
        assert report.failed_count == 2
        assert report.transfer_workers == 4
        assert "a.mp4" in report.selected_keys
        assert "b.mp4" in report.failed_keys

    def test_save_report_to_disk(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime

        t = datetime.now(UTC)
        report = _build_download_report(
            run_id="test_save",
            mode="download",
            requested=5,
            selected=5,
            downloaded=5,
            failed=0,
            skipped=0,
            t_start=t,
            t_end=t,
            transfer_workers=2,
        )
        out_dir = tmp_path / "output"
        _save_local_run_report(out_dir, report)
        report_file = out_dir / "runs" / "test_save.json"
        assert report_file.exists()
        data = json.loads(report_file.read_text())
        assert data["run_id"] == "test_save"
        assert data["mode"] == "download"
        assert data["downloaded_count"] == 5


# ---------------------------------------------------------------------------
# Download coordinator — run_source_download integration
# ---------------------------------------------------------------------------


class TestRunSourceDownload:
    def test_no_undownloaded_returns_empty(self, mock_storage: MagicMock, tmp_path: Path) -> None:
        mock_storage.list_objects.return_value = []
        mock_storage.relative_key = lambda k: k.replace("anon/", "")
        from pickup_putdown.remote.s3_storage import S3Storage

        mock_storage.is_video = S3Storage.is_video
        mock_storage.is_excluded = S3Storage.is_excluded

        ledger_path = tmp_path / "local_processing.csv"
        ll = LocalProcessingLedger(ledger_path)
        # Create the local file so reconciliation keeps downloaded=True
        source_dir = tmp_path / "sources"
        source_dir.mkdir()
        (source_dir / "a.mp4").write_bytes(b"fake video data")
        ll.entries["a.mp4"] = LocalLedgerEntry(
            "a.mp4", downloaded=True, local_source_path=str(source_dir / "a.mp4")
        )
        ll.save()

        config = DownloadConfig(
            target_count=5,
            local_source_dir=str(source_dir),
            local_output_dir=str(tmp_path / "output"),
        )

        report = run_source_download(mock_storage, ll, config)
        assert report.selected_count == 0
        assert report.downloaded_count == 0

    def test_selects_deterministic_batch(self, mock_storage: MagicMock, tmp_path: Path) -> None:
        objects = []
        for i in range(20):
            objects.append(
                {
                    "key": f"anon/cam/video_{i:03d}.mp4",
                    "size": 1000 + i,
                    "etag": f"etag{i:03d}",
                }
            )
        mock_storage.list_objects.return_value = objects
        mock_storage.relative_key = lambda k: k.replace("anon/", "")
        from pickup_putdown.remote.s3_storage import S3Storage

        mock_storage.is_video = S3Storage.is_video
        mock_storage.is_excluded = S3Storage.is_excluded

        ledger_path = tmp_path / "local_processing.csv"
        ll = LocalProcessingLedger(ledger_path)

        config = DownloadConfig(
            target_count=10,
            local_source_dir=str(tmp_path / "sources"),
            local_output_dir=str(tmp_path / "output"),
        )

        # Mock the download to avoid actual S3 calls
        with patch(
            "pickup_putdown.remote.download_coordinator._download_single_source"
        ) as mock_dl:
            mock_dl.return_value = DownloadResult(
                success=True,
                local_path=str(tmp_path / "sources" / "cam" / "video_000.mp4"),
                etag="etag000",
                size_bytes="1000",
            )
            report = run_source_download(mock_storage, ll, config)

        assert report.selected_count == 10
        assert report.downloaded_count == 10
        assert mock_dl.call_count == 10

    def test_failed_download_remains_false(self, mock_storage: MagicMock, tmp_path: Path) -> None:
        objects = [
            {"key": "anon/a.mp4", "size": 100, "etag": "e1"},
            {"key": "anon/b.mp4", "size": 200, "etag": "e2"},
        ]
        mock_storage.list_objects.return_value = objects
        mock_storage.relative_key = lambda k: k.replace("anon/", "")
        from pickup_putdown.remote.s3_storage import S3Storage

        mock_storage.is_video = S3Storage.is_video
        mock_storage.is_excluded = S3Storage.is_excluded

        ledger_path = tmp_path / "local_processing.csv"
        ll = LocalProcessingLedger(ledger_path)

        config = DownloadConfig(
            target_count=10,
            local_source_dir=str(tmp_path / "sources"),
            local_output_dir=str(tmp_path / "output"),
        )

        call_count = 0

        def side_effect(*args: object) -> DownloadResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return DownloadResult(
                    success=True, local_path="/tmp/a.mp4", etag="e1", size_bytes="100"
                )
            return DownloadResult(success=False, error="timeout")

        with patch(
            "pickup_putdown.remote.download_coordinator._download_single_source"
        ) as mock_dl:
            mock_dl.side_effect = side_effect
            report = run_source_download(mock_storage, ll, config)

        assert report.downloaded_count == 1
        assert report.failed_count == 1
        assert ll.entries["a.mp4"].downloaded is True
        assert ll.entries["b.mp4"].downloaded is False
        assert ll.entries["b.mp4"].last_error == "timeout"

    def test_ledger_persisted_after_each_download(
        self, mock_storage: MagicMock, tmp_path: Path
    ) -> None:
        objects = [
            {"key": "anon/a.mp4", "size": 100, "etag": "e1"},
            {"key": "anon/b.mp4", "size": 200, "etag": "e2"},
        ]
        mock_storage.list_objects.return_value = objects
        mock_storage.relative_key = lambda k: k.replace("anon/", "")
        from pickup_putdown.remote.s3_storage import S3Storage

        mock_storage.is_video = S3Storage.is_video
        mock_storage.is_excluded = S3Storage.is_excluded

        ledger_path = tmp_path / "local_processing.csv"
        ll = LocalProcessingLedger(ledger_path)

        config = DownloadConfig(
            target_count=10,
            local_source_dir=str(tmp_path / "sources"),
            local_output_dir=str(tmp_path / "output"),
        )

        call_count = 0

        def side_effect(*args: object) -> DownloadResult:
            nonlocal call_count
            call_count += 1
            fn = "a.mp4" if call_count == 1 else "b.mp4"
            return DownloadResult(
                success=True,
                local_path=str(tmp_path / "sources" / fn),
                etag="e1",
                size_bytes="100",
            )

        with patch(
            "pickup_putdown.remote.download_coordinator._download_single_source"
        ) as mock_dl:
            mock_dl.side_effect = side_effect
            run_source_download(mock_storage, ll, config)

        assert ledger_path.exists()
        text = ledger_path.read_text()
        assert "a.mp4" in text
        assert "b.mp4" in text


# ---------------------------------------------------------------------------
# Download coordinator — partial file safety
# ---------------------------------------------------------------------------


class TestPartialFileSafety:
    def test_part_file_not_treated_as_complete(self, tmp_path: Path) -> None:
        part = tmp_path / "video.mp4.part"
        part.write_bytes(b"partial data")
        from pickup_putdown.remote.download_coordinator import _validate_local_video

        assert _validate_local_video(part, {}) is False

    def test_part_file_cleaned_on_retry(self, tmp_path: Path, mock_storage: MagicMock) -> None:
        from pickup_putdown.remote.download_coordinator import _download_single_source

        source_dir = tmp_path / "sources"
        source_dir.mkdir()
        final = source_dir / "v.mp4"
        part = Path(f"{final}.part")
        part.write_bytes(b"stale")

        mock_storage.full_key.return_value = "anon/v.mp4"
        mock_storage.download.side_effect = RuntimeError("fail")

        result = _download_single_source("v.mp4", mock_storage, source_dir, {})
        assert result.success is False
        assert not part.exists()


# ---------------------------------------------------------------------------
# Download coordinator — run report not overwritten
# ---------------------------------------------------------------------------


class TestRunReportNotOverwritten:
    def test_prior_reports_preserved(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime

        out_dir = tmp_path / "output"
        t = datetime.now(UTC)

        r1 = _build_download_report(
            run_id="run_001",
            mode="download",
            requested=5,
            selected=5,
            downloaded=5,
            failed=0,
            skipped=0,
            t_start=t,
            t_end=t,
            transfer_workers=2,
        )
        _save_local_run_report(out_dir, r1)

        r2 = _build_download_report(
            run_id="run_002",
            mode="download",
            requested=5,
            selected=5,
            downloaded=3,
            failed=2,
            skipped=0,
            t_start=t,
            t_end=t,
            transfer_workers=2,
        )
        _save_local_run_report(out_dir, r2)

        runs_dir = out_dir / "runs"
        assert (runs_dir / "run_001.json").exists()
        assert (runs_dir / "run_002.json").exists()
        d1 = json.loads((runs_dir / "run_001.json").read_text())
        assert d1["downloaded_count"] == 5
        d2 = json.loads((runs_dir / "run_002.json").read_text())
        assert d2["downloaded_count"] == 3


# ---------------------------------------------------------------------------
# Existing behavior — immediate upload unchanged
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_immediate_upload_unchanged(self) -> None:
        from pickup_putdown.remote.ledger import LedgerEntry, ProcessingLedger

        storage = MagicMock()
        ledger = ProcessingLedger(storage)
        ledger.entries["v.mp4"] = LedgerEntry("v.mp4", False)
        ledger.mark_processed("v.mp4")
        assert ledger.entries["v.mp4"].processed is True

    def test_existing_coordinator_config_unchanged(self) -> None:
        from pickup_putdown.remote.coordinator import CoordinationConfig

        cfg = CoordinationConfig(target_count=5)
        assert cfg.target_count == 5
        assert cfg.workers == 4
        assert cfg.dry_run is False

    def test_existing_worker_config_unchanged(self) -> None:
        from pickup_putdown.remote.worker import WorkerConfig

        cfg = WorkerConfig(
            storage_config=Path("s.yaml"),
            pipeline_config=Path("p.yaml"),
            work_dir=Path("/tmp"),
        )
        assert cfg.camera_id == "store_camera_01"


# ---------------------------------------------------------------------------
# Repeated download and generation idempotency
# ---------------------------------------------------------------------------


class TestRepeatedRunsIdempotent:
    def test_repeated_download_runs_no_redownload(self, ledger_path: Path) -> None:
        ll = LocalProcessingLedger(ledger_path)
        for i in range(20):
            fn = f"cam/v_{i:03d}.mp4"
            ll.entries[fn] = LocalLedgerEntry(fn, downloaded=True)
        ll.save()

        selected = ll.select_not_downloaded(10)
        assert len(selected) == 0

    def test_repeated_generation_runs_no_rerun(self, ledger_path: Path) -> None:
        ll = LocalProcessingLedger(ledger_path)
        for i in range(20):
            fn = f"cam/v_{i:03d}.mp4"
            ll.entries[fn] = LocalLedgerEntry(fn, downloaded=True, generated=True)
        ll.save()

        selected = ll.select_ready_for_generation(10)
        assert len(selected) == 0

    def test_upload_selects_only_unuploaded(self, ledger_path: Path) -> None:
        ll = LocalProcessingLedger(ledger_path)
        ll.entries["a.mp4"] = LocalLedgerEntry("a.mp4", True, True, True)
        ll.entries["b.mp4"] = LocalLedgerEntry("b.mp4", True, True, False)
        selected = ll.select_ready_for_upload(10)
        assert len(selected) == 1
        assert selected[0].file_name == "b.mp4"
