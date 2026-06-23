"""Download coordinator: batch download source videos from S3 to local cache."""

from __future__ import annotations

import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DownloadConfig:
    target_count: int
    transfer_workers: int = 4
    local_source_dir: str = ".local/source_videos"
    local_output_dir: str = ".local/candidate_staging"
    minimum_free_disk_gb: float = 0.0
    refresh_changed: bool = False


@dataclass
class DownloadRunReport:
    run_id: str
    mode: str = "download"
    requested_count: int = 0
    selected_count: int = 0
    downloaded_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    started_at: str = ""
    completed_at: str = ""
    transfer_workers: int = 0
    selected_keys: list[str] = field(default_factory=list)
    failed_keys: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def run_source_download(
    storage: Any,
    local_ledger: Any,
    config: DownloadConfig,
) -> DownloadRunReport:
    """Coordinate batch download of source videos from S3.

    Flow:
    1. Discover source videos from S3
    2. Sync into local ledger
    3. Reconcile ledger with disk
    4. Select not-yet-downloaded entries
    5. Download concurrently with bounded workers
    6. Validate each download
    7. Persist ledger after each successful download
    8. Write local run report
    """
    run_id = f"dl_{uuid.uuid4().hex[:12]}"
    t_start = datetime.now(UTC)

    logger.info("=== Source download run %s ===", run_id)
    logger.info("Target count: %d", config.target_count)

    local_source_dir = Path(config.local_source_dir)
    local_output_dir = Path(config.local_output_dir)
    local_source_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover source videos
    from pickup_putdown.remote.discovery import discover_source_videos

    discovered = discover_source_videos(storage)

    # 2. Sync ledger
    local_ledger.load()
    local_ledger.sync_with_discovery(discovered)

    # Build S3 info map for reconciliation
    s3_info_map: dict[str, dict[str, str]] = {}
    all_objects = storage.list_objects()
    for obj in all_objects:
        full_key = obj["key"]
        rel_key = storage.relative_key(full_key)
        if storage.is_video(rel_key) and not storage.is_excluded(rel_key):
            s3_info_map[rel_key] = {
                "etag": obj.get("etag", ""),
                "size": str(obj.get("size", "")),
            }

    # 3. Reconcile
    warnings = local_ledger.reconcile_with_disk(
        local_source_dir=local_source_dir,
        current_s3_info=s3_info_map,
        refresh_changed=config.refresh_changed,
    )
    for w in warnings:
        logger.warning(w)
    local_ledger.save()

    # 4. Select not-downloaded
    selected = local_ledger.select_not_downloaded(config.target_count)

    if not selected:
        logger.info("No undownloaded videos to process.")
        report = _build_download_report(
            run_id=run_id,
            mode="download",
            requested=config.target_count,
            selected=0,
            downloaded=0,
            failed=0,
            skipped=0,
            t_start=t_start,
            transfer_workers=config.transfer_workers,
        )
        _save_local_run_report(local_output_dir, report)
        return report

    logger.info("Selected %d video(s) for download", len(selected))

    # Disk space check
    if config.minimum_free_disk_gb > 0:
        _check_disk_space(local_source_dir, config.minimum_free_disk_gb)

    # 5. Download in parallel
    max_workers = min(config.transfer_workers, len(selected))
    downloaded = 0
    failed = 0
    failed_keys: list[str] = []
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for entry in selected:
            future = executor.submit(
                _download_single_source,
                entry.file_name,
                storage,
                local_source_dir,
                s3_info_map.get(entry.file_name, {}),
            )
            futures[future] = entry.file_name

        for future in as_completed(futures):
            source_key = futures[future]
            try:
                result = future.result(timeout=3600)
                if result.success:
                    downloaded += 1
                    local_ledger.mark_downloaded(
                        file_name=source_key,
                        local_source_path=result.local_path,
                        source_etag=result.etag,
                        source_size_bytes=result.size_bytes,
                    )
                    local_ledger.save()
                else:
                    failed += 1
                    failed_keys.append(source_key)
                    err_msg = f"{source_key}: {result.error}"
                    errors.append(err_msg)
                    local_ledger.set_error(source_key, result.error)
                    local_ledger.save()
            except Exception as exc:
                failed += 1
                failed_keys.append(source_key)
                err_msg = f"{source_key}: {exc}"
                errors.append(err_msg)
                logger.error("Unhandled exception for %s: %s", source_key, exc)
                local_ledger.set_error(source_key, str(exc)[:500])
                local_ledger.save()

    t_end = datetime.now(UTC)

    report = _build_download_report(
        run_id=run_id,
        mode="download",
        requested=config.target_count,
        selected=len(selected),
        downloaded=downloaded,
        failed=failed,
        skipped=max(0, len(selected) - downloaded - failed),
        t_start=t_start,
        t_end=t_end,
        transfer_workers=config.transfer_workers,
        selected_keys=[e.file_name for e in selected],
        failed_keys=failed_keys,
        errors=errors,
    )

    _save_local_run_report(local_output_dir, report)

    logger.info("=== Download run complete ===")
    logger.info("Downloaded: %d, Failed: %d", downloaded, failed)

    return report


@dataclass
class DownloadResult:
    success: bool
    local_path: str = ""
    etag: str = ""
    size_bytes: str = ""
    error: str = ""


def _download_single_source(
    source_rel_key: str,
    storage: Any,
    local_source_dir: Path,
    s3_info: dict[str, str],
) -> DownloadResult:
    """Download a single source video with partial-file safety."""
    final_path = local_source_dir / source_rel_key
    part_path = Path(f"{final_path}.part")

    # Clean up stale .part file
    if part_path.exists():
        logger.info("Removing stale partial file: %s", part_path)
        part_path.unlink()

    # Skip if already valid
    if final_path.exists() and final_path.is_file() and _validate_local_video(final_path, s3_info):
        logger.info("Source already valid locally: %s", source_rel_key)
        return DownloadResult(
            success=True,
            local_path=str(final_path),
            etag=s3_info.get("etag", ""),
            size_bytes=s3_info.get("size", ""),
        )

    try:
        full_key = storage.full_key(source_rel_key)
        part_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s -> %s", full_key, part_path)
        storage.download(full_key, part_path)

        # Validate
        expected_size = s3_info.get("size", "")
        if expected_size:
            actual_size = str(part_path.stat().st_size)
            if actual_size != expected_size:
                part_path.unlink(missing_ok=True)
                return DownloadResult(
                    success=False,
                    error=f"Size mismatch: expected {expected_size}, got {actual_size}",
                )

        valid = _validate_local_video(part_path, s3_info)
        if not valid:
            part_path.unlink(missing_ok=True)
            return DownloadResult(
                success=False,
                error="Video probe validation failed",
            )

        # Atomic rename
        final_path.parent.mkdir(parents=True, exist_ok=True)
        part_path.rename(final_path)

        return DownloadResult(
            success=True,
            local_path=str(final_path),
            etag=s3_info.get("etag", ""),
            size_bytes=s3_info.get("size", ""),
        )

    except Exception as exc:
        if part_path.exists():
            part_path.unlink(missing_ok=True)
        return DownloadResult(success=False, error=str(exc))


def _validate_local_video(local_path: Path, s3_info: dict[str, str]) -> bool:
    """Validate downloaded video using probe facilities."""
    if not local_path.exists() or not local_path.is_file():
        return False
    if local_path.stat().st_size == 0:
        return False

    from pickup_putdown.ingestion.video_probe import probe_video

    result = probe_video(str(local_path))
    return result.decode_ok


def _check_disk_space(path: Path, minimum_gb: float) -> None:
    """Check available disk space."""
    p = os.statvfs(str(path.parent if path.parent.exists() else path))
    free_gb = (p.f_bavail * p.f_frsize) / (1024**3)
    if free_gb < minimum_gb:
        raise RuntimeError(
            f"Insufficient disk space: {free_gb:.1f} GB free, {minimum_gb:.1f} GB required"
        )
    logger.info("Disk space check: %.1f GB free (minimum %.1f GB)", free_gb, minimum_gb)


def _build_download_report(
    run_id: str,
    mode: str,
    requested: int,
    selected: int,
    downloaded: int,
    failed: int,
    skipped: int,
    t_start: datetime,
    t_end: datetime | None = None,
    transfer_workers: int = 0,
    selected_keys: list[str] | None = None,
    failed_keys: list[str] | None = None,
    errors: list[str] | None = None,
) -> DownloadRunReport:
    if t_end is None:
        t_end = datetime.now(UTC)
    return DownloadRunReport(
        run_id=run_id,
        mode=mode,
        requested_count=requested,
        selected_count=selected,
        downloaded_count=downloaded,
        skipped_count=skipped,
        failed_count=failed,
        started_at=t_start.isoformat(),
        completed_at=t_end.isoformat(),
        transfer_workers=transfer_workers,
        selected_keys=selected_keys or [],
        failed_keys=failed_keys or [],
        errors=errors or [],
    )


def _save_local_run_report(local_output_dir: Path, report: DownloadRunReport) -> None:
    """Save download run report to local output directory."""
    runs_dir = local_output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    report_data = {
        "run_id": report.run_id,
        "mode": report.mode,
        "requested_count": report.requested_count,
        "selected_count": report.selected_count,
        "downloaded_count": report.downloaded_count,
        "skipped_count": report.skipped_count,
        "failed_count": report.failed_count,
        "started_at": report.started_at,
        "completed_at": report.completed_at,
        "transfer_workers": report.transfer_workers,
        "selected_keys": report.selected_keys,
        "failed_keys": report.failed_keys,
        "errors": report.errors,
    }

    report_path = runs_dir / f"{report.run_id}.json"
    report_path.write_text(json.dumps(report_data, indent=2))
    logger.info("Download run report saved to %s", report_path)
