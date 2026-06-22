"""Coordinator: parallel processing of source videos with bounded concurrency."""

from __future__ import annotations

import json
import logging
import queue
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CoordinationConfig:
    target_count: int
    workers: int = 4
    transfer_workers: int = 4
    gpu_workers: int = 1
    encode_workers: int = 4
    work_dir: str = ".local/remote_candidates"
    keep_local_files: bool = False
    fail_fast: bool = False
    overwrite: bool = False
    dry_run: bool = False
    defer_upload: bool = False
    local_output_dir: str = ".local/candidate_staging"
    local_source_dir: str = ".local/source_videos"


@dataclass
class RunReport:
    run_id: str
    requested_count: int
    selected_count: int
    completed_count: int
    failed_count: int
    skipped_count: int
    total_candidates: int
    start_time: str
    end_time: str
    concurrency: dict[str, int]
    failed_sources: list[dict[str, str]] = field(default_factory=list)


def _save_local_run_report(local_output_dir: str | Path, report: RunReport) -> None:
    """Save run report to local directory."""
    runs_dir = Path(local_output_dir) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    report_data = {
        "run_id": report.run_id,
        "requested_count": report.requested_count,
        "selected_count": report.selected_count,
        "completed_count": report.completed_count,
        "failed_count": report.failed_count,
        "skipped_count": report.skipped_count,
        "total_candidates": report.total_candidates,
        "start_time": report.start_time,
        "end_time": report.end_time,
        "concurrency": report.concurrency,
        "failed_sources": report.failed_sources,
    }
    report_path = runs_dir / f"{report.run_id}.json"
    report_path.write_text(json.dumps(report_data, indent=2))
    logger.info("Run report saved to %s", report_path)


def run_candidate_generation(
    storage: Any,
    ledger: Any,
    config: CoordinationConfig,
    worker_cfg: Any,
) -> RunReport:
    """Coordinate parallel processing of source videos.

    Flow:
    1. Discover source videos
    2. Sync ledger
    3. Select unprocessed videos
    4. Process in parallel with bounded concurrency
    5. Update ledger and upload run report
    """
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    t_start = datetime.now(UTC)

    logger.info("=== Candidate generation run %s ===", run_id)
    logger.info("Target count: %d", config.target_count)

    # 1. Discover source videos (skip in defer-upload mode — use local ledger)
    if config.defer_upload:
        ledger.load()
        ledger.save()
    else:
        from pickup_putdown.remote.discovery import discover_source_videos

        discovered = discover_source_videos(storage)

        # 2. Sync ledger
        ledger.load()
        ledger.sync_with_discovery(discovered)
        ledger.save()

    # 3. Select targets
    if config.defer_upload:
        selected = ledger.select_ready_for_generation(config.target_count)
    else:
        selected = ledger.select_unprocessed(config.target_count)

    if not selected:
        msg = (
            "No ready videos to process."
            if config.defer_upload
            else "No unprocessed videos to process."
        )
        logger.info(msg)
        report = _build_report(
            run_id=run_id,
            requested=config.target_count,
            selected=0,
            completed=0,
            failed=0,
            skipped=0,
            total_candidates=0,
            t_start=t_start,
            config=config,
        )
        if config.defer_upload:
            _save_local_run_report(Path(config.local_output_dir), report)
        else:
            _upload_run_report(storage, report)
        return report

    logger.info("Selected %d video(s) for processing", len(selected))

    # 4. Dry run
    if config.dry_run:
        logger.info("=== DRY RUN — no processing will occur ===")
        for entry in selected:
            logger.info("  Would process: %s", entry.file_name)
        logger.info("=== End dry run ===")
        report = _build_report(
            run_id=run_id,
            requested=config.target_count,
            selected=len(selected),
            completed=0,
            failed=0,
            skipped=len(selected),
            total_candidates=0,
            t_start=t_start,
            config=config,
        )
        if config.defer_upload:
            _save_local_run_report(Path(config.local_output_dir), report)
        else:
            _upload_run_report(storage, report)
        return report

    # 5. Process in parallel
    work_dir = Path(config.work_dir)
    worker_cfg.work_dir = work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    local_source_dir = Path(config.local_source_dir) if config.defer_upload else None
    if config.defer_upload:
        local_source_dir.mkdir(parents=True, exist_ok=True)

    completed = 0
    failed = 0
    total_candidates = 0
    failed_sources: list[dict[str, str]] = []

    # Use ThreadPoolExecutor for parallel processing
    # gpu_workers controls GPU-bound inference concurrency
    max_workers = min(config.workers, len(selected))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for entry in selected:
            source_video_id = _make_source_video_id(entry.file_name)
            future = executor.submit(
                _process_single_source,
                entry.file_name,
                source_video_id,
                run_id,
                worker_cfg,
                storage,
                ledger,
                config,
                local_source_dir=str(local_source_dir) if config.defer_upload else None,
            )
            futures[future] = entry.file_name

        for future in as_completed(futures):
            source_key = futures[future]
            try:
                result = future.result(timeout=1800)
                if result.success:
                    completed += 1
                    total_candidates += result.candidate_count
                    if config.defer_upload:
                        ledger.mark_generated(source_key)
                    else:
                        ledger.mark_processed(source_key)
                    ledger.save()
                    if not config.keep_local_files:
                        from pickup_putdown.remote.worker import _cleanup_work_dir

                        work_path = work_dir / run_id / result.source_video_id
                        _cleanup_work_dir(work_path)
                else:
                    failed += 1
                    failed_sources.append(
                        {
                            "source_key": source_key,
                            "error": result.error[:500],
                        }
                    )
                    if config.fail_fast:
                        logger.error("Fail-fast: cancelling remaining jobs")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
            except Exception as exc:
                failed += 1
                failed_sources.append(
                    {
                        "source_key": source_key,
                        "error": str(exc)[:500],
                    }
                )
                logger.error("Unhandled exception for %s: %s", source_key, exc)
                if config.fail_fast:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

    t_end = datetime.now(UTC)

    report = _build_report(
        run_id=run_id,
        requested=config.target_count,
        selected=len(selected),
        completed=completed,
        failed=failed,
        skipped=max(0, len(selected) - completed - failed),
        total_candidates=total_candidates,
        t_start=t_start,
        t_end=t_end,
        config=config,
        failed_sources=failed_sources,
    )

    if config.defer_upload:
        _save_local_run_report(Path(config.local_output_dir), report)
    else:
        _upload_run_report(storage, report)

    logger.info("=== Run complete ===")
    logger.info("Completed: %d, Failed: %d, Candidates: %d", completed, failed, total_candidates)

    return report


def _process_single_source(
    source_rel_key: str,
    source_video_id: str,
    run_id: str,
    worker_cfg: Any,
    storage: Any,
    ledger: Any,
    config: Any,
    local_source_dir: str | None = None,
) -> Any:
    """Process a single source video. Returns WorkerResult."""
    from pickup_putdown.remote.worker import run_source_video

    return run_source_video(
        source_rel_key=source_rel_key,
        source_video_id=source_video_id,
        run_id=run_id,
        worker_cfg=worker_cfg,
        storage=storage,
        local_source_dir=local_source_dir,
    )


def _make_source_video_id(rel_key: str) -> str:
    """Create a stable, readable source video ID from a relative key.

    e.g. camera_01/video_001.mp4 -> camera_01_video_001
    """
    from pathlib import Path

    stem = Path(rel_key).stem
    parent = str(Path(rel_key).parent)
    if parent and parent != ".":
        parts = parent.replace("/", "_")
        return f"{parts}_{stem}"
    return stem


def _build_report(
    run_id: str,
    requested: int,
    selected: int,
    completed: int,
    failed: int,
    skipped: int,
    total_candidates: int,
    t_start: datetime,
    t_end: datetime | None = None,
    config: CoordinationConfig | None = None,
    failed_sources: list[dict[str, str]] | None = None,
) -> RunReport:
    if t_end is None:
        t_end = datetime.now(UTC)
    concurrency = {}
    if config:
        concurrency = {
            "workers": config.workers,
            "transfer_workers": config.transfer_workers,
            "gpu_workers": config.gpu_workers,
            "encode_workers": config.encode_workers,
        }
    return RunReport(
        run_id=run_id,
        requested_count=requested,
        selected_count=selected,
        completed_count=completed,
        failed_count=failed,
        skipped_count=skipped,
        total_candidates=total_candidates,
        start_time=t_start.isoformat(),
        end_time=t_end.isoformat(),
        concurrency=concurrency,
        failed_sources=failed_sources or [],
    )


def _upload_run_report(storage: Any, report: RunReport) -> None:
    """Upload run report to S3."""
    report_data = {
        "run_id": report.run_id,
        "requested_count": report.requested_count,
        "selected_count": report.selected_count,
        "completed_count": report.completed_count,
        "failed_count": report.failed_count,
        "skipped_count": report.skipped_count,
        "total_candidates": report.total_candidates,
        "start_time": report.start_time,
        "end_time": report.end_time,
        "concurrency": report.concurrency,
        "failed_sources": report.failed_sources,
    }
    tmp = Path(f"/tmp/_run_report_{report.run_id}.json")
    tmp.write_text(json.dumps(report_data, indent=2))
    dest_key = f"anon/candidates/runs/{report.run_id}.json"
    storage.upload(tmp, dest_key)
    tmp.unlink(missing_ok=True)
    logger.info("Run report uploaded to %s", dest_key)


def run_two_stage_pipeline(
    entries: list[Any],
    worker_cfg: Any,
    local_source_dir: Path,
    local_output_dir: Path,
    encode_workers: int = 4,
) -> RunReport:
    """Run GPU-inference sequentially and encoding in parallel.

    Stage 1 (GPU, sequential): triage + propose for each video.
    Stage 2 (CPU, parallel): encode candidates and stage to disk.

    Uses a queue between stages so CPU encoding overlaps with GPU inference.
    """
    run_id = f"local_{uuid.uuid4().hex[:12]}"
    t_start = datetime.now(UTC)

    logger.info("=== Two-stage local run %s ===", run_id)
    logger.info("Videos: %d, encode workers: %d", len(entries), encode_workers)

    work_dir = Path(worker_cfg.work_dir) / run_id
    work_dir.mkdir(parents=True, exist_ok=True)

    inter_queue: queue.Queue = queue.Queue()

    failed_sources: list[dict[str, str]] = []

    # Stage 2: CPU encoding workers (pull from queue)
    sentinel = None
    counters = {"completed": 0, "failed": 0, "total_candidates": 0}

    def _encode_worker() -> None:
        while True:
            item = inter_queue.get()
            if item is sentinel:
                inter_queue.task_done()
                break
            try:
                result = item
                if isinstance(result, Exception):
                    raise result
                source_video_id = result["source_video_id"]
                source_path = result["source_path"]
                candidates_data = result["candidates_data"]
                output_dir = result["output_dir"]

                from pickup_putdown.remote.worker import (
                    _stage_candidates_locally,
                    encode_candidates,
                )

                encoded = encode_candidates(
                    source_path=source_path,
                    candidates_data=candidates_data,
                    output_dir=output_dir / "candidates",
                    source_video_id=source_video_id,
                )

                _stage_candidates_locally(
                    source_video_id=source_video_id,
                    encoded_candidates=encoded,
                    candidates_dir=output_dir / "candidates",
                    metadata_dir=output_dir / "metadata",
                    local_output_dir=str(local_output_dir),
                )
                counters["total_candidates"] += len(encoded)
                counters["completed"] += 1
                logger.info("CPU: %s encoded %d candidate(s)", source_video_id, len(encoded))
            except Exception as exc:
                counters["failed"] += 1
                sid = (
                    result.get("source_video_id", "unknown")
                    if not isinstance(result, Exception)
                    else "unknown"
                )
                failed_sources.append({"source_key": sid, "error": str(exc)[:500]})
                logger.error("CPU worker failed for %s: %s", sid, exc)
            finally:
                inter_queue.task_done()

    with ThreadPoolExecutor(max_workers=encode_workers) as cpu_pool:
        for _ in range(encode_workers):
            cpu_pool.submit(_encode_worker)

        # Stage 1: GPU inference (sequential)
        from pickup_putdown.remote.worker import run_tasks_3_5

        for entry in entries:
            source_rel_key = entry.file_name
            source_video_id = _make_source_video_id(source_rel_key)
            source_path = local_source_dir / source_rel_key

            if not source_path.exists():
                logger.error("Source missing: %s", source_path)
                counters["failed"] += 1
                failed_sources.append(
                    {"source_key": source_rel_key, "error": f"Source missing: {source_path}"}
                )
                continue

            output_dir = work_dir / source_video_id
            output_dir.mkdir(parents=True, exist_ok=True)

            try:
                logger.info("GPU: processing %s", source_video_id)
                task_output = run_tasks_3_5(
                    video_path=source_path,
                    output_dir=output_dir / "intermediate",
                    worker_cfg=worker_cfg,
                )
                candidates_data = task_output.get("candidates", [])
                logger.info(
                    "GPU: %s produced %d candidate(s)", source_video_id, len(candidates_data)
                )

                inter_queue.put(
                    {
                        "source_video_id": source_video_id,
                        "source_path": source_path,
                        "candidates_data": candidates_data,
                        "output_dir": output_dir,
                    }
                )
            except Exception as exc:
                logger.error("GPU: %s failed: %s", source_video_id, exc)
                counters["failed"] += 1
                failed_sources.append({"source_key": source_rel_key, "error": str(exc)[:500]})
                inter_queue.put(exc)

        # Wait for GPU stage to finish feeding queue
        inter_queue.join()

        # Send sentinels to stop CPU workers
        for _ in range(encode_workers):
            inter_queue.put(sentinel)
        inter_queue.join()

    t_end = datetime.now(UTC)

    report = _build_report(
        run_id=run_id,
        requested=len(entries),
        selected=len(entries),
        completed=counters["completed"],
        failed=counters["failed"],
        skipped=0,
        total_candidates=counters["total_candidates"],
        t_start=t_start,
        t_end=t_end,
        config=None,
        failed_sources=failed_sources,
    )

    logger.info("=== Two-stage run complete ===")
    logger.info(
        "Completed: %d, Failed: %d, Candidates: %d",
        counters["completed"],
        counters["failed"],
        counters["total_candidates"],
    )

    return report
