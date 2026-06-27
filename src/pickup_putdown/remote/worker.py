"""Worker module: download, run Tasks 3-5, encode, validate, upload for one source video."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WorkerResult:
    source_video_id: str
    source_key: str
    success: bool
    candidate_count: int = 0
    candidates: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    duration_s: float = 0.0


@dataclass
class WorkerConfig:
    storage_config: Path
    pipeline_config: Path
    work_dir: Path
    keep_local_files: bool = False
    person_model: str = "models/person_detector.pt"
    pose_model: str = "models/pose_detector.pt"
    triage_config: str = "configs/triage.yaml"
    tracker_config: str = "configs/bytetrack_triage.yaml"
    proposals_config: str = "configs/proposals.yaml"
    shelves_config: str = "configs/shelves.yaml"
    camera_id: str = "store_camera_01"
    defer_upload: bool = False
    local_source_dir: str = ".local/source_videos"
    local_output_dir: str = ".local/candidate_staging"


def run_source_video(
    source_rel_key: str,
    source_video_id: str,
    run_id: str,
    worker_cfg: WorkerConfig,
    storage: Any,
    local_source_dir: str | None = None,
) -> WorkerResult:
    """Process one source video through the full pipeline.

    Steps:
    1. Download source video (or use local copy in defer_upload mode)
    2. Run Tasks 3-5 (triage + propose)
    3. Encode candidates to H.264 MP4
    4. Validate encoded candidates
    5. Upload candidates and metadata (or stage locally in defer_upload mode)
    6. Return result for ledger update
    """
    t_start = datetime.now(UTC)
    work_dir = worker_cfg.work_dir / run_id / source_video_id
    source_dir = work_dir / "source"
    intermediate_dir = work_dir / "intermediate"
    candidates_dir = work_dir / "candidates"
    metadata_dir = work_dir / "metadata"

    for d in (source_dir, intermediate_dir, candidates_dir, metadata_dir):
        d.mkdir(parents=True, exist_ok=True)

    source_path = source_dir / f"{source_video_id}.mp4"

    try:
        # 1. Download source video or use local copy
        if worker_cfg.defer_upload and local_source_dir:
            src_path = Path(local_source_dir) / source_rel_key
            if not src_path.exists():
                raise RuntimeError(f"Local source not found: {src_path}")
            shutil.copy2(src_path, source_path)
            logger.info("Copied local source %s -> %s", src_path, source_path)
        else:
            logger.info("Downloading %s -> %s", source_rel_key, source_path)
            storage.download(storage.full_key(source_rel_key), source_path)

        # 2. Run Tasks 3-5
        logger.info("Running Tasks 3-5 for %s", source_video_id)
        task_output = run_tasks_3_5(
            video_path=source_path,
            output_dir=intermediate_dir,
            worker_cfg=worker_cfg,
        )
        candidate_count = task_output.get("n_candidates", 0)
        candidates_data = task_output.get("candidates", [])

        # 3. Encode candidates
        logger.info("Encoding %d candidate(s) for %s", candidate_count, source_video_id)
        encoded_candidates = encode_candidates(
            source_path=source_path,
            candidates_data=candidates_data,
            output_dir=candidates_dir,
            source_video_id=source_video_id,
        )

        # 4. Validate encoded candidates
        logger.info("Validating %d encoded candidate(s)", len(encoded_candidates))
        for enc in encoded_candidates:
            validation = enc["validation"]
            if not validation.valid:
                raise RuntimeError(
                    f"Candidate {enc['candidate_id']} validation failed: {validation.error}"
                )

        # 5. Upload candidates and metadata (or stage locally)
        if worker_cfg.defer_upload:
            _stage_candidates_locally(
                source_video_id=source_video_id,
                encoded_candidates=encoded_candidates,
                candidates_dir=candidates_dir,
                metadata_dir=metadata_dir,
                local_output_dir=worker_cfg.local_output_dir,
            )
        else:
            logger.info(
                "Uploading %d candidate(s) for %s", len(encoded_candidates), source_video_id
            )
            upload_candidates_and_metadata(
                storage=storage,
                source_rel_key=source_rel_key,
                source_video_id=source_video_id,
                encoded_candidates=encoded_candidates,
                candidates_dir=candidates_dir,
                metadata_dir=metadata_dir,
            )

        t_end = datetime.now(UTC)
        duration = (t_end - t_start).total_seconds()

        return WorkerResult(
            source_video_id=source_video_id,
            source_key=source_rel_key,
            success=True,
            candidate_count=len(encoded_candidates),
            candidates=[enc["metadata"] for enc in encoded_candidates],
            duration_s=duration,
        )

    except Exception as exc:
        t_end = datetime.now(UTC)
        duration = (t_end - t_start).total_seconds()
        logger.error("Worker failed for %s: %s", source_video_id, exc)
        if not worker_cfg.keep_local_files:
            _cleanup_work_dir(work_dir)
        return WorkerResult(
            source_video_id=source_video_id,
            source_key=source_rel_key,
            success=False,
            error=str(exc),
            duration_s=duration,
        )


def run_tasks_3_5(
    video_path: Path,
    output_dir: Path,
    worker_cfg: WorkerConfig,
) -> dict[str, Any]:
    """Run triage and propose for a single video."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Task 3: triage
    triage_cmd = [
        "pickup-putdown",
        "triage",
        str(video_path),
        "--config",
        worker_cfg.triage_config,
        "--tracker-config",
        worker_cfg.tracker_config,
        "--output-dir",
        str(output_dir / "task_3"),
        "--verbose",
    ]
    logger.info("Task 3: %s", " ".join(triage_cmd))
    result = subprocess.run(triage_cmd, capture_output=True, text=True, timeout=3600, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Task 3 failed: {result.stderr[-500:]}")

    # Task 5: propose
    propose_cmd = [
        "pickup-putdown",
        "propose",
        str(video_path),
        "--config",
        worker_cfg.proposals_config,
        "--shelves-config",
        worker_cfg.shelves_config,
        "--camera-id",
        worker_cfg.camera_id,
        "--person-tracks",
        str(output_dir / "task_3" / "tracks_person.parquet"),
        "--active-spans",
        str(output_dir / "task_3" / "active_spans.parquet"),
        "--output-dir",
        str(output_dir / "task_5"),
        "--verbose",
    ]
    logger.info("Task 5: %s", " ".join(propose_cmd))
    result = subprocess.run(propose_cmd, capture_output=True, text=True, timeout=3600, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Task 5 failed: {result.stderr[-500:]}")

    # Read candidates from task 5 output
    candidates_path = output_dir / "task_5" / "candidates.parquet"
    candidates_data = []
    if candidates_path.exists():
        import pyarrow.parquet as pq

        table = pq.read_table(str(candidates_path))
        candidates_data = table.to_pandas().to_dict("records")

    return {
        "n_candidates": len(candidates_data),
        "candidates": candidates_data,
    }


def encode_candidates(
    source_path: Path,
    candidates_data: list[dict[str, Any]],
    output_dir: Path,
    source_video_id: str,
) -> list[dict[str, Any]]:
    """Encode each candidate window to H.264 MP4."""
    from pickup_putdown.remote.encoding import (
        EncodingConfig,
        encode_candidate,
        validate_encoding,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    encoding_cfg = EncodingConfig()
    encoded: list[dict[str, Any]] = []

    for idx, cand in enumerate(candidates_data):
        candidate_id = cand.get("candidate_id", f"{source_video_id}_candidate_{idx:04d}")
        start_s = float(cand.get("window_start_s", 0))
        end_s = float(cand.get("window_end_s", 0))
        duration_s = end_s - start_s

        if duration_s <= 0:
            logger.warning("Skipping candidate %s with non-positive duration", candidate_id)
            continue

        tmp_input = output_dir / f"{candidate_id}_input.mp4"
        tmp_output = output_dir / f"{candidate_id}.mp4"

        # Extract candidate window from source
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg not found on PATH")

        extract_cmd = [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-ss",
            str(start_s),
            "-i",
            str(source_path),
            "-to",
            str(duration_s),
            "-c",
            "copy",
            str(tmp_input),
        ]
        subprocess.run(extract_cmd, capture_output=True, text=True, timeout=120, check=False)

        if not tmp_input.exists():
            logger.warning("Failed to extract candidate window for %s", candidate_id)
            continue

        # Re-encode to H.264
        encode_candidate(tmp_input, tmp_output, encoding_cfg)

        # Validate
        validation = validate_encoding(tmp_output)
        if not validation.valid:
            raise RuntimeError(
                f"Encoding validation failed for {candidate_id}: {validation.error}"
            )

        candidate_metadata = {
            "candidate_id": candidate_id,
            "source_start_s": round(start_s, 4),
            "source_end_s": round(end_s, 4),
            "duration_s": round(duration_s, 4),
            "codec": "h264",
            "pixel_format": "yuv420p",
            "actor_id": cand.get("actor_id"),
            "hand_side": cand.get("hand_side"),
            "region_id": cand.get("region_id"),
        }

        encoded.append(
            {
                "candidate_id": candidate_id,
                "local_path": tmp_output,
                "metadata": candidate_metadata,
                "validation": validation,
            }
        )

        # Clean up intermediate extract
        if tmp_input.exists():
            tmp_input.unlink()

    return encoded


def upload_candidates_and_metadata(
    storage: Any,
    source_rel_key: str,
    source_video_id: str,
    encoded_candidates: list[dict[str, Any]],
    candidates_dir: Path,
    metadata_dir: Path,
) -> None:
    """Upload encoded candidates and source metadata to S3."""
    # Upload each candidate
    for enc in encoded_candidates:
        candidate_id = enc["candidate_id"]
        local_path = enc["local_path"]
        dest_key = f"anon/candidates/videos/{source_video_id}/{candidate_id}.mp4"
        logger.info("Uploading candidate %s -> %s", candidate_id, dest_key)
        storage.upload(local_path, dest_key)

        # Update candidate metadata with S3 key
        enc["metadata"]["candidate_key"] = dest_key

    # Build and upload source metadata
    metadata = {
        "source_bucket": storage.bucket,
        "source_key": storage.full_key(source_rel_key),
        "source_video_id": source_video_id,
        "candidate_count": len(encoded_candidates),
        "candidates": [enc["metadata"] for enc in encoded_candidates],
        "processed_at": datetime.now(UTC).isoformat(),
    }

    metadata_path = metadata_dir / f"{source_video_id}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    metadata_dest = f"anon/candidates/metadata/{source_video_id}.json"
    logger.info("Uploading metadata -> %s", metadata_dest)
    storage.upload(metadata_path, metadata_dest)


def _stage_candidates_locally(
    source_video_id: str,
    encoded_candidates: list[dict[str, Any]],
    candidates_dir: Path,
    metadata_dir: Path,
    local_output_dir: str,
) -> None:
    """Copy encoded candidates and metadata to local staging directory."""
    staging_candidates = Path(local_output_dir) / "candidates" / source_video_id
    staging_candidates.mkdir(parents=True, exist_ok=True)

    for enc in encoded_candidates:
        candidate_id = enc["candidate_id"]
        local_path = Path(enc["local_path"])
        dest = staging_candidates / f"{candidate_id}.mp4"
        shutil.copy2(local_path, dest)
        enc["metadata"]["candidate_key"] = str(dest)
        logger.info("Staged candidate %s -> %s", candidate_id, dest)

    metadata = {
        "source_video_id": source_video_id,
        "candidate_count": len(encoded_candidates),
        "candidates": [enc["metadata"] for enc in encoded_candidates],
        "processed_at": datetime.now(UTC).isoformat(),
    }
    metadata_path = staging_candidates / f"{source_video_id}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Staged metadata -> %s", metadata_path)


def _cleanup_work_dir(work_dir: Path) -> None:
    """Remove local work directory."""
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
        logger.debug("Cleaned up work dir: %s", work_dir)
