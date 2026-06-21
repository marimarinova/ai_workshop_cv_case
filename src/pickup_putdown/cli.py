"""CLI entry point for the pickup-putdown package."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(
    name="pickup-putdown",
    help="Pickup and putdown temporal action detection in store video.",
    add_completion=False,
)

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    """Configure root logger based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@app.command()
def ingest(
    config: str = typer.Option(
        "configs/storage.yaml",
        "--config",
        "-c",
        help="Path to storage configuration YAML file.",
    ),
    output_dir: str = typer.Option(
        "data",
        "--output-dir",
        "-o",
        help="Directory for output files (clips.parquet, clips.csv).",
    ),
    cache_dir: str = typer.Option(
        "cache/downloads",
        "--cache-dir",
        help="Directory for the download cache.",
    ),
    max_cache_mb: int = typer.Option(
        5120,
        "--max-cache-mb",
        help="Maximum cache size in megabytes.",
    ),
    max_cache_count: int = typer.Option(
        50,
        "--max-cache-count",
        help="Maximum number of files in the cache.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="List and probe objects without downloading anything.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Index a cloud storage bucket, probe video metadata, and export clip registry.

    This command lists objects in the configured S3 bucket, extracts metadata
    via ffprobe for each video, detects duplicates, manages a bounded local
    download cache, and emits clips.parquet + clips.csv.
    """
    _setup_logging(verbose)

    from pickup_putdown.config import load_config
    from pickup_putdown.ingestion.cache import DownloadCache
    from pickup_putdown.ingestion.clip_registry import ClipRegistry, generate_clip_id
    from pickup_putdown.ingestion.index_bucket import list_objects
    from pickup_putdown.ingestion.video_probe import probe_video

    cfg_path = Path(config)
    cfg = load_config(cfg_path)

    storage = cfg.storage
    bucket_uri = storage.bucket_uri or typer.prompt("S3 bucket URI (s3://bucket/prefix)")
    if not bucket_uri:
        typer.echo("Error: bucket_uri is required.", err=True)
        raise SystemExit(1)

    # Step 1: List objects
    typer.echo(f"Listing objects in {bucket_uri}...")
    try:
        objects = list_objects(
            bucket_uri,
            endpoint_url=storage.endpoint_url,
            region=storage.region,
            anonymous=storage.anonymous,
        )
    except Exception as exc:
        typer.echo(f"Error listing objects: {exc}", err=True)
        raise SystemExit(1) from exc

    if not objects:
        typer.echo("No objects found in bucket. Exiting.")
        raise SystemExit(0)

    typer.echo(f"Found {len(objects)} objects.")

    # Extract bucket name for downloads
    import re

    bucket_match = re.match(r"^s3://([^/]+)/(.*)$", bucket_uri)
    bucket_name = bucket_match.group(1) if bucket_match else ""

    # Step 2: Build registry and detect duplicates
    registry = ClipRegistry()
    seen_signatures: dict[tuple[str, int], str] = {}  # (etag, size) -> clip_id

    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    skipped_non_video = 0
    skipped_probe_fail = 0
    decoded_ok_count = 0

    for obj in objects:
        # Skip non-video files
        if Path(obj.key).suffix.lower() not in video_extensions:
            skipped_non_video += 1
            continue

        clip_id = generate_clip_id(obj.key, obj.etag, obj.size)

        # Duplicate detection
        sig = (obj.etag or "", obj.size)
        if sig in seen_signatures and seen_signatures[sig] != clip_id:
            existing_id = seen_signatures[sig]
            # Keep the first one; mark this as duplicate
            dup_clip = registry.get_clip(existing_id)
            if dup_clip is not None:
                dup_clip.duplicate_of = clip_id
            continue
        seen_signatures[sig] = clip_id

        if dry_run:
            # In dry-run mode, create a minimal clip without probing
            clip = registry.get_clip(clip_id)
            if clip is None:
                from pickup_putdown.common.schemas import Clip as ClipSchema

                clip = ClipSchema(
                    clip_id=clip_id,
                    s3_key=obj.key,
                    duration_s=0.0,
                    fps=0.0,
                    width=0,
                    height=0,
                    object_size_bytes=obj.size,
                    etag=obj.etag,
                    decode_ok=True,
                )
                registry.add_clip(clip)
            continue

        # Step 3: Download to cache (or skip if cache not configured)
        cache = DownloadCache(cache_dir, max_size_mb=max_cache_mb, max_count=max_cache_count)
        cache.set_download_fn(lambda sk, lp, b=bucket_name: _s3_download_fn(sk, lp, b))
        try:
            local_path = cache.get(obj.key)
        except Exception as exc:
            typer.echo(f"  Download failed for {obj.key}: {exc}", err=True)
            skipped_probe_fail += 1
            continue

        # Step 4: Probe metadata
        result = probe_video(local_path)
        if not result.decode_ok:
            typer.echo(f"  Decode failed for {obj.key}: {result.probe_error}", err=True)
            skipped_probe_fail += 1

        from pickup_putdown.common.schemas import Clip as ClipSchema

        clip = ClipSchema(
            clip_id=clip_id,
            s3_key=obj.key,
            duration_s=result.duration_s or 0.0,
            fps=result.fps or 0.0,
            width=result.width or 0,
            height=result.height or 0,
            object_size_bytes=obj.size,
            etag=obj.etag,
            video_codec=result.video_codec,
            audio_codec=result.audio_codec,
            decode_ok=result.decode_ok,
            probe_error=result.probe_error,
            probe_fps=result.probe_fps,
        )
        registry.add_clip(clip)
        if result.decode_ok:
            decoded_ok_count += 1

    if skipped_non_video:
        typer.echo(f"Skipped {skipped_non_video} non-video objects.")

    # Step 3: Export
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_path = output_path / "clips.parquet"
    csv_path = output_path / "clips.csv"

    registry.export_parquet(parquet_path)
    registry.export_csv(csv_path)

    # Step 4: Summary
    cache_usage: dict[str, Any] = {}
    if not dry_run:
        try:
            cache_usage = cache.usage()
        except Exception:
            pass  # noqa: SIM105, RET505

    summary = registry.summary(cache_usage if cache_usage else None)
    typer.echo("")
    typer.echo("=== Ingestion Summary ===")
    typer.echo(f"  Indexed clips:       {summary['indexed_count']}")
    typer.echo(f"  Decode failures:     {summary['failures']}")
    typer.echo(f"  Duplicate candidates:{summary['duplicate_candidates']}")
    typer.echo(f"  Total source bytes:  {summary['total_source_bytes']}")
    if cache_usage:
        typer.echo(
            f"  Cache used:          {cache_usage.get('used_mb', '?')} MB / {cache_usage.get('max_mb', '?')} MB"
        )
    typer.echo(f"  Parquet:             {parquet_path}")
    typer.echo(f"  CSV:                 {csv_path}")


def _s3_download_fn(s3_key: str, local_path: Path, bucket: str) -> None:
    """Download an object from S3.

    Parameters
    ----------
    s3_key : str
        The S3 object key (without bucket prefix).
    local_path : Path
        Destination path on disk.
    bucket : str
        The S3 bucket name.
    """
    import boto3

    from pickup_putdown.config import load_config

    cfg = load_config()
    storage = cfg.storage

    client_kwargs = {}
    if storage.endpoint_url:
        client_kwargs["endpoint_url"] = storage.endpoint_url
    if storage.region:
        client_kwargs["region_name"] = storage.region
    if storage.anonymous:
        client_kwargs["aws_access_key_id"] = ""
        client_kwargs["aws_secret_access_key"] = ""
        client_kwargs["aws_session_token"] = ""

    client = boto3.client("s3", **client_kwargs)  # noqa: S301

    local_path.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, s3_key, str(local_path))


if __name__ == "__main__":
    app()


def main() -> None:
    """Entry point for the pickup-putdown CLI."""
    app()


# ---------------------------------------------------------------------------
# Triage command
# ---------------------------------------------------------------------------


def _resolve_video_paths(input_arg: str, output_dir: str) -> list[Path]:
    """Resolve input argument to a sorted list of video paths.

    Parameters
    ----------
    input_arg : str
        Path to a single video file or a directory of videos.
    output_dir : str
        Output directory (used to derive output paths).

    Returns
    -------
    list[Path]
        Sorted list of video file paths.
    """
    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    inp = Path(input_arg)

    if inp.is_file():
        return [inp]

    if inp.is_dir():
        videos = sorted([f for f in inp.iterdir() if f.suffix.lower() in video_extensions])
        if not videos:
            typer.echo(f"No video files found in {inp}", err=True)
            raise SystemExit(1)
        return videos

    typer.echo(f"Input not found: {input_arg}", err=True)
    raise SystemExit(1)


@app.command()
def triage(
    input_path: str = typer.Argument(..., help="Path to a video file or directory of videos."),
    config: str = typer.Option(
        "configs/triage.yaml",
        "--config",
        "-c",
        help="Path to triage configuration YAML file.",
    ),
    tracker_config: str = typer.Option(
        "configs/bytetrack_triage.yaml",
        "--tracker-config",
        help="Path to ByteTrack configuration YAML file.",
    ),
    output_dir: str = typer.Option(
        "outputs",
        "--output-dir",
        "-o",
        help="Base directory for output files.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Run person detection and tracking on video files to derive active spans.

    Processes each input video with YOLO person detection + ByteTrack at the
    configured target FPS. Produces tracks_person.parquet, active_spans.parquet,
    an updated clips.parquet, and a triage sampling report.
    """
    _setup_logging(verbose)

    import hashlib
    import json
    import subprocess
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml

    from pickup_putdown.config import load_config
    from pickup_putdown.ingestion.video_probe import probe_video

    cfg_path = Path(config)
    cfg = load_config(cfg_path)

    # Load tracker config
    tracker_cfg_path = Path(tracker_config)
    if tracker_cfg_path.exists():
        with open(tracker_cfg_path) as fh:
            yaml.safe_load(fh)  # validated, not used further
    triage_cfg = cfg.triage
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Resolve video paths
    video_paths = _resolve_video_paths(input_path, output_dir)
    typer.echo(f"Triage: {len(video_paths)} video(s) to process.")

    all_observations: list[dict] = []
    all_spans: list[dict] = []
    clip_updates: dict[str, dict] = {}
    clip_durations: dict[str, float] = {}
    clip_fps: dict[str, float] = {}
    report_rows: list[dict] = []

    for vp in video_paths:
        stem = vp.stem
        clip_id = f"clip_{stem}"
        typer.echo(f"  Processing {vp.name}...")

        # Probe clip metadata
        probe = probe_video(vp)
        if not probe.decode_ok:
            typer.echo(f"    Decode failed: {probe.probe_error}")
            clip_updates[clip_id] = {
                "clip_id": clip_id,
                "n_person_tracks": 0,
                "has_person": False,
                "active_start_s": None,
                "active_end_s": None,
            }
            report_rows.append(
                {
                    "clip_id": clip_id,
                    "decision": "decode_failed",
                    "source_duration_s": 0.0,
                    "target_fps": triage_cfg.target_fps,
                    "effective_sample_fps": 0.0,
                    "n_raw_tracks": 0,
                    "n_stable_tracks": 0,
                    "n_observations": 0,
                    "preview_path": None,
                }
            )
            continue

        duration_s = probe.duration_s or 0.0
        fps = probe.fps or 0.0
        clip_durations[clip_id] = duration_s
        clip_fps[clip_id] = fps

        # Run person tracker
        from pickup_putdown.perception.person_tracker import PersonTracker

        tracker = PersonTracker(video_path=vp, triage_cfg=triage_cfg)
        observations, summaries = tracker.run()

        obs_dicts = [o.model_dump() for o in observations]
        all_observations.extend(obs_dicts)

        # Derive active spans
        from pickup_putdown.perception.active_spans import (
            compute_clip_summary,
            derive_active_spans,
        )

        effective_fps = (
            fps / max(1, round(fps / triage_cfg.target_fps)) if fps > 0 else triage_cfg.target_fps
        )
        spans = derive_active_spans(
            observations,
            clip_id=clip_id,
            clip_duration_s=duration_s,
            merge_gap_s=triage_cfg.merge_gap_s,
            effective_sample_fps=effective_fps,
        )

        span_dicts = [s.model_dump() for s in spans]
        all_spans.extend(span_dicts)

        # Compute clip summary
        summary = compute_clip_summary(observations, spans)
        clip_updates[clip_id] = {
            "clip_id": clip_id,
            "n_person_tracks": summary["n_person_tracks"],
            "has_person": summary["has_person"],
            "active_start_s": summary["active_start_s"],
            "active_end_s": summary["active_end_s"],
        }

        # Preview rendering (optional, for QA)
        preview_path = None
        if spans or triage_cfg.preview_sample_rate > 0:
            from pickup_putdown.perception.previews import OverlayConfig, render_triage_preview

            preview_dir = output_base / "triage_previews"
            preview_path = str(preview_dir / f"{clip_id}.mp4")
            try:
                from pickup_putdown.common.schemas import ActiveSpan

                span_objs = [ActiveSpan(**sd) for sd in span_dicts]
                render_triage_preview(
                    vp,
                    observations,
                    span_objs,
                    Path(preview_path),
                    OverlayConfig(),
                )
            except Exception as exc:
                logger.debug("Preview failed for %s: %s", clip_id, exc)
                preview_path = None

        # Sampling report row
        n_raw = len(summaries)
        n_stable = sum(1 for s in summaries if s.is_stable)
        report_rows.append(
            {
                "clip_id": clip_id,
                "decision": "person_detected" if summary["has_person"] else "no_person",
                "source_duration_s": duration_s,
                "target_fps": triage_cfg.target_fps,
                "effective_sample_fps": effective_fps,
                "n_raw_tracks": n_raw,
                "n_stable_tracks": n_stable,
                "n_observations": len(observations),
                "preview_path": preview_path,
            }
        )

        typer.echo(
            f"    {summary['n_person_tracks']} stable tracks, "
            f"{len(spans)} active spans, has_person={summary['has_person']}"
        )

    # Write tracks_person.parquet
    if all_observations:
        obs_table = pa.Table.from_pylist(all_observations)
    else:
        obs_schema = pa.schema(
            [
                ("clip_id", pa.string()),
                ("person_track_id", pa.string()),
                ("tracker_track_id", pa.int64()),
                ("sample_index", pa.int64()),
                ("source_frame_index", pa.int64()),
                ("timestamp_s", pa.float64()),
                ("bbox_x1", pa.float64()),
                ("bbox_y1", pa.float64()),
                ("bbox_x2", pa.float64()),
                ("bbox_y2", pa.float64()),
                ("confidence", pa.float64()),
                ("is_stable", pa.bool_()),
            ]
        )
        obs_table = pa.Table.from_pydict(
            {
                "clip_id": [],
                "person_track_id": [],
                "tracker_track_id": [],
                "sample_index": [],
                "source_frame_index": [],
                "timestamp_s": [],
                "bbox_x1": [],
                "bbox_y1": [],
                "bbox_x2": [],
                "bbox_y2": [],
                "confidence": [],
                "is_stable": [],
            },
            schema=obs_schema,
        )
    pq.write_table(obs_table, str(output_base / "tracks_person.parquet"))

    # Write active_spans.parquet
    if all_spans:
        spans_table = pa.Table.from_pylist(all_spans)
    else:
        spans_schema = pa.schema(
            [
                ("clip_id", pa.string()),
                ("active_span_id", pa.string()),
                ("t_start", pa.float64()),
                ("t_end", pa.float64()),
                ("n_person_tracks", pa.int64()),
            ]
        )
        spans_table = pa.Table.from_pydict(
            {
                "clip_id": [],
                "active_span_id": [],
                "t_start": [],
                "t_end": [],
                "n_person_tracks": [],
            },
            schema=spans_schema,
        )
    pq.write_table(spans_table, str(output_base / "active_spans.parquet"))

    # Update clips.parquet
    clips_path = output_base / "clips.parquet"
    if clips_path.exists():
        clips_table = pq.read_table(str(clips_path))
        clips_df = clips_table.to_pandas()
        for cid, updates in clip_updates.items():
            mask = clips_df["clip_id"] == cid
            if mask.any():
                for k, v in updates.items():
                    clips_df.loc[mask, k] = v
        pq.write_table(pa.Table.from_pandas(clips_df, preserve_index=False), str(clips_path))
    else:
        # Create minimal clips.parquet from clip_updates
        clips_data = list(clip_updates.values())
        for cd in clips_data:
            cd["s3_key"] = f"local://{clip_durations.get(cd['clip_id'], 0)}"
            cd["duration_s"] = clip_durations.get(cd["clip_id"], 0.0)
            cd["fps"] = clip_fps.get(cd["clip_id"], 0.0)
            cd["width"] = 0
            cd["height"] = 0
            cd["decode_ok"] = True
        clips_table = pa.Table.from_pylist(clips_data)
        pq.write_table(clips_table, str(clips_path))

    # Write sampling report
    from pickup_putdown.perception.sampling_report import (
        generate_sampling_report,
        write_sampling_report,
    )

    full_report = generate_sampling_report(
        report_rows, triage_cfg.sampling_seed, triage_cfg.preview_sample_rate
    )
    write_sampling_report(full_report, output_base / "triage_sampling_report.parquet")

    # Write run metadata

    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        git_commit = "unknown"

    try:
        model_hash = hashlib.md5(Path(triage_cfg.model_path).read_bytes()).hexdigest()
    except Exception:
        model_hash = "unknown"

    metadata = {
        "git_commit": git_commit,
        "model_path": triage_cfg.model_path,
        "model_hash": model_hash,
        "triage_config": config,
        "tracker_config": tracker_config,
        "n_videos": len(video_paths),
        "n_observations": len(all_observations),
        "n_active_spans": len(all_spans),
        "target_fps": triage_cfg.target_fps,
        "sampling_seed": triage_cfg.sampling_seed,
    }
    with open(output_base / "triage_run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    typer.echo("")
    typer.echo("=== Triage Summary ===")
    typer.echo(f"  Videos processed:    {len(video_paths)}")
    typer.echo(f"  Total observations:  {len(all_observations)}")
    typer.echo(f"  Total active spans:  {len(all_spans)}")
    typer.echo(f"  Tracks parquet:      {output_base / 'tracks_person.parquet'}")
    typer.echo(f"  Spans parquet:       {output_base / 'active_spans.parquet'}")
    typer.echo(f"  Clips parquet:       {clips_path}")
    typer.echo(f"  Sampling report:     {output_base / 'triage_sampling_report.parquet'}")
    typer.echo(f"  Run metadata:        {output_base / 'triage_run_metadata.json'}")


# ---------------------------------------------------------------------------
# Triage comparison command
# ---------------------------------------------------------------------------


@app.command()
def triage_comparison(
    input_path: str = typer.Argument(..., help="Path to a directory of videos for comparison."),
    config: str = typer.Option(
        "configs/triage.yaml",
        "--config",
        "-c",
        help="Path to triage configuration YAML file.",
    ),
    output_dir: str = typer.Option(
        "outputs",
        "--output-dir",
        "-o",
        help="Directory for comparison output.",
    ),
    max_clips: int = typer.Option(
        5,
        "--max-clips",
        help="Maximum number of clips to compare.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Run 1 FPS vs 2 FPS triage comparison on a subset of videos.

    Produces a machine-readable comparison CSV showing how detection decisions
    and track counts differ between sampling rates.
    """
    _setup_logging(verbose)

    from pathlib import Path

    from pickup_putdown.config import load_config

    cfg_path = Path(config)
    cfg = load_config(cfg_path)

    video_paths = _resolve_video_paths(input_path, output_dir)
    typer.echo(f"Comparison: {min(len(video_paths), max_clips)} video(s) to test.")

    from pickup_putdown.perception.comparison import run_fps_comparison

    df = run_fps_comparison(video_paths, cfg.triage, max_clips=max_clips)

    out_path = Path(output_dir) / "fps_comparison.csv"
    df.to_csv(out_path, index=False)

    typer.echo("")
    typer.echo("=== FPS Comparison Summary ===")
    typer.echo(f"  Clips compared:      {len(df)}")
    typer.echo(f"  Decisions changed:   {df['decision_changed'].sum()}")
    typer.echo(f"  Output:              {out_path}")
