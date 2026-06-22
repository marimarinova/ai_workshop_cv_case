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


# ---------------------------------------------------------------------------
# Propose command (Layer 0B)
# ---------------------------------------------------------------------------


@app.command()
def propose(
    input_path: str = typer.Argument(
        ...,
        help="Path to a video file or directory of videos.",
    ),
    config: str = typer.Option(
        "configs/proposals.yaml",
        "--config",
        "-c",
        help="Path to proposals configuration YAML file.",
    ),
    shelves_config: str = typer.Option(
        "configs/shelves.yaml",
        "--shelves-config",
        help="Path to shelf/surface region configuration YAML file.",
    ),
    camera_id: str | None = typer.Option(
        None,
        "--camera-id",
        help=(
            "Camera ID from the shelf configuration. Required when the "
            "configuration contains more than one camera."
        ),
    ),
    person_tracks: str | None = typer.Option(
        None,
        "--person-tracks",
        "-p",
        help="Path to tracks_person.parquet produced by triage.",
    ),
    active_spans: str | None = typer.Option(
        None,
        "--active-spans",
        "-a",
        help=(
            "Path to active_spans.parquet produced by triage. When omitted "
            "and --person-tracks is provided, a sibling active_spans.parquet "
            "is auto-detected when present."
        ),
    ),
    output_dir: str = typer.Option(
        "outputs",
        "--output-dir",
        "-o",
        help="Base directory for output files.",
    ),
    render_previews: bool = typer.Option(
        False,
        "--render-previews",
        "-r",
        help="Render candidate preview clips.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Run pose inference and generate actor-specific interaction candidates.

    Produces tracks_pose.parquet, candidates.parquet, and optionally candidate
    preview clips. Candidates are proposals only and are never written to
    events.csv or predictions.csv.
    """
    _setup_logging(verbose)

    import hashlib
    import json
    import math
    import subprocess

    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml

    from pickup_putdown.common.schemas import (
        ActiveSpan,
        PersonObservation,
    )
    from pickup_putdown.config import load_config
    from pickup_putdown.ingestion.video_probe import probe_video
    from pickup_putdown.perception.proposals import (
        associate_poses_with_actors,
        detect_raw_interactions,
        generate_candidates,
    )
    from pickup_putdown.perception.shelf_regions import (
        get_expanded_regions,
        get_regions_for_camera,
        load_shelf_config,
    )

    cfg_path = Path(config)
    cfg = load_config(cfg_path)

    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Shelf and camera configuration
    # ------------------------------------------------------------------
    shelf_cfg_path = Path(shelves_config)
    if not shelf_cfg_path.is_file():
        raise typer.BadParameter(
            f"Shelf configuration not found: {shelf_cfg_path}",
            param_hint="--shelves-config",
        )

    shelf_cfg = load_shelf_config(shelf_cfg_path)

    if camera_id is None:
        available_camera_ids = list(shelf_cfg.cameras)
        if len(available_camera_ids) != 1:
            raise typer.BadParameter(
                "The shelf configuration contains multiple cameras; provide "
                f"--camera-id. Available values: {available_camera_ids}",
                param_hint="--camera-id",
            )
        selected_camera_id = available_camera_ids[0]
    else:
        selected_camera_id = camera_id

    if selected_camera_id not in shelf_cfg.cameras:
        raise typer.BadParameter(
            f"Unknown camera ID {selected_camera_id!r}. Available values: "
            f"{list(shelf_cfg.cameras)}",
            param_hint="--camera-id",
        )

    camera_config = get_regions_for_camera(
        shelf_cfg,
        selected_camera_id,
    )
    expanded_regions = get_expanded_regions(camera_config)
    original_regions = {region.region_id: region.points for region in camera_config.regions}

    typer.echo(f"Using shelf camera: {selected_camera_id}")

    # ------------------------------------------------------------------
    # Task 3 inputs
    # ------------------------------------------------------------------
    person_observations: list[dict[str, Any]] = []
    person_tracks_path: Path | None = None

    if person_tracks is not None:
        person_tracks_path = Path(person_tracks)
        if not person_tracks_path.is_file():
            raise typer.BadParameter(
                f"Person tracks file not found: {person_tracks_path}",
                param_hint="--person-tracks",
            )

        person_table = pq.read_table(person_tracks_path)
        person_observations = person_table.to_pandas().to_dict("records")
        typer.echo(f"Loaded {len(person_observations)} person observations.")

    active_spans_path: Path | None = Path(active_spans) if active_spans is not None else None

    if active_spans_path is None and person_tracks_path is not None:
        sibling_spans = person_tracks_path.with_name("active_spans.parquet")
        if sibling_spans.is_file():
            active_spans_path = sibling_spans
            typer.echo(f"Auto-detected active spans: {active_spans_path}")

    active_span_records: list[dict[str, Any]] = []
    if active_spans_path is not None:
        if not active_spans_path.is_file():
            raise typer.BadParameter(
                f"Active spans file not found: {active_spans_path}",
                param_hint="--active-spans",
            )

        span_table = pq.read_table(active_spans_path)
        active_span_records = span_table.to_pandas().to_dict("records")
        typer.echo(f"Loaded {len(active_span_records)} active spans.")
    else:
        logger.warning(
            "No active-spans file supplied or auto-detected; pose "
            "inference will process the full clip."
        )

    # ------------------------------------------------------------------
    # Resolve video inputs
    # ------------------------------------------------------------------
    video_paths = _resolve_video_paths(input_path, output_dir)
    typer.echo(f"Propose: {len(video_paths)} video(s) to process.")

    all_pose_obs: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    clip_durations: dict[str, float] = {}

    skipped_invalid_track_rows = 0
    total_matching_track_rows = 0
    videos_processed = 0

    for video_path in video_paths:
        clip_id = f"clip_{video_path.stem}"
        typer.echo(f"  Processing {video_path.name}...")

        probe = probe_video(video_path)
        if not probe.decode_ok:
            typer.echo(
                f"    Decode failed: {probe.probe_error}",
                err=True,
            )
            continue

        videos_processed += 1
        duration_s = float(probe.duration_s or 0.0)
        clip_durations[clip_id] = duration_s

        # --------------------------------------------------------------
        # Resolve active spans for this clip.
        # None means no active-span input was provided and the tracker
        # processes the full clip. An empty list means an active-span file
        # was supplied but this clip has no active span, so no frames run.
        # --------------------------------------------------------------
        clip_active_spans: list[ActiveSpan] | None
        if active_spans_path is None:
            clip_active_spans = None
        else:
            matching_spans = [
                record for record in active_span_records if record.get("clip_id") == clip_id
            ]
            clip_active_spans = [ActiveSpan(**record) for record in matching_spans]

            if not clip_active_spans:
                logger.warning(
                    "No active spans found for %s; pose inference will "
                    "produce an empty result for this clip.",
                    clip_id,
                )
            else:
                typer.echo(f"    {len(clip_active_spans)} active span(s)")

        # Run pose inference only over the resolved active spans.
        from pickup_putdown.perception.pose_tracker import PoseTracker

        pose_tracker = PoseTracker(
            video_path=video_path,
            pose_cfg=cfg.pose,
            active_spans=clip_active_spans,
        )
        pose_results = pose_tracker.run()
        typer.echo(f"    {len(pose_results)} pose observations")

        # --------------------------------------------------------------
        # Associate pose detections with stable Task 3 actor tracks.
        # Task 3 may contain unmatched detections whose tracker_track_id
        # round-trips through pandas as NaN. They cannot participate in
        # actor association and are skipped explicitly.
        # --------------------------------------------------------------
        if person_observations:
            matching_person_rows = [
                observation
                for observation in person_observations
                if observation.get("clip_id") == clip_id
            ]
            total_matching_track_rows += len(matching_person_rows)

            valid_person_tracks: list[PersonObservation] = []
            skipped_for_clip = 0

            for observation in matching_person_rows:
                tracker_track_id = observation.get("tracker_track_id")
                try:
                    valid_track_id = tracker_track_id is not None and math.isfinite(
                        float(tracker_track_id)
                    )
                except (TypeError, ValueError):
                    valid_track_id = False

                if not valid_track_id:
                    skipped_for_clip += 1
                    continue

                valid_person_tracks.append(PersonObservation(**observation))

            skipped_invalid_track_rows += skipped_for_clip
            if skipped_for_clip:
                logger.warning(
                    "Skipped %d/%d person observations without a finite tracker ID for clip %s",
                    skipped_for_clip,
                    len(matching_person_rows),
                    clip_id,
                )

            associated = associate_poses_with_actors(
                pose_results,
                valid_person_tracks,
                cfg.actor_association,
            )
        else:
            associated = pose_results

        # Persist the actor-associated records, not the temporary YOLO IDs.
        all_pose_obs.extend(observation.model_dump() for observation in associated)

        raw_interactions = detect_raw_interactions(
            associated,
            camera_config,
            cfg.proposals,
            cfg.region_measurements,
        )
        typer.echo(f"    {len(raw_interactions)} raw interactions")

        candidates = generate_candidates(
            raw_interactions,
            clip_durations,
            cfg.proposals,
        )
        all_candidates.extend(candidate.model_dump() for candidate in candidates)
        typer.echo(f"    {len(candidates)} candidates")

        # --------------------------------------------------------------
        # Optional candidate previews
        # --------------------------------------------------------------
        if render_previews and candidates:
            from pickup_putdown.perception.candidate_previews import (
                CandidateOverlayConfig,
                render_candidate_preview,
            )

            preview_dir = output_base / "candidate_previews"
            preview_dir.mkdir(parents=True, exist_ok=True)

            for candidate in candidates:
                region_id = candidate.region_id or ""
                original_polygon = original_regions.get(region_id)
                expanded_polygon = expanded_regions.get(region_id)

                if original_polygon is None or expanded_polygon is None:
                    logger.warning(
                        "Skipping preview for candidate %s because "
                        "region %r is not present in camera %s",
                        candidate.candidate_id,
                        region_id,
                        selected_camera_id,
                    )
                    continue

                candidate_pose_observations = [
                    observation
                    for observation in associated
                    if (
                        observation.clip_id == candidate.clip_id
                        and observation.actor_id == candidate.actor_id
                        and observation.hand_side == candidate.hand_side
                        and candidate.window_start_s
                        <= observation.timestamp_s
                        <= candidate.window_end_s
                    )
                ]

                output_preview = preview_dir / f"{candidate.candidate_id}.mp4"

                try:
                    render_candidate_preview(
                        video_path,
                        candidate,
                        candidate_pose_observations,
                        original_polygon,
                        expanded_polygon,
                        output_preview,
                        config=CandidateOverlayConfig(
                            draw_actor_box=(cfg.preview.draw_actor_box),
                            draw_wrist_positions=(cfg.preview.draw_wrist_positions),
                            draw_region_polygons=(cfg.preview.draw_region_polygons),
                            draw_region_labels=(cfg.preview.draw_region_labels),
                            draw_candidate_intervals=(cfg.preview.draw_candidate_intervals),
                            text_scale=cfg.preview.text_scale,
                            line_thickness=(cfg.preview.line_thickness),
                            max_output_width=(cfg.preview.max_output_width),
                            max_output_height=(cfg.preview.max_output_height),
                            preview_fps=cfg.preview.preview_fps,
                        ),
                        clip_duration_s=duration_s,
                    )
                except Exception as exc:
                    logger.warning(
                        "Preview failed for candidate %s: %s",
                        candidate.candidate_id,
                        exc,
                    )

    # Deterministic cross-clip output ordering.
    all_pose_obs.sort(
        key=lambda observation: (
            observation["clip_id"],
            observation["timestamp_s"],
            observation.get("actor_id") or "",
            observation.get("hand_side") or "",
        )
    )
    all_candidates.sort(
        key=lambda candidate: (
            candidate["clip_id"],
            candidate.get("actor_id") or "",
            candidate.get("hand_side") or "",
            candidate.get("region_id") or "",
            candidate["raw_start_s"],
            candidate["raw_end_s"],
        )
    )

    # ------------------------------------------------------------------
    # Write tracks_pose.parquet
    # ------------------------------------------------------------------
    if all_pose_obs:
        pose_table = pa.Table.from_pylist(all_pose_obs)
    else:
        pose_schema = pa.schema(
            [
                ("clip_id", pa.string()),
                ("timestamp_s", pa.float64()),
                ("source_frame_index", pa.int64()),
                ("sample_index", pa.int64()),
                ("actor_id", pa.string()),
                ("hand_side", pa.string()),
                ("wrist_x", pa.float64()),
                ("wrist_y", pa.float64()),
                ("wrist_confidence", pa.float64()),
                ("person_bbox_x1", pa.float64()),
                ("person_bbox_y1", pa.float64()),
                ("person_bbox_x2", pa.float64()),
                ("person_bbox_y2", pa.float64()),
                (
                    "pose_association_confidence",
                    pa.float64(),
                ),
                ("is_valid", pa.bool_()),
            ]
        )
        pose_table = pa.Table.from_pydict(
            {field.name: [] for field in pose_schema},
            schema=pose_schema,
        )

    tracks_pose_path = output_base / "tracks_pose.parquet"
    pq.write_table(pose_table, tracks_pose_path)

    # ------------------------------------------------------------------
    # Write candidates.parquet
    # ------------------------------------------------------------------
    if all_candidates:
        candidate_table = pa.Table.from_pylist(all_candidates)
    else:
        candidate_schema = pa.schema(
            [
                ("candidate_id", pa.string()),
                ("clip_id", pa.string()),
                ("actor_id", pa.string()),
                ("hand_side", pa.string()),
                ("region_id", pa.string()),
                ("raw_start_s", pa.float64()),
                ("raw_end_s", pa.float64()),
                ("window_start_s", pa.float64()),
                ("window_end_s", pa.float64()),
                ("n_raw_interactions", pa.int64()),
                ("min_region_distance", pa.float64()),
                ("max_wrist_confidence", pa.float64()),
                (
                    "total_dwell_duration_s",
                    pa.float64(),
                ),
                ("config_fingerprint", pa.string()),
                ("proposal_reason", pa.string()),
                ("proposal_score", pa.float64()),
                ("review_status", pa.string()),
            ]
        )
        candidate_table = pa.Table.from_pydict(
            {field.name: [] for field in candidate_schema},
            schema=candidate_schema,
        )

    candidates_path = output_base / "candidates.parquet"
    pq.write_table(candidate_table, candidates_path)

    # ------------------------------------------------------------------
    # Resolved configuration and run metadata
    # ------------------------------------------------------------------
    resolved = {
        "pose": cfg.pose.model_dump(),
        "actor_association": (cfg.actor_association.model_dump()),
        "region_measurements": (cfg.region_measurements.model_dump()),
        "proposals": cfg.proposals.model_dump(),
        "preview": cfg.preview.model_dump(),
        "runtime": {
            "camera_id": selected_camera_id,
            "shelves_config": str(shelf_cfg_path),
            "person_tracks": (str(person_tracks_path) if person_tracks_path is not None else None),
            "active_spans": (str(active_spans_path) if active_spans_path is not None else None),
            "render_previews": render_previews,
        },
    }

    resolved_config_path = output_base / "resolved_proposals_config.yaml"
    with resolved_config_path.open("w") as file_handle:
        yaml.safe_dump(
            resolved,
            file_handle,
            default_flow_style=False,
            sort_keys=False,
        )

    try:
        git_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        git_commit = "unknown"

    try:
        pose_model_hash = hashlib.sha256(Path(cfg.pose.model_path).read_bytes()).hexdigest()
    except Exception:
        pose_model_hash = "unknown"

    metadata = {
        "git_commit": git_commit,
        "proposals_config": str(cfg_path),
        "shelves_config": str(shelf_cfg_path),
        "camera_id": selected_camera_id,
        "person_tracks": (str(person_tracks_path) if person_tracks_path is not None else None),
        "active_spans": (str(active_spans_path) if active_spans_path is not None else None),
        "pose_model_path": cfg.pose.model_path,
        "pose_model_sha256": pose_model_hash,
        "n_videos_requested": len(video_paths),
        "n_videos_processed": videos_processed,
        "n_person_track_rows": len(person_observations),
        "n_matching_person_track_rows": (total_matching_track_rows),
        "n_invalid_person_track_rows_skipped": (skipped_invalid_track_rows),
        "n_pose_observations": len(all_pose_obs),
        "n_candidates": len(all_candidates),
        "target_fps": cfg.pose.target_fps,
        "render_previews": render_previews,
    }

    metadata_path = output_base / "propose_run_metadata.json"
    with metadata_path.open("w") as file_handle:
        json.dump(metadata, file_handle, indent=2)

    typer.echo("")
    typer.echo("=== Propose Summary ===")
    typer.echo(f"  Videos processed:    {videos_processed}/{len(video_paths)}")
    typer.echo(f"  Shelf camera:        {selected_camera_id}")
    typer.echo(f"  Pose observations:   {len(all_pose_obs)}")
    typer.echo(f"  Candidates:          {len(all_candidates)}")
    typer.echo(f"  Invalid track rows:  {skipped_invalid_track_rows}")
    typer.echo(f"  Tracks pose parquet: {tracks_pose_path}")
    typer.echo(f"  Candidates parquet:  {candidates_path}")
    typer.echo(f"  Resolved config:     {resolved_config_path}")
    typer.echo(f"  Run metadata:        {metadata_path}")
    if render_previews:
        typer.echo(f"  Previews dir:        {output_base / 'candidate_previews'}")


# ---------------------------------------------------------------------------
# Annotation commands
# ---------------------------------------------------------------------------


@app.command()
def annotation_build_tasks(
    clips_path: str = typer.Argument(
        ...,
        help="Path to clips JSON file.",
    ),
    candidates_path: str | None = typer.Option(
        None,
        "--candidates",
        "-c",
        help="Path to candidates JSON file (optional).",
    ),
    output_path: str = typer.Option(
        "annotation/tasks.json",
        "--output",
        "-o",
        help="Output path for Label Studio task JSON.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Build Label Studio tasks from clip metadata and candidate predictions.

    Candidates are placed in the prediction (pre-annotation) structure,
    never in the completed annotation structure.
    """
    _setup_logging(verbose)

    import json
    from pathlib import Path

    from pickup_putdown.annotation.import_export import (
        build_label_studio_tasks,
    )

    clips = json.loads(Path(clips_path).read_text())
    candidates = None
    if candidates_path:
        candidates = json.loads(Path(candidates_path).read_text())

    tasks = build_label_studio_tasks(clips, candidates)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps([t.model_dump() for t in tasks], indent=2, default=str)
    )
    typer.echo(f"Wrote {len(tasks)} task(s) to {output_path}")


@app.command()
def annotation_export(
    input_path: str = typer.Argument(
        ...,
        help="Path to Label Studio export JSON.",
    ),
    events_output: str = typer.Option(
        "events.csv",
        "--events",
        "-e",
        help="Output path for canonical events.csv.",
    ),
    ignore_output: str = typer.Option(
        "ignore_intervals.parquet",
        "--ignore",
        "-i",
        help="Output path for ignore_intervals.parquet.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Export Label Studio annotations to canonical repository formats.

    Only accepted visible pickup/putdown events with
    complete_active_span_reviewed=true are exported to events.csv.
    Ignore intervals are exported to ignore_intervals.parquet.
    """
    _setup_logging(verbose)

    import json
    from pathlib import Path

    from pickup_putdown.annotation.import_export import (
        export_events_csv,
        export_ignore_intervals_parquet,
    )

    export_data = json.loads(Path(input_path).read_text())

    events_result = export_events_csv(export_data, events_output)
    typer.echo(f"Events: {len(events_result.canonical_events)} rows ({events_output})")
    if not events_result.is_valid:
        for err in events_result.validation.errors:
            typer.echo(f"  WARN: {err.message}", err=True)

    ignore_result = export_ignore_intervals_parquet(export_data, ignore_output)
    typer.echo(f"Ignore intervals: {len(ignore_result.ignore_intervals)} rows ({ignore_output})")


@app.command()
def annotation_validate(
    input_path: str = typer.Argument(
        ...,
        help="Path to Label Studio export JSON.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Validate a Label Studio export JSON before conversion."""
    _setup_logging(verbose)

    import json
    from pathlib import Path

    from pickup_putdown.annotation.import_export import (
        validate_export,
    )

    export_data = json.loads(Path(input_path).read_text())
    errors = validate_export(export_data)

    if errors.is_valid:
        typer.echo("Validation passed.")
    else:
        typer.echo(f"Validation failed with {len(errors.errors)} error(s):", err=True)
        for err in errors.errors:
            typer.echo(
                f"  [{err.task_id}/{err.region_id}/{err.field_name}] {err.message}",
                err=True,
            )
        raise SystemExit(1)


@app.command()
def annotation_roundtrip(
    export_path: str = typer.Argument(
        ...,
        help="Path to Label Studio export JSON.",
    ),
    fps: float = typer.Option(
        30.0,
        "--fps",
        help="Frame rate for round-trip check.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Verify timestamp fidelity of a round-trip export."""
    _setup_logging(verbose)

    import json
    from pathlib import Path

    from pickup_putdown.annotation.import_export import (
        export_events_csv,
        round_trip_check,
    )

    export_data = json.loads(Path(export_path).read_text())
    result = export_events_csv(export_data)

    if not result.is_valid:
        typer.echo("Export validation failed. Cannot perform round-trip check.", err=True)
        raise SystemExit(1)

    # Use the exported events as both original and re-exported
    original = result.canonical_events
    passed = round_trip_check(original, export_data, fps=fps)

    if passed:
        typer.echo(f"Round-trip check passed ({len(original)} events, {fps} fps).")
    else:
        typer.echo("Round-trip check FAILED: timestamps differ beyond tolerance.", err=True)
        raise SystemExit(1)


if __name__ == "__main__":
    app()
