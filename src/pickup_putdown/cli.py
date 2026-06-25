"""CLI entry point for the pickup-putdown package."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import typer

from pickup_putdown.cli_infer import infer_app

app = typer.Typer(
    name="pickup-putdown",
    help="Pickup and putdown temporal action detection in store video.",
    add_completion=False,
)
app.add_typer(infer_app)

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

        # Run person tracker (pipelined for better GPU utilization)
        from pickup_putdown.perception.pipelined_tracker import PipelinedPersonTracker

        tracker = PipelinedPersonTracker(
            video_path=vp,
            triage_cfg=triage_cfg,
            use_pipeline=triage_cfg.pipeline_enabled,
        )
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
    clips_path: str | None = typer.Option(
        None,
        "--clips",
        help="Path to clips JSON file (for legacy clip-based tasks).",
    ),
    candidates_path: str | None = typer.Option(
        None,
        "--candidates",
        "-c",
        help="Path to candidates JSON file (optional, for legacy mode).",
    ),
    candidate_metadata_dir: str | None = typer.Option(
        None,
        "--candidate-metadata-dir",
        help=(
            "Directory containing candidate metadata JSON files from Task 6.1. "
            "When provided, builds candidate-backed tasks instead of clip-based tasks."
        ),
    ),
    output_path: str = typer.Option(
        "annotation/tasks.json",
        "--output",
        "-o",
        help="Output path for Label Studio task JSON.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help=(
            "Select at most this many candidates for a pilot. "
            "Use with --seed for deterministic selection."
        ),
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for deterministic pilot selection.",
    ),
    video_url_mode: str = typer.Option(
        "s3_key",
        "--video-url-mode",
        help=("How to format video references: local, s3_key, s3_storage, or presigned."),
    ),
    s3_bucket: str | None = typer.Option(
        None,
        "--s3-bucket",
        help="S3 bucket name (required for s3_storage mode).",
    ),
    s3_prefix: str | None = typer.Option(
        "anon/candidates/videos",
        "--s3-prefix",
        help="S3 prefix for candidate videos.",
    ),
    local_video_dir: str | None = typer.Option(
        None,
        "--local-video-dir",
        help="Local directory for candidate videos (required for local mode).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Build Label Studio tasks from clip metadata or candidate metadata.

    Legacy mode (--clips): Build tasks from clip metadata with optional
    candidate predictions.

    Candidate mode (--candidate-metadata-dir): Build tasks from Task 6.1
    candidate metadata. Each candidate becomes one task with source offset
    information for timestamp conversion during export. No default event
    label is assigned.

    Use --limit and --seed for deterministic pilot selection from existing
    candidates.
    """
    _setup_logging(verbose)

    import json
    from pathlib import Path

    from pickup_putdown.annotation.schemas import VideoUrlMode

    if candidate_metadata_dir:
        from pickup_putdown.annotation.import_export import (
            _load_candidate_metadata_from_dir,
            build_candidate_tasks,
            select_candidate_pilot,
        )

        # Load metadata
        metadata_path = Path(candidate_metadata_dir)
        load_stats = None
        if metadata_path.is_file():
            candidate_metadata = json.loads(metadata_path.read_text())
            if isinstance(candidate_metadata, dict) and "candidates" in candidate_metadata:
                source_meta = candidate_metadata
                nested = source_meta.get("candidates", [])
                if not isinstance(nested, list):
                    typer.echo(
                        f"Error: 'candidates' in {metadata_path} must be a list.",
                        err=True,
                    )
                    raise SystemExit(1)
                source_video_id = source_meta.get("source_video_id")
                source_bucket = source_meta.get("source_bucket")
                source_key = source_meta.get("source_key")
                candidate_metadata = []
                for cand in nested:
                    if not isinstance(cand, dict):
                        continue
                    enriched = dict(cand)
                    if source_video_id:
                        enriched.setdefault("clip_id", str(source_video_id))
                    if source_bucket:
                        enriched.setdefault("source_bucket", source_bucket)
                    if source_key:
                        enriched.setdefault("source_key", source_key)
                    candidate_metadata.append(enriched)
            elif isinstance(candidate_metadata, list):
                pass
            else:
                candidate_metadata = [candidate_metadata]
        elif metadata_path.is_dir():
            candidate_metadata, load_stats = _load_candidate_metadata_from_dir(metadata_path)
            # Report load errors immediately
            if load_stats.errors:
                typer.echo("Metadata load errors:", err=True)
                for err_msg in load_stats.errors:
                    typer.echo(f"  ERROR: {err_msg}", err=True)
                raise SystemExit(1)
        else:
            typer.echo(f"Metadata path not found: {metadata_path}", err=True)
            raise SystemExit(1)

        # Report load stats
        if load_stats:
            typer.echo(f"Scanned {load_stats.source_files_scanned} source metadata file(s).")
            if load_stats.zero_candidate_sources_skipped:
                typer.echo(
                    f"Skipped {load_stats.zero_candidate_sources_skipped} "
                    f"source(s) with zero candidates."
                )
            typer.echo(f"Loaded {load_stats.candidates_loaded} candidate(s).")

        # Pilot selection
        if limit is not None:
            if limit <= 0:
                typer.echo(f"Error: --limit must be positive, got {limit}", err=True)
                raise SystemExit(1)
            try:
                candidate_metadata = select_candidate_pilot(
                    candidate_metadata,
                    limit=limit,
                    seed=seed,
                )
                typer.echo(
                    f"Selected {len(candidate_metadata)} candidate(s) (limit={limit}, seed={seed})"
                )
            except ValueError as exc:
                typer.echo(f"Error selecting pilot: {exc}", err=True)
                raise SystemExit(1) from exc

        # Validate video URL mode
        try:
            url_mode = VideoUrlMode(video_url_mode)
        except ValueError as e:
            typer.echo(
                f"Error: invalid --video-url-mode {video_url_mode!r}. "
                f"Choose from: {', '.join(v.value for v in VideoUrlMode)}",
                err=True,
            )
            raise SystemExit(1) from e

        tasks, errors = build_candidate_tasks(
            candidate_metadata,
            video_url_mode=url_mode,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            local_video_dir=local_video_dir,
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            json.dumps([t.model_dump() for t in tasks], indent=2, default=str)
        )
        typer.echo(f"Generated {len(tasks)} task(s) -> {output_path}")
        if errors:
            typer.echo(f"Rejected {len(errors)} candidate(s) due to validation errors:", err=True)
            for err in errors:
                typer.echo(
                    f"  [{err.candidate_id}/{err.field_name}] {err.message}",
                    err=True,
                )
            raise SystemExit(1)
    elif clips_path:
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
    else:
        typer.echo(
            "Error: provide either --clips or --candidate-metadata-dir.",
            err=True,
        )
        raise SystemExit(1)


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
    provenance_output: str | None = typer.Option(
        None,
        "--provenance",
        help=(
            "Output path for event_provenance.parquet containing candidate traceability metadata."
        ),
    ),
    candidate_mode: bool = typer.Option(
        False,
        "--candidate-mode",
        help=(
            "Enable candidate-backed export mode. Converts candidate-relative "
            "timestamps to source-video timestamps using source_start_s offset."
        ),
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

    In --candidate-mode, timestamps are converted from candidate-relative
    to source-video timestamps using the source_start_s offset stored in
    each task's data. Legacy tasks without source offset pass through
    unchanged.

    The --provenance option produces a separate event_provenance.parquet
    with candidate traceability metadata (candidate_id, actor_id, etc.)
    that is not included in the official canonical events.csv.
    """
    _setup_logging(verbose)

    import json
    from pathlib import Path

    export_data = json.loads(Path(input_path).read_text())

    if candidate_mode:
        from pickup_putdown.annotation.import_export import (
            export_candidate_annotations,
        )

        result = export_candidate_annotations(
            export_data,
            events_output=events_output,
            ignore_output=ignore_output,
            provenance_output=provenance_output,
        )
        typer.echo(f"Events: {len(result.canonical_events)} rows ({events_output})")
        typer.echo(f"Ignore intervals: {len(result.ignore_intervals)} rows ({ignore_output})")
        if provenance_output:
            typer.echo(f"Provenance: written to {provenance_output}")
        if not result.is_valid:
            for err in result.validation.errors:
                typer.echo(f"  WARN: {err.message}", err=True)
    else:
        from pickup_putdown.annotation.import_export import (
            export_events_csv,
            export_ignore_intervals_parquet,
        )

        events_result = export_events_csv(export_data, events_output)
        typer.echo(f"Events: {len(events_result.canonical_events)} rows ({events_output})")
        if not events_result.is_valid:
            for err in events_result.validation.errors:
                typer.echo(f"  WARN: {err.message}", err=True)

        ignore_result = export_ignore_intervals_parquet(export_data, ignore_output)
        typer.echo(
            f"Ignore intervals: {len(ignore_result.ignore_intervals)} rows ({ignore_output})"
        )


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


@app.command("annotation-check-media")
def annotation_check_media(
    tasks_path: str = typer.Argument(
        ...,
        help="Path to Label Studio task JSON file.",
    ),
    video_url_mode: str = typer.Option(
        "s3_key",
        "--video-url-mode",
        help=("Expected video URL mode: local, s3_key, s3_storage, or presigned."),
    ),
    local_video_dir: str | None = typer.Option(
        None,
        "--local-video-dir",
        help="Local directory for candidate videos (required for local mode).",
    ),
    s3_bucket: str | None = typer.Option(
        None,
        "--s3-bucket",
        help="S3 bucket name (required for s3_storage mode).",
    ),
    s3_endpoint_url: str | None = typer.Option(
        None,
        "--s3-endpoint-url",
        help="S3 endpoint URL for object existence checks.",
    ),
    s3_region: str | None = typer.Option(
        None,
        "--s3-region",
        help="S3 region for object existence checks.",
    ),
    s3_anonymous: bool = typer.Option(
        False,
        "--s3-anonymous",
        help="Use anonymous S3 access for checks.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Verify media references in a Label Studio task file.

    Checks that each task has a valid video reference for the configured
    playback mode. For local mode, verifies files exist. For s3_storage
    mode, checks S3 object existence (requires credentials). For s3_key
    and presigned modes, validates URL format.
    """
    _setup_logging(verbose)

    from pickup_putdown.annotation.import_export import check_media_references
    from pickup_putdown.annotation.schemas import VideoUrlMode

    try:
        url_mode = VideoUrlMode(video_url_mode)
    except ValueError as e:
        typer.echo(
            f"Error: invalid --video-url-mode {video_url_mode!r}. "
            f"Choose from: {', '.join(v.value for v in VideoUrlMode)}",
            err=True,
        )
        raise SystemExit(1) from e

    report = check_media_references(
        tasks_path,
        video_url_mode=url_mode,
        local_video_dir=local_video_dir,
        s3_bucket=s3_bucket,
        s3_endpoint_url=s3_endpoint_url,
        s3_region=s3_region,
        s3_anonymous=s3_anonymous,
    )

    typer.echo(f"Total tasks: {report.total}")
    typer.echo(f"Passed: {report.passed}")
    typer.echo(f"Failed: {report.failed}")

    if report.failed > 0:
        typer.echo("\nFailed checks:", err=True)
        for r in report.results:
            if not r.ok:
                typer.echo(f"  [{r.task_id}/{r.candidate_id}] {r.message}", err=True)
        raise SystemExit(1)


@app.command("candidates-download")
def candidates_download(
    storage_config: str = typer.Option(
        "configs/storage.s3.yaml",
        "--storage-config",
        "-s",
        help="Path to S3 storage configuration YAML file.",
    ),
    target_count: int = typer.Option(
        10,
        "--target-count",
        "-t",
        help="Number of not-yet-downloaded source videos to download.",
    ),
    transfer_workers: int = typer.Option(
        4,
        "--transfer-workers",
        help="Maximum concurrent S3 download operations.",
    ),
    local_source_dir: str = typer.Option(
        ".local/source_videos",
        "--local-source-dir",
        help="Local directory for cached source videos.",
    ),
    local_output_dir: str = typer.Option(
        ".local/candidate_staging",
        "--local-output-dir",
        help="Local directory for candidate staging and run reports.",
    ),
    minimum_free_disk_gb: float = typer.Option(
        0.0,
        "--minimum-free-disk-gb",
        help="Minimum free disk space in GB before downloading.",
    ),
    refresh_changed: bool = typer.Option(
        False,
        "--refresh-changed",
        help="Redownload sources that changed in S3.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Download source videos from S3 to local cache in batches.

    Tracks download state in local_processing.csv. Repeated invocations
    select different not-yet-downloaded source videos deterministically.
    """
    _setup_logging(verbose)

    from pathlib import Path

    from pickup_putdown.config import load_config
    from pickup_putdown.remote.download_coordinator import (
        DownloadConfig,
        run_source_download,
    )
    from pickup_putdown.remote.local_ledger import LocalProcessingLedger
    from pickup_putdown.remote.s3_storage import S3Storage

    cfg = load_config(Path(storage_config))
    storage_cfg = cfg.storage

    if not storage_cfg.bucket_uri:
        typer.echo("Error: storage.bucket_uri is required.", err=True)
        raise SystemExit(1)

    typer.echo(f"Storage: {storage_cfg.bucket_uri}")
    typer.echo(f"Target count: {target_count}")
    typer.echo(f"Source dir: {local_source_dir}")
    typer.echo(f"Output dir: {local_output_dir}")

    storage = S3Storage(
        bucket_uri=storage_cfg.bucket_uri,
        endpoint_url=storage_cfg.endpoint_url,
        region=storage_cfg.region,
        anonymous=storage_cfg.anonymous,
    )

    ledger_path = Path(local_output_dir) / "local_processing.csv"
    local_ledger = LocalProcessingLedger(ledger_path)

    download_cfg = DownloadConfig(
        target_count=target_count,
        transfer_workers=transfer_workers,
        local_source_dir=local_source_dir,
        local_output_dir=local_output_dir,
        minimum_free_disk_gb=minimum_free_disk_gb,
        refresh_changed=refresh_changed,
    )

    try:
        report = run_source_download(
            storage=storage,
            local_ledger=local_ledger,
            config=download_cfg,
        )
    except KeyboardInterrupt as kb_exc:
        typer.echo("\nInterrupted by user.", err=True)
        raise SystemExit(130) from kb_exc
    except Exception as exc:
        typer.echo(f"Fatal error: {exc}", err=True)
        raise SystemExit(1) from exc

    typer.echo("")
    typer.echo("=== Download Summary ===")
    typer.echo(f"  Run ID:        {report.run_id}")
    typer.echo(f"  Requested:     {report.requested_count}")
    typer.echo(f"  Selected:      {report.selected_count}")
    typer.echo(f"  Downloaded:    {report.downloaded_count}")
    typer.echo(f"  Failed:        {report.failed_count}")
    typer.echo(f"  Skipped:       {report.skipped_count}")
    typer.echo(f"  Started:       {report.started_at}")
    typer.echo(f"  Completed:     {report.completed_at}")
    typer.echo(f"  Ledger:        {ledger_path}")
    typer.echo(f"  Report:        {Path(local_output_dir) / 'runs' / f'{report.run_id}.json'}")

    if report.errors:
        typer.echo("")
        typer.echo("Errors:")
        for err in report.errors:
            typer.echo(f"  {err}")

    if report.failed_count > 0:
        raise SystemExit(1)


@app.command("candidates-remote")
def candidates_remote(
    storage_config: str = typer.Option(
        "configs/storage.s3.yaml",
        "--storage-config",
        "-s",
        help="Path to S3 storage configuration YAML file.",
    ),
    pipeline_config: str = typer.Option(
        "configs/candidates.yaml",
        "--pipeline-config",
        "-p",
        help="Path to candidate pipeline configuration YAML file.",
    ),
    target_count: int = typer.Option(
        5,
        "--target-count",
        "-t",
        help="Number of unprocessed videos to select for candidate generation.",
    ),
    workers: int = typer.Option(
        4,
        "--workers",
        "-w",
        help="Maximum concurrent source-video processing jobs.",
    ),
    transfer_workers: int = typer.Option(
        4,
        "--transfer-workers",
        help="Maximum concurrent S3 download/upload operations.",
    ),
    gpu_workers: int = typer.Option(
        1,
        "--gpu-workers",
        help="Maximum concurrent GPU inference jobs.",
    ),
    encode_workers: int = typer.Option(
        4,
        "--encode-workers",
        help="Maximum concurrent H.264 encoding jobs.",
    ),
    work_dir: str = typer.Option(
        ".local/remote_candidates",
        "--work-dir",
        help="Local working directory for intermediate files.",
    ),
    keep_local_files: bool = typer.Option(
        False,
        "--keep-local-files",
        help="Keep local intermediate files after successful upload.",
    ),
    fail_fast: bool = typer.Option(
        False,
        "--fail-fast",
        help="Stop all processing on first source video failure.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Re-process videos already marked as processed.",
    ),
    defer_upload: bool = typer.Option(
        False,
        "--defer-upload",
        help="Generate candidates locally without uploading to S3.",
    ),
    local_source_dir: str = typer.Option(
        ".local/source_videos",
        "--local-source-dir",
        help="Local directory for cached source videos (deferred mode).",
    ),
    local_output_dir: str = typer.Option(
        ".local/candidate_staging",
        "--local-output-dir",
        help="Local directory for candidate staging and run reports.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Synchronize ledger and report selections without processing.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Generate annotation candidates from remote S3 source videos.

    Downloads unprocessed source videos from S3, runs the Tasks 3-5
    candidate-generation pipeline, encodes candidates as H.264 MP4 files,
    and uploads them back to S3 for Label Studio annotation.

    With --defer-upload, processes locally cached sources and stages
    candidates without uploading. Use candidates-download first to
    populate the local source cache.
    """
    _setup_logging(verbose)

    from pathlib import Path

    from pickup_putdown.config import load_config
    from pickup_putdown.remote.coordinator import (
        CoordinationConfig,
        run_candidate_generation,
    )
    from pickup_putdown.remote.ledger import ProcessingLedger
    from pickup_putdown.remote.s3_storage import S3Storage
    from pickup_putdown.remote.worker import WorkerConfig

    # Load storage config
    cfg = load_config(Path(storage_config))
    storage_cfg = cfg.storage

    if not storage_cfg.bucket_uri:
        typer.echo("Error: storage.bucket_uri is required.", err=True)
        raise SystemExit(1)

    typer.echo(f"Storage: {storage_cfg.bucket_uri}")
    typer.echo(f"Target count: {target_count}")
    typer.echo(
        f"Workers: {workers} (transfer={transfer_workers}, gpu={gpu_workers}, encode={encode_workers})"
    )
    if defer_upload:
        typer.echo("Mode: deferred (local generation only)")

    # Initialize S3 storage
    storage = S3Storage(
        bucket_uri=storage_cfg.bucket_uri,
        endpoint_url=storage_cfg.endpoint_url,
        region=storage_cfg.region,
        anonymous=storage_cfg.anonymous,
    )

    # Initialize ledger — local for defer-upload, S3 for normal mode
    if defer_upload:
        from pickup_putdown.remote.local_ledger import LocalProcessingLedger

        ledger_path = Path(local_output_dir) / "local_processing.csv"
        local_ledger = LocalProcessingLedger(ledger_path)
        local_ledger.load()
        ledger = local_ledger
    else:
        ledger = ProcessingLedger(storage)

    # Load pipeline config for encoding settings
    pipeline_cfg = load_config(Path(pipeline_config))

    # Build coordination config
    coord_cfg = CoordinationConfig(
        target_count=target_count,
        workers=workers,
        transfer_workers=transfer_workers,
        gpu_workers=gpu_workers,
        encode_workers=encode_workers,
        work_dir=work_dir,
        keep_local_files=keep_local_files,
        fail_fast=fail_fast,
        overwrite=overwrite,
        dry_run=dry_run,
        defer_upload=defer_upload,
        local_output_dir=local_output_dir,
        local_source_dir=local_source_dir,
    )

    # Build worker config
    worker_cfg = WorkerConfig(
        storage_config=Path(storage_config),
        pipeline_config=Path(pipeline_config),
        work_dir=Path(work_dir),
        keep_local_files=keep_local_files,
        triage_config=cfg.triage.model_dump()
        .get("model_path", "models/person_detector.pt")
        .rsplit("/", 1)[0]
        + "/triage.yaml"
        if False
        else "configs/triage.yaml",
        proposals_config="configs/proposals.yaml",
        shelves_config="configs/shelves.yaml",
        camera_id="store_camera_01",
        defer_upload=defer_upload,
        local_source_dir=local_source_dir,
        local_output_dir=local_output_dir,
    )

    # Override from pipeline config if present
    if hasattr(pipeline_cfg, "triage"):
        worker_cfg.triage_config = "configs/triage.yaml"
    if hasattr(pipeline_cfg, "proposals"):
        worker_cfg.proposals_config = "configs/proposals.yaml"

    try:
        report = run_candidate_generation(
            storage=storage,
            ledger=ledger,
            config=coord_cfg,
            worker_cfg=worker_cfg,
        )
    except KeyboardInterrupt as kb_exc:
        typer.echo("\nInterrupted by user.", err=True)
        raise SystemExit(130) from kb_exc
    except Exception as exc:
        typer.echo(f"Fatal error: {exc}", err=True)
        raise SystemExit(1) from exc

    # Print summary
    typer.echo("")
    typer.echo("=== Candidate Generation Summary ===")
    typer.echo(f"  Run ID:              {report.run_id}")
    typer.echo(f"  Requested:           {report.requested_count}")
    typer.echo(f"  Selected:            {report.selected_count}")
    typer.echo(f"  Completed:           {report.completed_count}")
    typer.echo(f"  Failed:              {report.failed_count}")
    typer.echo(f"  Skipped:             {report.skipped_count}")
    typer.echo(f"  Total candidates:    {report.total_candidates}")
    typer.echo(f"  Start:               {report.start_time}")
    typer.echo(f"  End:                 {report.end_time}")

    if report.failed_sources:
        typer.echo("")
        typer.echo("Failed sources:")
        for fs in report.failed_sources:
            typer.echo(f"  {fs['source_key']}: {fs['error'][:120]}")

    if report.failed_count > 0:
        raise SystemExit(1)


@app.command("candidates-process-local")
def candidates_process_local(
    pipeline_config: str = typer.Option(
        "configs/candidates.yaml",
        "--pipeline-config",
        "-p",
        help="Path to candidate pipeline configuration YAML file.",
    ),
    target_count: int = typer.Option(
        10,
        "--target-count",
        "-t",
        help="Number of downloaded videos to process.",
    ),
    encode_workers: int = typer.Option(
        12,
        "--encode-workers",
        help="Maximum concurrent H.264 encoding jobs (CPU-bound).",
    ),
    gpu_workers: int = typer.Option(
        8,
        "--gpu-workers",
        help="Maximum concurrent GPU inference workers (triage + propose).",
    ),
    work_dir: str = typer.Option(
        ".local/remote_candidates",
        "--work-dir",
        help="Local working directory for intermediate files.",
    ),
    keep_local_files: bool = typer.Option(
        False,
        "--keep-local-files",
        help="Keep local intermediate files after successful processing.",
    ),
    local_source_dir: str = typer.Option(
        ".local/source_videos",
        "--local-source-dir",
        help="Local directory for downloaded source videos.",
    ),
    local_output_dir: str = typer.Option(
        ".local/candidate_staging",
        "--local-output-dir",
        help="Local directory for candidate staging and run reports.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Re-process videos already marked as generated.",
    ),
    skip_file: str = typer.Option(
        ".local/processing_skip.txt",
        "--skip-file",
        help="File with one filename per line to skip during selection.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Process downloaded source videos locally with GPU/CPU two-stage pipeline.

    Runs GPU inference (triage + propose) in parallel and encoding
    (H.264 MP4) in parallel. No S3 interaction — purely local.

    Uses the local ledger to track downloaded/generated state.
    """
    _setup_logging(verbose)

    from pathlib import Path

    from pickup_putdown.config import load_config
    from pickup_putdown.remote.coordinator import run_two_stage_pipeline
    from pickup_putdown.remote.local_ledger import LocalProcessingLedger
    from pickup_putdown.remote.worker import WorkerConfig

    # Load pipeline config
    pipeline_cfg = load_config(Path(pipeline_config))

    # Initialize local ledger
    ledger_path = Path(local_output_dir) / "local_processing.csv"
    local_ledger = LocalProcessingLedger(ledger_path)
    local_ledger.load()

    # Load skip list
    skip_path = Path(skip_file)
    skip_names: set[str] = set()
    if skip_path.exists():
        skip_names = {
            line.split()[0]
            for line in skip_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if skip_names:
            typer.echo(f"Skipping {len(skip_names)} file(s) from {skip_file}")

    # Select downloaded but not generated
    if overwrite:
        ready = sorted(
            [
                e
                for e in local_ledger.entries.values()
                if e.downloaded and e.file_name not in skip_names
            ],
            key=lambda e: e.file_name,
        )[:target_count]
    else:
        ready = local_ledger.select_ready_for_generation(target_count, skip_names)

    if not ready:
        typer.echo("No downloaded videos ready for processing.")
        raise SystemExit(0)

    typer.echo(f"Processing {len(ready)} video(s) locally...")
    typer.echo(f"GPU workers:     {gpu_workers}")
    typer.echo(f"Encode workers:  {encode_workers}")
    typer.echo(f"Source dir:      {local_source_dir}")
    typer.echo(f"Output dir:      {local_output_dir}")

    worker_cfg = WorkerConfig(
        storage_config=Path("configs/storage.s3.yaml"),
        pipeline_config=Path(pipeline_config),
        work_dir=Path(work_dir),
        keep_local_files=keep_local_files,
        triage_config="configs/triage.yaml",
        tracker_config="configs/bytetrack_triage.yaml",
        proposals_config="configs/proposals.yaml",
        shelves_config="configs/shelves.yaml",
        camera_id="store_camera_01",
        defer_upload=True,
        local_source_dir=local_source_dir,
        local_output_dir=local_output_dir,
    )

    if hasattr(pipeline_cfg, "triage"):
        worker_cfg.triage_config = "configs/triage.yaml"
    if hasattr(pipeline_cfg, "proposals"):
        worker_cfg.proposals_config = "configs/proposals.yaml"

    try:
        report = run_two_stage_pipeline(
            entries=ready,
            worker_cfg=worker_cfg,
            local_source_dir=Path(local_source_dir),
            local_output_dir=Path(local_output_dir),
            encode_workers=encode_workers,
            gpu_workers=gpu_workers,
        )
    except KeyboardInterrupt as kb_exc:
        typer.echo("\nInterrupted by user.", err=True)
        raise SystemExit(130) from kb_exc
    except Exception as exc:
        typer.echo(f"Fatal error: {exc}", err=True)
        raise SystemExit(1) from exc

    # Update ledger for completed entries
    completed_keys = {fs["source_key"] for fs in report.failed_sources}
    for entry in ready:
        if entry.file_name not in completed_keys:
            local_ledger.mark_generated(entry.file_name)
    local_ledger.save()

    # Save local run report
    from pickup_putdown.remote.coordinator import _save_local_run_report

    _save_local_run_report(Path(local_output_dir), report)

    # Print summary
    typer.echo("")
    typer.echo("=== Local Processing Summary ===")
    typer.echo(f"  Run ID:              {report.run_id}")
    typer.echo(f"  Selected:            {report.selected_count}")
    typer.echo(f"  Completed:           {report.completed_count}")
    typer.echo(f"  Failed:              {report.failed_count}")
    typer.echo(f"  Total candidates:    {report.total_candidates}")
    typer.echo(f"  Start:               {report.start_time}")
    typer.echo(f"  End:                 {report.end_time}")
    typer.echo(f"  Ledger:              {ledger_path}")
    typer.echo(
        f"  Report:              {Path(local_output_dir) / 'runs' / f'{report.run_id}.json'}"
    )

    if report.failed_sources:
        typer.echo("")
        typer.echo("Failed sources:")
        for fs in report.failed_sources:
            typer.echo(f"  {fs['source_key']}: {fs['error'][:120]}")

    if report.failed_count > 0:
        raise SystemExit(1)


@app.command("candidates-upload")
def candidates_upload(
    storage_config: str = typer.Option(
        "configs/storage.s3.yaml",
        "--storage-config",
        "-s",
        help="Path to S3 storage configuration YAML file.",
    ),
    local_output_dir: str = typer.Option(
        ".local/candidate_staging",
        "--local-output-dir",
        help="Local directory for candidate staging and ledger.",
    ),
    target_count: int = typer.Option(
        0,
        "--target-count",
        "-t",
        help="Maximum sources to upload (0 = all ready).",
    ),
    upload_ledger: str | None = typer.Option(
        None,
        "--upload-ledger",
        help="Separate ledger file for tracking uploaded state. "
        "When set, reads generated state from the main ledger but writes "
        "uploaded=true here instead, avoiding race conditions with a running "
        "processing pipeline. Use ledger-reconcile to merge back.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Upload locally staged candidates to S3.

    Selects entries where generated=true and uploaded=false from the
    local processing ledger, uploads candidate videos and metadata,
    then marks them as uploaded.
    """
    _setup_logging(verbose)

    from pathlib import Path

    from pickup_putdown.config import load_config
    from pickup_putdown.remote.local_ledger import LocalProcessingLedger
    from pickup_putdown.remote.s3_storage import S3Storage

    cfg = load_config(Path(storage_config))
    storage_cfg = cfg.storage

    if not storage_cfg.bucket_uri:
        typer.echo("Error: storage.bucket_uri is required.", err=True)
        raise SystemExit(1)

    storage = S3Storage(
        bucket_uri=storage_cfg.bucket_uri,
        endpoint_url=storage_cfg.endpoint_url,
        region=storage_cfg.region,
        anonymous=storage_cfg.anonymous,
    )

    main_ledger_path = Path(local_output_dir) / "local_processing.csv"
    main_ledger = LocalProcessingLedger(main_ledger_path)
    main_ledger.load()

    upload_ledger_path = None
    upload_ledger_obj = None
    if upload_ledger:
        upload_ledger_path = Path(upload_ledger)
        upload_ledger_obj = LocalProcessingLedger(upload_ledger_path)
        upload_ledger_obj.load()
        typer.echo(
            f"Using separate upload ledger: {upload_ledger_path} (main ledger left unmodified)"
        )

    # When using a separate upload ledger, exclude already-uploaded entries
    # from the main-ledger selection (since main ledger's uploaded flag isn't set).
    already_uploaded: set[str] | None = None
    if upload_ledger_obj:
        already_uploaded = {fn for fn, e in upload_ledger_obj.entries.items() if e.uploaded}
        if already_uploaded:
            typer.echo(f"  Excluding {len(already_uploaded)} already-uploaded entries")

    ready = main_ledger.select_ready_for_upload(target_count, skip_names=already_uploaded)
    if not ready:
        typer.echo("No candidates ready for upload.")
        raise SystemExit(0)

    typer.echo(f"Uploading {len(ready)} source(s)...")

    uploaded = 0
    failed = 0
    for entry in ready:
        source_video_id = entry.file_name.replace("/", "_").replace(".mp4", "")
        candidates_dir = Path(local_output_dir) / "candidates" / source_video_id
        if not candidates_dir.exists():
            typer.echo(f"  Missing candidates dir for {entry.file_name}, skipping", err=True)
            failed += 1
            continue

        try:
            candidate_files = sorted(candidates_dir.glob("*.mp4"))
            metadata_files = sorted(candidates_dir.glob("*.json"))

            for cf in candidate_files:
                dest_key = f"anon/candidates/videos/{source_video_id}/{cf.name}"
                storage.upload(cf, dest_key)
                typer.echo(f"  Uploaded {cf.name} -> {dest_key}")

            for mf in metadata_files:
                dest_key = f"anon/candidates/metadata/{source_video_id}/{mf.name}"
                storage.upload(mf, dest_key)

            if upload_ledger_obj:
                upload_ledger_obj.mark_uploaded(entry.file_name)
                upload_ledger_obj.save()
            else:
                main_ledger.mark_uploaded(entry.file_name)
                main_ledger.save()
            uploaded += 1
        except Exception as exc:
            failed += 1
            typer.echo(f"  Failed {entry.file_name}: {exc}", err=True)

    typer.echo("")
    typer.echo("=== Upload Summary ===")
    typer.echo(f"  Uploaded: {uploaded}")
    typer.echo(f"  Failed:   {failed}")

    if failed > 0:
        raise SystemExit(1)


@app.command("ledger-reconcile")
def ledger_reconcile(
    main_ledger: str = typer.Option(
        ".local/candidate_staging/local_processing.csv",
        "--main-ledger",
        "-m",
        help="Path to the main processing ledger.",
    ),
    upload_ledger: str = typer.Option(
        ".local/candidate_staging/local_processing_upload.csv",
        "--upload-ledger",
        "-u",
        help="Path to the separate upload ledger.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Merge uploaded flags from a separate upload ledger into the main ledger.

    After running candidates-upload with --upload-ledger, use this to
    reconcile the two ledgers. Copies uploaded=true entries from the
    upload ledger into the main ledger without overwriting other fields.
    """
    _setup_logging(verbose)

    from pathlib import Path

    from pickup_putdown.remote.local_ledger import LocalProcessingLedger

    main = LocalProcessingLedger(Path(main_ledger))
    main.load()

    upload_path = Path(upload_ledger)
    if not upload_path.exists():
        typer.echo(f"Upload ledger not found: {upload_path}")
        raise SystemExit(1)

    upload = LocalProcessingLedger(upload_path)
    upload.load()

    merged = 0
    for fn, entry in upload.entries.items():
        if entry.uploaded:
            main_entry = main.get_entry(fn)
            if main_entry and not main_entry.uploaded:
                main_entry.uploaded = True
                main_entry.last_error = ""
                merged += 1
                logger.info("Reconciled: marked %s as uploaded in main ledger", fn)
            elif not main_entry:
                logger.warning("Entry %s uploaded but not in main ledger — skipping", fn)

    main.save()
    typer.echo(f"Reconciled {merged} uploaded entries into {main_ledger}")


@app.command("annotate-vlm")
def annotate_vlm(
    candidates_dir: str = typer.Argument(
        ...,
        help="Path to candidate staging directory containing candidate videos and metadata.",
    ),
    output_dir: str = typer.Option(
        ".local/vlm_annotations",
        "--output-dir",
        "-o",
        help="Output directory for VLM annotation results.",
    ),
    review_fps: float = typer.Option(
        5.0,
        "--review-fps",
        help="Frame extraction rate for review (frames per second).",
    ),
    max_frame_width: int = typer.Option(
        640,
        "--max-frame-width",
        help="Maximum width for extracted review frames.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Reprocess already-annotated candidates.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Process at most this many candidates.",
    ),
    annotator: str = typer.Option(
        "vlm_pipeline",
        "--annotator",
        help="Annotator identifier for output records.",
    ),
    vlm_base_url: str = typer.Option(
        "http://localhost:8080",
        "--vlm-base-url",
        help="llama.cpp server base URL for VLM calls.",
    ),
    vlm_model: str = typer.Option(
        "",
        "--vlm-model",
        help="Model name for VLM (auto-detected if empty).",
    ),
    vlm_temperature: float = typer.Option(
        0.0,
        "--vlm-temperature",
        help="Temperature for VLM sampling.",
    ),
    vlm_max_tokens: int = typer.Option(
        2048,
        "--vlm-max-tokens",
        help="Max tokens for VLM response.",
    ),
    vlm_timeout_s: int = typer.Option(
        120,
        "--vlm-timeout",
        help="Timeout in seconds for each VLM request.",
    ),
    no_vlm: bool = typer.Option(
        False,
        "--no-vlm",
        help="Disable VLM calls, produce frames only for manual review.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """VLM-assisted visual annotation of candidate videos.

    Discovers candidate videos, extracts review frames, creates contact
    sheets, sends them to a local VLM for analysis, and produces canonical
    event annotations. Outputs are written to the output directory in the
    canonical repository format.

    Usage:
        pickup-putdown annotate-vlm .local/candidate_staging/candidates \\
            --output-dir .local/vlm_annotations --limit 5
    """
    _setup_logging(verbose)

    from pathlib import Path

    from pickup_putdown.annotation.vlm_annotate import PipelineConfig, run_pipeline

    if not Path(candidates_dir).exists():
        typer.echo(f"Candidates directory not found: {candidates_dir}", err=True)
        raise SystemExit(1)

    config = PipelineConfig(
        candidates_dir=candidates_dir,
        output_dir=output_dir,
        review_fps=review_fps,
        max_frame_width=max_frame_width,
        force=force,
        limit=limit,
        annotator=annotator,
        vlm_base_url=vlm_base_url,
        vlm_model=vlm_model,
        vlm_temperature=vlm_temperature,
        vlm_max_tokens=vlm_max_tokens,
        vlm_timeout_s=vlm_timeout_s,
        vlm_enabled=not no_vlm,
    )

    summary = run_pipeline(config)

    typer.echo("")
    typer.echo("=== VLM Annotation Summary ===")
    typer.echo(f"  Total candidates:  {summary.total_candidates}")
    typer.echo(f"  Processed:         {summary.processed}")
    typer.echo(f"  Skipped:           {summary.skipped}")
    typer.echo(f"  Failed:            {summary.failed}")
    typer.echo(f"  Events found:      {summary.events_found}")
    typer.echo(f"  Time:              {summary.processing_time_s:.1f}s")
    typer.echo(f"  Output:            {output_dir}")

    if summary.errors:
        typer.echo("")
        typer.echo("Errors:")
        for err in summary.errors:
            typer.echo(f"  {err}")

    if summary.failed > 0:
        raise SystemExit(1)


@app.command("finalize-task-7")
def finalize_task_7(
    vlm_output_dir: str = typer.Option(
        ".local/vlm_annotations",
        "--vlm-output-dir",
        "-i",
        help="Path to VLM annotation output directory.",
    ),
    candidate_metadata_dir: str | None = typer.Option(
        None,
        "--candidate-metadata-dir",
        "-c",
        help="Path to candidate staging directory for clip metadata discovery.",
    ),
    source_videos_dir: str | None = typer.Option(
        None,
        "--source-videos-dir",
        "-s",
        help="Path to source video files for duration probing.",
    ),
    output_dir: str = typer.Option(
        ".local/task_7_vlm",
        "--output-dir",
        "-o",
        help="Output directory for Task 7 artifacts.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
    copy_artifacts: bool = typer.Option(
        False,
        "--copy-artifacts",
        help="Copy raw/ and normalized/ instead of symlinking (self-contained export).",
    ),
) -> None:
    """Finalize Task 7: build canonical dataset artifacts from VLM annotations.

    Reads normalized per-candidate JSON files and produces a reproducible
    artifact directory with clips.csv, events.csv, processing.csv,
    summary.json, provenance.json, and dedup_audit.json.

    By default, raw/ and normalized/ are relative symlinks to the VLM output
    directory. Use --copy-artifacts for a self-contained export.

    Usage:
        pickup-putdown finalize-task-7 \\
            --vlm-output-dir .local/vlm_annotations \\
            --output-dir .local/task_7_vlm
    """
    _setup_logging(verbose)

    from pathlib import Path

    from pickup_putdown.annotation.finalize_task7 import finalize_task_7 as run_finalizer

    vlm_path = Path(vlm_output_dir)
    if not vlm_path.is_dir():
        typer.echo(f"VLM output directory not found: {vlm_path}", err=True)
        raise SystemExit(1)

    normalized_dir = vlm_path / "normalized"
    if not normalized_dir.is_dir():
        typer.echo(
            f"Normalized candidates directory not found: {normalized_dir}",
            err=True,
        )
        raise SystemExit(1)

    result = run_finalizer(
        vlm_output_dir=vlm_output_dir,
        output_dir=output_dir,
        candidate_metadata_dir=candidate_metadata_dir,
        source_videos_dir=source_videos_dir,
        copy_artifacts=copy_artifacts,
    )

    typer.echo("")
    typer.echo("=== Task 7 Finalization Summary ===")
    typer.echo(f"  Candidates:    {result.candidates_count}")
    typer.echo(f"  Clips:         {result.clips_count}")
    typer.echo(f"  Events:        {result.events_count}")
    typer.echo(f"    Pickup:      {result.pickup_count}")
    typer.echo(f"    Putdown:     {result.putdown_count}")
    typer.echo(f"  Hard cases:    {result.hard_case_count}")
    typer.echo(f"  Confidence:    {result.confidence_counts}")
    typer.echo(f"  Output:        {result.output_dir}")
    typer.echo(f"  Artifacts:     {'copied' if copy_artifacts else 'symlinked'}")

    if not result.is_valid:
        typer.echo("")
        typer.echo(f"Validation errors: {len(result.errors)}")
        for err in result.errors[:20]:
            typer.echo(f"  [{err.source}] {err.message}")
        if len(result.errors) > 20:
            typer.echo(f"  ... and {len(result.errors) - 20} more")
        raise SystemExit(1)

    typer.echo("")
    typer.echo("Artifacts written successfully.")
    typer.echo("")
    typer.echo("NOTE: These are VLM pseudo-labels, not human-adjudicated ground truth.")
    typer.echo("Review required before use as final evaluation ground truth.")


@app.command()
def build_track_a_dataset(
    events_csv: str = typer.Option(
        ".local/task_7_vlm/events.csv",
        "--events-csv",
        "-e",
        help="Path to canonical events CSV.",
    ),
    clips_csv: str = typer.Option(
        ".local/task_7_vlm/clips.csv",
        "--clips-csv",
        help="Path to clips CSV.",
    ),
    review_manifest: str = typer.Option(
        ".local/task_7_review/review_manifest.csv",
        "--review-manifest",
        "-r",
        help="Path to review manifest CSV.",
    ),
    candidate_metadata_dir: str = typer.Option(
        ".local/candidate_staging",
        "--candidate-metadata-dir",
        help="Path to candidate staging directory.",
    ),
    source_video_dir: str = typer.Option(
        ".local/source_videos",
        "--source-video-dir",
        help="Path to source video directory.",
    ),
    output_dir: str = typer.Option(
        ".local/track_a_features",
        "--output-dir",
        "-o",
        help="Output directory for features and manifest.",
    ),
    split_seed: int = typer.Option(
        42,
        "--split-seed",
        help="Random seed for deterministic split assignment.",
    ),
    config: str = typer.Option(
        "configs/proposals.yaml",
        "--config",
        "-c",
        help="Path to configuration YAML file.",
    ),
    shelves_config: str = typer.Option(
        "configs/shelves.yaml",
        "--shelves-config",
        help="Path to shelf/surface region configuration YAML file.",
    ),
    camera_id: str = typer.Option(
        "store_camera_01",
        "--camera-id",
        help="Camera ID from the shelf configuration.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Build the reviewed Track A feature dataset.

    Loads the reviewed Task 7 data, resolves reviewed examples, assigns
    train/val/test splits by recording day, runs pose inference, extracts
    features, and saves the feature dataset manifest with cached embeddings.
    """
    _setup_logging(verbose)

    from pathlib import Path

    from pickup_putdown.config import load_config
    from pickup_putdown.layer1.track_a.reviewed_dataset import (
        build_reviewed_feature_dataset,
    )
    from pickup_putdown.perception.shelf_regions import (
        get_expanded_regions,
        get_regions_for_camera,
        load_shelf_config,
    )

    cfg = load_config(Path(config))

    # Load shelf regions
    shelf_cfg = load_shelf_config(Path(shelves_config))
    camera_cfg = get_regions_for_camera(shelf_cfg, camera_id)
    _expanded_regions = get_expanded_regions(camera_cfg)  # noqa: F841
    shelf_regions = {region.region_id: region.points for region in camera_cfg.regions}

    typer.echo("=== Building Reviewed Track A Dataset ===")
    typer.echo(f"  Review manifest:   {review_manifest}")
    typer.echo(f"  Events CSV:        {events_csv}")
    typer.echo(f"  Clips CSV:         {clips_csv}")
    typer.echo(f"  Candidate staging: {candidate_metadata_dir}")
    typer.echo(f"  Source videos:     {source_video_dir}")
    typer.echo(f"  Output dir:        {output_dir}")
    typer.echo(f"  Split seed:        {split_seed}")
    typer.echo(f"  Camera:            {camera_id}")
    typer.echo(f"  Shelf regions:     {len(shelf_regions)}")
    typer.echo("")

    try:
        dataset, summary = build_reviewed_feature_dataset(
            review_manifest_path=review_manifest,
            events_path=events_csv,
            clips_path=clips_csv,
            candidate_staging_dir=candidate_metadata_dir,
            source_video_dir=source_video_dir,
            output_dir=output_dir,
            pose_cfg=cfg.pose,
            track_a_cfg=cfg.track_a_features,
            shelf_regions=shelf_regions,
            split_seed=split_seed,
        )
    except Exception as exc:
        typer.echo(f"Error building dataset: {exc}", err=True)
        raise SystemExit(1) from exc

    typer.echo("")
    typer.echo("=== Build Summary ===")
    typer.echo(f"  Total reviewed:    {summary.total_reviewed}")
    typer.echo(f"  Positives:         {summary.positives}")
    typer.echo(f"  Negatives:         {summary.negatives}")
    typer.echo(f"  Excluded unreviewed: {summary.excluded_unreviewed}")
    typer.echo(f"  Total records:     {len(dataset.records)}")
    typer.echo("")
    typer.echo("  Records by split:")
    for split, count in sorted(summary.records_by_split.items()):
        typer.echo(f"    {split}: {count}")
    typer.echo("  Records by label:")
    for label, count in sorted(summary.records_by_label.items()):
        typer.echo(f"    {label}: {count}")
    typer.echo("  Records by position:")
    for pos, count in sorted(summary.records_by_position.items()):
        typer.echo(f"    {pos}: {count}")
    typer.echo("  Records by crop type:")
    for ct, count in sorted(summary.records_by_crop_type.items()):
        typer.echo(f"    {ct}: {count}")
    typer.echo("")
    typer.echo(f"  Manifest:          {output_dir}/feature_dataset.parquet")
    typer.echo(f"  Splits:            {output_dir}/splits.json")
    typer.echo(f"  Build summary:     {output_dir}/build_summary.json")


@app.command()
def train_track_a(
    config: str = typer.Option(
        "configs/track_a.yaml",
        "--config",
        "-c",
        help="Path to Track A classifier configuration YAML.",
    ),
    feature_manifest: str = typer.Option(
        ".local/track_a_features/feature_dataset.parquet",
        "--feature-manifest",
        "-m",
        help="Path to the Phase 1 feature dataset parquet.",
    ),
    output_dir: str = typer.Option(
        ".local/track_a_artifacts",
        "--output-dir",
        "-o",
        help="Output directory for classifier artifacts.",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="Override random seed from config.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing artifacts.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Train Track A hand-state and shelf-transition classifiers.

    Loads the reviewed Phase 1 feature dataset, derives supervised labels,
    trains logistic regression classifiers, evaluates on the validation split,
    and persists artifacts with metadata and metrics.
    """
    _setup_logging(verbose)

    import json

    import yaml

    from pickup_putdown.layer1.track_a.hand_state import (
        train_hand_classifier,
    )
    from pickup_putdown.layer1.track_a.manifest import load_manifest
    from pickup_putdown.layer1.track_a.shelf_state import (
        train_shelf_classifier,
    )

    # Load config
    cfg_path = Path(config)
    if not cfg_path.exists():
        typer.echo(f"Config not found: {cfg_path}", err=True)
        raise SystemExit(1)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}

    classifiers_cfg = cfg.get("classifiers", {})
    seed = seed if seed is not None else classifiers_cfg.get("random_seed", 42)

    hand_cfg = classifiers_cfg.get("hand_state", {})
    shelf_cfg = classifiers_cfg.get("shelf_state", {})

    # Check output directory
    output_path = Path(output_dir)
    if output_path.exists():
        existing = list(output_path.glob("*.joblib"))
        if existing and not force:
            typer.echo(
                f"Output directory exists with artifacts: {output_path}",
                err=True,
            )
            typer.echo("Use --force to overwrite.", err=True)
            raise SystemExit(1)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load feature dataset
    manifest_path = Path(feature_manifest)
    if not manifest_path.exists():
        typer.echo(f"Feature manifest not found: {manifest_path}", err=True)
        raise SystemExit(1)

    typer.echo("=== Training Track A Classifiers ===")
    typer.echo(f"  Config:           {config}")
    typer.echo(f"  Feature manifest: {feature_manifest}")
    typer.echo(f"  Output dir:       {output_dir}")
    typer.echo(f"  Seed:             {seed}")
    typer.echo("")

    dataset = load_manifest(manifest_path)
    typer.echo(f"  Total records:    {len(dataset.records)}")
    typer.echo(f"  Encoder:          {dataset.encoder_name}")
    typer.echo("")

    # --- Hand-state classifier ---
    typer.echo("--- Hand-State Classifier ---")
    try:
        hand_classifier, hand_report = train_hand_classifier(
            records=dataset.records,
            confidence_threshold=hand_cfg.get("confidence_threshold", 0.60),
            margin_threshold=hand_cfg.get("margin_threshold", 0.15),
            random_seed=seed,
            class_weight=hand_cfg.get("class_weight", "balanced"),
            max_iter=hand_cfg.get("max_iter", 1000),
        )

        # Save artifacts
        hand_joblib = output_path / "hand_state.joblib"
        hand_metadata = hand_classifier.build_metadata(
            encoder_name=dataset.encoder_name,
            encoder_version=dataset.encoder_version,
            training_record_counts=hand_report["train_class_counts"],
            train_split_count=hand_report["train_records"],
            val_split_count=hand_report["val_records"],
        )
        hand_classifier.save_pipeline(hand_joblib)
        (output_path / "hand_state_metadata.json").write_text(
            json.dumps(hand_metadata.to_dict(), indent=2) + "\n"
        )
        (output_path / "hand_state_metrics.json").write_text(
            json.dumps(hand_report, indent=2, default=str) + "\n"
        )

        typer.echo(f"  Train records:  {hand_report['train_records']}")
        typer.echo(f"  Train classes:  {hand_report['train_class_counts']}")
        typer.echo(f"  Val records:    {hand_report['val_records']}")
        if "val_class_counts" in hand_report:
            typer.echo(f"  Val classes:    {hand_report['val_class_counts']}")
        val = hand_report.get("validation", {})
        if "accuracy" in val:
            typer.echo(f"  Val accuracy:   {val['accuracy']:.4f}")
            typer.echo(f"  Val bal. acc:   {val['balanced_accuracy']:.4f}")
            typer.echo(f"  Val macro F1:   {val['macro_f1']:.4f}")
            typer.echo(
                f"  Uncertain:      {val.get('n_uncertain', '?')}/{val.get('n_samples', '?')}"
            )
        typer.echo(f"  Artifacts:      {hand_joblib}")
        typer.echo("")

    except Exception as exc:
        typer.echo(f"Hand-state training failed: {exc}", err=True)
        raise SystemExit(1) from exc

    # --- Shelf-transition classifier ---
    typer.echo("--- Shelf-Transition Classifier ---")
    try:
        shelf_classifier, shelf_report = train_shelf_classifier(
            records=dataset.records,
            confidence_threshold=shelf_cfg.get("confidence_threshold", 0.60),
            margin_threshold=shelf_cfg.get("margin_threshold", 0.15),
            random_seed=seed,
            class_weight=shelf_cfg.get("class_weight", "balanced"),
            max_iter=shelf_cfg.get("max_iter", 1000),
        )

        # Save artifacts
        shelf_joblib = output_path / "shelf_state.joblib"
        shelf_metadata = shelf_classifier.build_metadata(
            encoder_name=dataset.encoder_name,
            encoder_version=dataset.encoder_version,
            training_record_counts=shelf_report["train_class_counts"],
            train_split_count=shelf_report["train_records"],
            val_split_count=shelf_report["val_records"],
        )
        shelf_classifier.save_pipeline(shelf_joblib)
        (output_path / "shelf_state_metadata.json").write_text(
            json.dumps(shelf_metadata.to_dict(), indent=2) + "\n"
        )
        (output_path / "shelf_state_metrics.json").write_text(
            json.dumps(shelf_report, indent=2, default=str) + "\n"
        )

        typer.echo(f"  Train records:  {shelf_report['train_records']}")
        typer.echo(f"  Train classes:  {shelf_report['train_class_counts']}")
        typer.echo(f"  Val records:    {shelf_report['val_records']}")
        if "val_class_counts" in shelf_report:
            typer.echo(f"  Val classes:    {shelf_report['val_class_counts']}")
        val = shelf_report.get("validation", {})
        if "accuracy" in val:
            typer.echo(f"  Val accuracy:   {val['accuracy']:.4f}")
            typer.echo(f"  Val bal. acc:   {val['balanced_accuracy']:.4f}")
            typer.echo(f"  Val macro F1:   {val['macro_f1']:.4f}")
            typer.echo(
                f"  Uncertain:      {val.get('n_uncertain', '?')}/{val.get('n_samples', '?')}"
            )
        typer.echo(f"  Artifacts:      {shelf_joblib}")
        typer.echo("")

    except Exception as exc:
        typer.echo(f"Shelf-state training failed: {exc}", err=True)
        raise SystemExit(1) from exc

    typer.echo("=== Training Complete ===")
    typer.echo(f"  Artifacts saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Track A inference command (Phase 5)
# ---------------------------------------------------------------------------


@app.command(name="infer-track-a")
def infer_track_a(
    config: str = typer.Option(
        "configs/track_a.yaml",
        "--config",
        "-c",
        help="Path to Track A configuration YAML.",
    ),
    candidate_metadata: str = typer.Option(
        ".local/candidate_staging/metadata",
        "--candidate-metadata",
        help="Directory containing candidate metadata JSON files.",
    ),
    candidates: str | None = typer.Option(
        None,
        "--candidates",
        help=(
            "Path to candidates parquet file. When provided, used instead of "
            "candidate metadata directory."
        ),
    ),
    pose_observations: str | None = typer.Option(
        None,
        "--pose-observations",
        help=(
            "Path to tracks_pose.parquet or directory containing per-clip "
            "pose parquet files. Auto-detected from .local/remote_candidates/ "
            "when omitted."
        ),
    ),
    source_video_dir: str = typer.Option(
        ".local/source_videos",
        "--source-video-dir",
        help="Directory containing source video files.",
    ),
    shelves_config: str = typer.Option(
        "configs/shelves.yaml",
        "--shelves-config",
        help="Path to shelf/surface region configuration YAML.",
    ),
    camera_id: str = typer.Option(
        "store_camera_01",
        "--camera-id",
        help="Camera ID from the shelf configuration.",
    ),
    artifact_dir: str = typer.Option(
        ".local/track_a_artifacts",
        "--artifact-dir",
        help="Directory containing trained classifier artifacts.",
    ),
    cache_dir: str = typer.Option(
        ".local/track_a_features",
        "--cache-dir",
        help="Directory for cached feature embeddings.",
    ),
    output_dir: str = typer.Option(
        ".local/track_a_output",
        "--output-dir",
        "-o",
        help="Output directory for predictions and diagnostics.",
    ),
    clip_id: str | None = typer.Option(
        None,
        "--clip-id",
        help="Process only candidates from this clip.",
    ),
    candidate_id: str | None = typer.Option(
        None,
        "--candidate-id",
        help="Process only this specific candidate.",
    ),
    debug_traces: bool = typer.Option(
        False,
        "--debug-traces",
        help="Enable per-observation debug traces in diagnostics.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing output files.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Run Track A inference pipeline on candidates.

    Loads candidates, pose data, shelf regions, source videos, and classifier
    artifacts, then runs the full inference pipeline with feature extraction,
    classifier prediction, state-machine processing, boundary refinement, and
    deduplication.
    """
    _setup_logging(verbose)

    from pathlib import Path

    import yaml

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    cfg_path = Path(config)
    if not cfg_path.exists():
        typer.echo(f"Config not found: {cfg_path}", err=True)
        raise SystemExit(1)

    with open(cfg_path) as f:
        cfg_data = yaml.safe_load(f) or {}

    # ------------------------------------------------------------------
    # Resolve inference config
    # ------------------------------------------------------------------
    from pickup_putdown.layer1.track_a.inference import (
        InferenceConfig,
        load_inference_config,
    )
    from pickup_putdown.layer1.track_a.state_machine import StateMachineConfig

    inf_cfg = load_inference_config(cfg_path)
    if debug_traces:
        inf_cfg = InferenceConfig(
            sampling=inf_cfg.sampling,
            boundary_refinement=inf_cfg.boundary_refinement,
            deduplication=inf_cfg.deduplication,
            transition_grace_s=inf_cfg.transition_grace_s,
            debug_traces=True,
        )

    sm_cfg_dict = cfg_data.get("state_machine", {})
    # Map nested confidence_weights to flat fields expected by StateMachineConfig
    if sm_cfg_dict:
        cw = sm_cfg_dict.pop("confidence_weights", {})
        if isinstance(cw, dict):
            sm_cfg_dict.setdefault("confidence_weight_hand", cw.get("hand", 0.40))
            sm_cfg_dict.setdefault("confidence_weight_shelf", cw.get("shelf", 0.40))
            sm_cfg_dict.setdefault("confidence_weight_trajectory", cw.get("trajectory", 0.20))
    sm_cfg = StateMachineConfig(**sm_cfg_dict) if sm_cfg_dict else StateMachineConfig()

    # ------------------------------------------------------------------
    # Resolve shelf regions
    # ------------------------------------------------------------------
    from pickup_putdown.perception.shelf_regions import (
        get_regions_for_camera,
        load_shelf_config,
    )

    shelf_cfg_path = Path(shelves_config)
    if not shelf_cfg_path.exists():
        typer.echo(f"Shelf config not found: {shelf_cfg_path}", err=True)
        raise SystemExit(1)

    shelf_cfg = load_shelf_config(shelf_cfg_path)
    if camera_id not in shelf_cfg.cameras:
        typer.echo(
            f"Unknown camera ID {camera_id!r}. Available: {list(shelf_cfg.cameras)}",
            err=True,
        )
        raise SystemExit(1)

    camera_cfg = get_regions_for_camera(shelf_cfg, camera_id)
    shelf_regions = {r.region_id: r.points for r in camera_cfg.regions}

    # ------------------------------------------------------------------
    # Resolve classifier artifacts
    # ------------------------------------------------------------------
    artifact_path = Path(artifact_dir)
    hand_artifact = artifact_path / "hand_state.joblib"
    shelf_artifact = artifact_path / "shelf_state.joblib"

    if not hand_artifact.exists():
        typer.echo(f"Hand classifier artifact not found: {hand_artifact}", err=True)
        raise SystemExit(1)
    if not shelf_artifact.exists():
        typer.echo(f"Shelf classifier artifact not found: {shelf_artifact}", err=True)
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # Resolve output directory
    # ------------------------------------------------------------------
    output_path = Path(output_dir)
    if output_path.exists():
        existing_outputs = [
            output_path / "predictions.csv",
            output_path / "raw_state_machine_events.json",
            output_path / "inference_summary.json",
        ]
        if any(p.exists() for p in existing_outputs) and not force:
            typer.echo(f"Output directory exists with previous results: {output_path}", err=True)
            typer.echo("Use --force to overwrite.", err=True)
            raise SystemExit(1)
    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load candidates
    # ------------------------------------------------------------------
    all_candidates = _load_candidates_for_inference(
        candidate_metadata_dir=Path(candidate_metadata),
        candidates_path=Path(candidates) if candidates else None,
    )

    if not all_candidates:
        typer.echo("No candidates found. Check --candidate-metadata and --candidates.", err=True)
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # Enrich candidates with actor_id/hand_side/region_id from feature dataset
    # ------------------------------------------------------------------
    feature_manifest = Path(cache_dir) / "feature_dataset.parquet"
    if feature_manifest.exists():
        all_candidates = _enrich_candidates_from_feature_dataset(all_candidates, feature_manifest)

    # ------------------------------------------------------------------
    # Filter candidates
    # ------------------------------------------------------------------
    filtered_candidates = _filter_candidates(
        all_candidates,
        clip_id=clip_id,
        candidate_id=candidate_id,
    )

    if not filtered_candidates:
        if clip_id:
            typer.echo(
                f"No candidates found for clip {clip_id!r}.",
                err=True,
            )
        elif candidate_id:
            typer.echo(
                f"Candidate {candidate_id!r} not found.",
                err=True,
            )
        else:
            typer.echo("No candidates matched the filter criteria.", err=True)
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # Resolve source videos
    # ------------------------------------------------------------------
    source_video_path = Path(source_video_dir)
    source_videos = _resolve_source_videos(filtered_candidates, source_video_path)

    # ------------------------------------------------------------------
    # Resolve pose observations
    # ------------------------------------------------------------------
    all_pose_obs = _resolve_pose_observations(
        filtered_candidates,
        pose_path=Path(pose_observations) if pose_observations else None,
        source_video_dir=source_video_path,
    )

    # ------------------------------------------------------------------
    # Resolve clip durations
    # ------------------------------------------------------------------
    clip_durations = _probe_clip_durations(source_videos)

    # ------------------------------------------------------------------
    # Resolve embedder
    # ------------------------------------------------------------------
    from pickup_putdown.layer1.track_a.image_features import (
        TorchVisionEmbedder,
    )

    embedder_cfg_dict = cfg_data.get("track_a_features", {})
    encoder_name = embedder_cfg_dict.get("encoder_name", "mobilenet_v3_small")
    hand_crop_size = embedder_cfg_dict.get("hand_crop_size", 224)
    shelf_patch_size = embedder_cfg_dict.get("shelf_patch_size", 224)

    embedder = TorchVisionEmbedder(encoder_name)

    # ------------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------------
    typer.echo("=== Track A Inference ===")
    typer.echo(f"  Config:           {config}")
    typer.echo(f"  Candidates:       {len(filtered_candidates)}")
    typer.echo(f"  Pose obs:         {len(all_pose_obs)}")
    typer.echo(f"  Source videos:    {len(source_videos)}")
    typer.echo(f"  Shelf regions:    {len(shelf_regions)}")
    typer.echo(f"  Artifacts:        {artifact_dir}")
    typer.echo(f"  Cache dir:        {cache_dir}")
    typer.echo(f"  Output dir:       {output_dir}")
    typer.echo("")

    from pickup_putdown.layer1.track_a.inference import TrackAInferencePipeline

    pipeline = TrackAInferencePipeline(
        config=inf_cfg,
        state_machine_config=sm_cfg,
    )

    # Convert dicts to objects for pipeline getattr() compatibility
    obj_candidates = [
        SimpleNamespace(**c) if isinstance(c, dict) else c for c in filtered_candidates
    ]
    obj_poses = [SimpleNamespace(**p) if isinstance(p, dict) else p for p in all_pose_obs]

    try:
        result = pipeline.run(
            candidates=obj_candidates,
            pose_observations=obj_poses,
            source_videos=source_videos,
            hand_classifier_path=hand_artifact,
            shelf_classifier_path=shelf_artifact,
            output_dir=output_path,
            shelf_regions=shelf_regions,
            embedder=embedder,
            cache_dir=cache_dir,
            clip_durations=clip_durations,
            hand_crop_size=hand_crop_size,
            shelf_patch_size=shelf_patch_size,
        )
    except Exception as exc:
        typer.echo(f"Inference failed: {exc}", err=True)
        raise SystemExit(1) from exc

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    typer.echo("")
    typer.echo("=== Inference Summary ===")
    typer.echo(f"  Candidates processed: {result.summary.candidates_processed}")
    typer.echo(f"  Candidates skipped:   {result.summary.candidates_skipped}")
    typer.echo(f"  Total samples:        {result.summary.total_samples}")
    typer.echo(f"  Cache hits:           {result.summary.feature_cache_hits}")
    typer.echo(f"  Cache misses:         {result.summary.feature_cache_misses}")
    typer.echo(f"  Raw events:           {result.summary.raw_events_emitted}")
    typer.echo(f"  Final predictions:    {result.summary.final_events_after_dedup}")
    typer.echo(f"  Pickups:              {result.summary.pickup_count}")
    typer.echo(f"  Putdowns:             {result.summary.putdown_count}")
    if result.summary.final_events_after_dedup > 0:
        typer.echo(f"  Mean confidence:    {result.summary.mean_confidence:.4f}")
    typer.echo(f"  Output directory:     {output_path}")

    if result.output_paths:
        typer.echo("")
        typer.echo("  Output files:")
        for key, path in result.output_paths.items():
            typer.echo(f"    {key}: {path}")

    if result.summary.candidates_processed == 0 and result.summary.candidates_skipped > 0:
        typer.echo("")
        typer.echo("  Skip reasons:")
        for reason, count in result.summary.skip_reasons.items():
            typer.echo(f"    {reason}: {count}")


def _load_candidates_for_inference(
    candidate_metadata_dir: Path,
    candidates_path: Path | None = None,
) -> list[dict[str, object]]:
    """Load candidates from metadata directory or parquet file.

    Returns list of dicts with at least: candidate_id, clip_id,
    source_start_s, source_end_s, actor_id, hand_side, region_id.
    """
    if candidates_path and candidates_path.exists():
        import pyarrow.parquet as pq

        table = pq.read_table(str(candidates_path))
        df = table.to_pandas()
        records = []
        for _, row in df.iterrows():
            records.append(
                {
                    "candidate_id": row.get("candidate_id", ""),
                    "clip_id": row.get("clip_id", ""),
                    "raw_start_s": float(row.get("raw_start_s", 0.0)),
                    "raw_end_s": float(row.get("raw_end_s", 0.0)),
                    "window_start_s": float(
                        row.get("window_start_s", row.get("raw_start_s", 0.0))
                    ),
                    "window_end_s": float(row.get("window_end_s", row.get("raw_end_s", 0.0))),
                    "actor_id": str(row.get("actor_id", ""))
                    if pd_notna(row.get("actor_id"))
                    else "",
                    "hand_side": str(row.get("hand_side", ""))
                    if pd_notna(row.get("hand_side"))
                    else "",
                    "region_id": str(row.get("region_id", ""))
                    if pd_notna(row.get("region_id"))
                    else "",
                }
            )
        return records

    # Load from candidate staging metadata
    # Two layouts supported:
    #   1) <dir>/candidates/<clip_id>/<clip_id>.json  (remote_candidates layout)
    #   2) <dir>/<clip_id>/<clip_id>.json             (candidate_staging/metadata layout)
    index: dict[str, dict] = {}

    candidates_subdir = candidate_metadata_dir / "candidates"
    if candidates_subdir.exists():
        from pickup_putdown.layer1.track_a.reviewed_dataset import (
            load_candidate_metadata_index,
        )

        index = load_candidate_metadata_index(candidate_metadata_dir)
    elif candidate_metadata_dir.exists():
        # ponytail: direct scan of <dir>/<clip_id>/<clip_id>.json
        for clip_dir in sorted(candidate_metadata_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            meta_file = clip_dir / f"{clip_dir.name}.json"
            if not meta_file.exists():
                continue
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            clip_id = data.get("source_video_id", clip_dir.name)
            for cand in data.get("candidates", []):
                cid = cand.get("candidate_id")
                if not cid:
                    continue
                index[cid] = {
                    "candidate_id": cid,
                    "clip_id": clip_id,
                    "source_start_s": float(cand.get("source_start_s", 0)),
                    "source_end_s": float(cand.get("source_end_s", 0)),
                    "actor_id": cand.get("actor_id"),
                    "hand_side": cand.get("hand_side"),
                    "region_id": cand.get("region_id"),
                }

    records = []
    for meta in index.values():
        if isinstance(meta, dict):
            records.append(
                {
                    "candidate_id": meta["candidate_id"],
                    "clip_id": meta["clip_id"],
                    "raw_start_s": meta["source_start_s"],
                    "raw_end_s": meta["source_end_s"],
                    "window_start_s": meta["source_start_s"],
                    "window_end_s": meta["source_end_s"],
                    "actor_id": meta.get("actor_id") or "",
                    "hand_side": meta.get("hand_side") or "",
                    "region_id": meta.get("region_id") or "",
                }
            )
        else:
            records.append(
                {
                    "candidate_id": meta.candidate_id,
                    "clip_id": meta.clip_id,
                    "raw_start_s": meta.source_start_s,
                    "raw_end_s": meta.source_end_s,
                    "window_start_s": meta.source_start_s,
                    "window_end_s": meta.source_end_s,
                    "actor_id": meta.actor_id or "",
                    "hand_side": meta.hand_side or "",
                    "region_id": meta.region_id or "",
                }
            )
    return records


def _enrich_candidates_from_feature_dataset(
    candidates: list[dict[str, object]],
    feature_manifest: Path,
) -> list[dict[str, object]]:
    """Enrich candidates with actor_id/hand_side/region_id from feature dataset."""
    import pyarrow.parquet as pq

    table = pq.read_table(str(feature_manifest))
    df = table.to_pandas()

    # Build lookup: candidate_id -> {actor_id, hand_side, region_id}
    lookup: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        cid = row.get("candidate_id", "")
        if not cid:
            continue
        if cid not in lookup:
            lookup[cid] = {
                "actor_id": str(row.get("actor_id", "")) if pd_notna(row.get("actor_id")) else "",
                "hand_side": str(row.get("hand_side", ""))
                if pd_notna(row.get("hand_side"))
                else "",
                "region_id": str(row.get("region_id", ""))
                if pd_notna(row.get("region_id"))
                else "",
            }

    enriched = []
    for cand in candidates:
        cid = cand["candidate_id"]
        info = lookup.get(cid, {})
        c = dict(cand)
        if not c.get("actor_id") and info.get("actor_id"):
            c["actor_id"] = info["actor_id"]
        if not c.get("hand_side") and info.get("hand_side"):
            c["hand_side"] = info["hand_side"]
        if not c.get("region_id") and info.get("region_id"):
            c["region_id"] = info["region_id"]
        enriched.append(c)
    return enriched


def _filter_candidates(
    candidates: list[dict[str, object]],
    clip_id: str | None = None,
    candidate_id: str | None = None,
) -> list[dict[str, object]]:
    """Filter candidates by clip_id and/or candidate_id."""
    filtered = candidates
    if candidate_id:
        filtered = [c for c in filtered if c["candidate_id"] == candidate_id]
    if clip_id:
        filtered = [c for c in filtered if c["clip_id"] == clip_id]
    return filtered


def _resolve_source_videos(
    candidates: list[dict[str, object]],
    source_video_dir: Path,
) -> dict[str, Path]:
    """Resolve source video paths for unique clip IDs."""
    clip_ids = sorted({c["clip_id"] for c in candidates})
    videos: dict[str, Path] = {}
    for clip_id in clip_ids:
        video_path = source_video_dir / f"{clip_id}.mp4"
        if video_path.exists():
            videos[clip_id] = video_path
    return videos


def _resolve_pose_observations(
    candidates: list[dict[str, object]],
    pose_path: Path | None = None,
    source_video_dir: Path | None = None,
) -> list[dict[str, object]]:
    """Resolve pose observations from parquet files.

    Search order:
    1. Explicit --pose-observations path (file or directory)
    2. Auto-detect from .local/remote_candidates/
    3. Empty list (pipeline will skip feature extraction)
    """
    import pyarrow.parquet as pq

    clip_ids = sorted({c["clip_id"] for c in candidates})
    all_obs: list[dict[str, object]] = []

    pose_files: list[Path] = []

    if pose_path:
        if pose_path.is_file():
            pose_files = [pose_path]
        elif pose_path.is_dir():
            pose_files = sorted(pose_path.glob("*.parquet"))
    else:
        # Auto-detect from .local/remote_candidates/
        remote_dir = Path(".local/remote_candidates")
        if remote_dir.exists():
            for clip_id in clip_ids:
                found = _find_pose_file_for_clip(remote_dir, clip_id)
                if found:
                    pose_files.append(found)

    for pf in pose_files:
        try:
            table = pq.read_table(str(pf))
            df = table.to_pandas()
            for _, row in df.iterrows():
                cid = str(row.get("clip_id", ""))
                # ponytail: strip clip_ prefix to match candidate clip_id format
                if cid.startswith("clip_"):
                    cid = cid[5:]
                all_obs.append(
                    {
                        "clip_id": cid,
                        "timestamp_s": float(row.get("timestamp_s", 0.0)),
                        "actor_id": str(row.get("actor_id", "")),
                        "hand_side": str(row.get("hand_side", "")),
                        "wrist_x": float(row.get("wrist_x", 0.0)),
                        "wrist_y": float(row.get("wrist_y", 0.0)),
                        "wrist_confidence": float(row.get("wrist_confidence", 0.5)),
                    }
                )
        except Exception as exc:
            logger.warning("Failed to load pose file %s: %s", pf, exc)

    return all_obs


def _find_pose_file_for_clip(
    remote_dir: Path,
    clip_id: str,
) -> Path | None:
    """Find tracks_pose.parquet for a clip in remote_candidates directory."""
    import fnmatch

    for parquet in remote_dir.rglob("tracks_pose.parquet"):
        # The parquet is in <run>/<clip_id>/intermediate/task_5/tracks_pose.parquet
        # or similar structure. Check if clip_id matches any ancestor.
        parts = parquet.parts
        for part in parts:
            if fnmatch.fnmatch(part, clip_id) or part == clip_id:
                return parquet
    return None


def _probe_clip_durations(
    source_videos: dict[str, Path],
) -> dict[str, float]:
    """Probe video durations using ffprobe."""
    from pickup_putdown.ingestion.video_probe import probe_video

    durations: dict[str, float] = {}
    for clip_id, video_path in source_videos.items():
        try:
            probe = probe_video(video_path)
            if probe.decode_ok and probe.duration_s:
                durations[clip_id] = float(probe.duration_s)
        except Exception as exc:
            logger.warning("Failed to probe %s: %s", video_path, exc)
    return durations


def pd_notna(val: object) -> bool:
    """Check if a value is not NaN/None (handles pandas NaN)."""
    if val is None:
        return False
    import math

    return not (isinstance(val, float) and math.isnan(val))


# ---------------------------------------------------------------------------
# Track A evaluation command (Phase 6)
# ---------------------------------------------------------------------------


@app.command(name="evaluate-track-a")
def evaluate_track_a(
    config: str = typer.Option(
        "configs/track_a.yaml",
        "--config",
        "-c",
        help="Path to Track A configuration YAML.",
    ),
    splits: str = typer.Option(
        ".local/track_a_features/splits.json",
        "--splits",
        help="Path to splits.json from the feature dataset.",
    ),
    feature_manifest: str = typer.Option(
        ".local/track_a_features/feature_dataset.parquet",
        "--feature-manifest",
        help="Path to the feature dataset manifest.",
    ),
    events: str = typer.Option(
        ".local/task_7_vlm/events.csv",
        "--events",
        "-e",
        help="Path to canonical ground-truth events CSV.",
    ),
    clips: str = typer.Option(
        ".local/task_7_vlm/clips.csv",
        "--clips",
        help="Path to clips CSV.",
    ),
    artifact_dir: str = typer.Option(
        ".local/track_a_artifacts",
        "--artifact-dir",
        help="Directory containing trained classifier artifacts.",
    ),
    candidate_metadata: str = typer.Option(
        ".local/candidate_staging/metadata",
        "--candidate-metadata",
        help="Directory containing candidate metadata JSON files.",
    ),
    source_video_dir: str = typer.Option(
        ".local/source_videos",
        "--source-video-dir",
        help="Directory containing source video files.",
    ),
    shelves_config: str = typer.Option(
        "configs/shelves.yaml",
        "--shelves-config",
        help="Path to shelf/surface region configuration YAML.",
    ),
    camera_id: str = typer.Option(
        "store_camera_01",
        "--camera-id",
        help="Camera ID from the shelf configuration.",
    ),
    output_dir: str = typer.Option(
        ".local/track_a_evaluation",
        "--output-dir",
        "-o",
        help="Output directory for evaluation results.",
    ),
    split: str = typer.Option(
        "val",
        "--split",
        help="Dataset split to evaluate (train, val, test).",
    ),
    limit_clips: int | None = typer.Option(
        None,
        "--limit-clips",
        help="Evaluate only the first N clips from the split (deterministic).",
    ),
    clip_id: str | None = typer.Option(
        None,
        "--clip-id",
        help="Evaluate only this single clip.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing output files.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Run Track A evaluation: inference + Task-8 metrics + reports.

    Resolves clips from a split, validates split isolation, runs inference,
    combines predictions, filters ground truth, invokes the Task 8 evaluator,
    and exports metrics, failure tables, and a Markdown report.

    Default split is ``val`` (development data). Metrics are labeled as
    validation metrics, not independent test performance.
    """
    _setup_logging(verbose)

    from pathlib import Path

    from pickup_putdown.layer1.track_a.evaluation import (
        EvaluationSummary,
    )
    from pickup_putdown.layer1.track_a.evaluation import (
        evaluate_track_a as _evaluate,
    )

    try:
        summary: EvaluationSummary = _evaluate(
            config=config,
            splits=Path(splits),
            feature_manifest=Path(feature_manifest),
            events=Path(events),
            clips=Path(clips),
            artifact_dir=Path(artifact_dir),
            candidate_metadata=Path(candidate_metadata),
            source_video_dir=Path(source_video_dir),
            shelves_config=Path(shelves_config),
            camera_id=camera_id,
            output_dir=Path(output_dir),
            split=split,
            limit_clips=limit_clips,
            clip_id=clip_id,
            force=force,
            verbose=verbose,
        )
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from exc
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from exc

    typer.echo("")
    typer.echo("=== Track A Evaluation Summary ===")
    typer.echo(f"  Split:             {summary.split}")
    typer.echo(f"  Limited run:       {'yes' if summary.limited else 'no'}")
    if summary.limited:
        typer.echo(f"  Limit:             {summary.limit_count} clip(s)")
    typer.echo(f"  Total clips:       {summary.total_clips}")
    typer.echo(f"  Evaluated:         {summary.evaluated_clips}")
    typer.echo(f"  Skipped:           {summary.skipped_clips}")
    typer.echo(f"  GT events:         {summary.gt_event_count}")
    typer.echo(f"  Predictions:       {summary.pred_event_count}")
    typer.echo(f"  Pickups:           {summary.pickup_count}")
    typer.echo(f"  Putdowns:          {summary.putdown_count}")
    if summary.mean_confidence > 0:
        typer.echo(f"  Mean confidence:   {summary.mean_confidence:.4f}")

    metrics = summary.metrics
    for thr in [0.3, 0.5]:
        key = f"tiou@{thr}"
        if key in metrics:
            m = metrics[key]
            typer.echo(
                f"  tIoU@{thr}  P={m.get('precision', 0):.4f} "
                f"R={m.get('recall', 0):.4f} F1={m.get('f1', 0):.4f}"
            )

    typer.echo(f"  Leakage check:     {summary.leakage_check}")

    if summary.skipped:
        typer.echo("")
        typer.echo("  Skipped clips:")
        for cs in summary.skipped:
            typer.echo(f"    {cs.clip_id}: {cs.status} ({cs.reason})")

    typer.echo("")
    typer.echo(f"  Outputs: {output_dir}")
    typer.echo(f"    predictions.csv        {output_dir}/predictions.csv")
    typer.echo(f"    ground_truth.csv       {output_dir}/ground_truth.csv")
    typer.echo(f"    matches.csv            {output_dir}/matches.csv")
    typer.echo(f"    false_positives.csv    {output_dir}/false_positives.csv")
    typer.echo(f"    false_negatives.csv    {output_dir}/false_negatives.csv")
    typer.echo(f"    metrics.json           {output_dir}/metrics.json")
    typer.echo(f"    evaluation_summary.json {output_dir}/evaluation_summary.json")
    typer.echo(f"    validation_report.md   {output_dir}/validation_report.md")

    if summary.skipped_clips > 0:
        typer.echo("")
        typer.echo("  WARNING: Some clips were skipped. See evaluation_summary.json.")


if __name__ == "__main__":
    app()
