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
