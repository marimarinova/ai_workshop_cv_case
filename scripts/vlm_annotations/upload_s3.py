#!/usr/bin/env python3
"""Upload VLM annotation artifacts to S3 with dated prefix."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

STORAGE_CONFIG_PATH: Final = "configs/storage.s3.yaml"
LOCAL_ROOT: Final = ".local"


@dataclass
class ArtifactDirectory:
    """Describes a local directory to upload and its S3 target."""

    name: str
    local_path: Path
    upload_prefix: str
    skip_patterns: list[str] = field(default_factory=list)
    skip_symlinks: bool = True

    def collect_files(self) -> list[Path]:
        """Collect all regular files under local_path, respecting skip rules."""
        collected: list[Path] = []
        for p in sorted(self.local_path.rglob("*")):
            if p.is_dir():
                continue
            if self.skip_symlinks and p.is_symlink():
                logger.debug("Skipping symlink: %s", p)
                continue
            for pattern in self.skip_patterns:
                if pattern in str(p):
                    logger.debug("Skipping (pattern %s): %s", pattern, p)
                    break
            else:
                collected.append(p)
        return collected

    def relative_key(self, file_path: Path) -> str:
        """Return S3 key relative to upload_prefix."""
        rel = file_path.relative_to(self.local_path)
        return f"{self.upload_prefix}/{self.name}/{rel}"


@dataclass
class UploadStats:
    """Accumulates upload statistics."""

    uploaded: int = 0
    skipped: int = 0
    failed: int = 0
    bytes_uploaded: int = 0

    def summary(self) -> str:
        mb = self.bytes_uploaded / (1024 * 1024)
        lines = [
            f"  Uploaded: {self.uploaded} files ({mb:.1f} MB)",
            f"  Skipped:  {self.skipped} files",
            f"  Failed:   {self.failed} files",
        ]
        return "\n".join(lines)


class S3Uploader:
    """Wraps boto3 S3 client for artifact upload."""

    def __init__(
        self, bucket: str, region: str | None = None, endpoint_url: str | None = None
    ) -> None:
        import boto3

        kwargs: dict[str, str] = {}
        if region:
            kwargs["region_name"] = region
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url
        self.bucket = bucket
        self.client = boto3.client("s3", **kwargs)

    def upload_file(self, local_path: Path, key: str) -> int:
        """Upload a single file. Returns size in bytes."""
        return _upload_file(self.client, local_path, self.bucket, key)


def _upload_file(s3_client: object, local_path: Path, bucket: str, key: str) -> int:
    """Upload a single file via boto3. Returns size in bytes."""
    s3_client.upload_file(str(local_path), bucket, key)
    return local_path.stat().st_size


def _parse_bucket_uri(bucket_uri: str) -> tuple[str, str]:
    """Parse s3://bucket/prefix into (bucket, prefix)."""
    if not bucket_uri.startswith("s3://"):
        raise ValueError(f"Invalid bucket URI: {bucket_uri}")
    rest = bucket_uri[5:]
    slash = rest.index("/")
    return rest[:slash], rest[slash + 1 :].rstrip("/")


def _load_storage_config(config_path: str) -> tuple[str, str, str | None, str | None]:
    """Load storage config YAML. Returns (bucket, prefix, region, endpoint_url)."""
    from pickup_putdown.config import load_config

    cfg = load_config(Path(config_path))
    storage = cfg.storage

    if not storage.bucket_uri:
        raise SystemExit("Error: storage.bucket_uri is empty in config.")

    bucket, prefix = _parse_bucket_uri(storage.bucket_uri)
    return bucket, prefix, storage.region, storage.endpoint_url


class UploadSession:
    """Orchestrates the upload of VLM annotation artifacts."""

    def __init__(self, bucket: str, prefix: str, upload_date: str, dry_run: bool = False) -> None:
        self.bucket = bucket
        self.prefix = prefix
        self.upload_date = upload_date
        self.dry_run = dry_run
        self.stats = UploadStats()

        base = f"{prefix}/vlm/{upload_date}"
        self.directories = [
            ArtifactDirectory(
                name="vlm_annotations",
                local_path=Path(LOCAL_ROOT) / "vlm_annotations",
                upload_prefix=base,
                skip_patterns=["review_frames", "logs", ".bak"],
            ),
            ArtifactDirectory(
                name="task_7_review",
                local_path=Path(LOCAL_ROOT) / "task_7_review",
                upload_prefix=base,
                skip_patterns=[".bak"],
            ),
            ArtifactDirectory(
                name="task_7_vlm",
                local_path=Path(LOCAL_ROOT) / "task_7_vlm",
                upload_prefix=base,
                skip_patterns=[".bak"],
            ),
        ]

    def run(self) -> None:
        """Execute the upload."""
        if not self.dry_run:
            self.uploader = S3Uploader(
                self.bucket,
                region=None,
                endpoint_url=None,
            )

        for directory in self.directories:
            self._upload_directory(directory)

        print("\n=== Upload Summary ===")
        print(self.stats.summary())

        if self.stats.failed > 0:
            raise SystemExit(1)

    def _upload_directory(self, directory: ArtifactDirectory) -> None:
        """Upload all files from a single artifact directory."""
        files = directory.collect_files()
        if not files:
            logger.info("No files to upload in %s", directory.name)
            return

        print(f"\n--- {directory.name}: {len(files)} file(s) ---")
        logger.info("Processing %s: %d file(s)", directory.name, len(files))

        for file_path in files:
            key = directory.relative_key(file_path)
            if self.dry_run:
                print(f"  [DRY-RUN] Would upload: {key}")
                self.stats.uploaded += 1
                continue

            try:
                size = self.uploader.upload_file(file_path, key)
                self.stats.uploaded += 1
                self.stats.bytes_uploaded += size
                logger.debug("Uploaded %s (%d bytes)", key, size)
            except Exception as exc:
                self.stats.failed += 1
                logger.error("Failed to upload %s: %s", key, exc)
                print(f"  FAILED: {key}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload VLM annotation artifacts to S3 with dated prefix."
    )
    parser.add_argument(
        "--config",
        default=STORAGE_CONFIG_PATH,
        help=f"Path to S3 storage config (default: {STORAGE_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Date suffix for S3 prefix (default: today, YYYY-MM-DD).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be uploaded without uploading.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    upload_date = args.date or datetime.utcnow().strftime("%Y-%m-%d")
    print(f"Upload date: {upload_date}")

    try:
        bucket, prefix, region, endpoint_url = _load_storage_config(args.config)
    except FileNotFoundError as exc:
        raise SystemExit(f"Config not found: {args.config}") from exc

    print(f"Target: s3://{bucket}/{prefix}/vlm/{upload_date}/")
    if args.dry_run:
        print("[DRY-RUN MODE - no files will be uploaded]")

    session = UploadSession(
        bucket=bucket,
        prefix=prefix,
        upload_date=upload_date,
        dry_run=args.dry_run,
    )
    session.run()


if __name__ == "__main__":
    main()
