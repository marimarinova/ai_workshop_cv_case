"""S3 storage helper for authenticated read/write operations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import boto3

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def _parse_bucket_uri(bucket_uri: str) -> tuple[str, str]:
    """Parse s3://bucket/prefix into (bucket, prefix)."""
    if not bucket_uri.startswith("s3://"):
        raise ValueError(f"Invalid bucket URI: {bucket_uri}")
    rest = bucket_uri[5:]
    slash = rest.index("/")
    return rest[:slash], rest[slash + 1 :].rstrip("/")


def _build_client_kwargs(
    endpoint_url: str | None,
    region: str | None,
    anonymous: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    if region:
        kwargs["region_name"] = region
    if anonymous:
        kwargs["aws_access_key_id"] = ""
        kwargs["aws_secret_access_key"] = ""
        kwargs["aws_session_token"] = ""
    return kwargs


class S3Storage:
    """Thin wrapper around boto3 S3 client for candidate generation workflow."""

    def __init__(
        self,
        bucket_uri: str,
        *,
        endpoint_url: str | None = None,
        region: str | None = None,
        anonymous: bool = False,
    ) -> None:
        self.bucket_uri = bucket_uri
        self.endpoint_url = endpoint_url
        self.region = region
        self.anonymous = anonymous
        self.bucket, self.prefix = _parse_bucket_uri(bucket_uri)
        self._client = boto3.client(
            "s3",
            **_build_client_kwargs(endpoint_url, region, anonymous),
        )

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_objects(self, prefix: str | None = None) -> list[dict[str, Any]]:
        """List objects under prefix. Returns list of dicts with key, size, etag."""
        full = f"{self.prefix}/{prefix}" if prefix else self.prefix
        if full.endswith("/"):
            full = full.rstrip("/")
        objects: list[dict[str, Any]] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=self.bucket, Prefix=full, PaginationConfig={"PageSize": 1000}
        ):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith("/"):
                    continue
                objects.append(
                    {
                        "key": obj["Key"],
                        "size": obj["Size"],
                        "etag": obj.get("ETag", "").strip('"'),
                    }
                )
        return objects

    # ------------------------------------------------------------------
    # Download / Upload
    # ------------------------------------------------------------------

    def download(self, key: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self.bucket, key, str(local_path))

    def upload(self, local_path: Path, key: str) -> None:
        self._client.upload_file(str(local_path), self.bucket, key)

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    def relative_key(self, full_key: str) -> str:
        """Return key relative to the configured prefix (anon/)."""
        expected = self.prefix
        if expected and not expected.endswith("/"):
            expected += "/"
        if full_key.startswith(expected):
            return full_key[len(expected) :]
        return full_key

    def full_key(self, relative: str) -> str:
        """Return full S3 key from a prefix-relative path."""
        prefix = self.prefix
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        return f"{prefix}{relative}"

    @staticmethod
    def is_video(key: str) -> bool:
        return Path(key).suffix.lower() in _VIDEO_EXTENSIONS

    @staticmethod
    def is_excluded(relative_key: str) -> bool:
        """Check if the key falls under excluded paths."""
        parts = relative_key.lstrip("/").split("/", 1)
        if parts[0] == "candidates":
            return True
        return relative_key == "process_for_candidates.csv"
