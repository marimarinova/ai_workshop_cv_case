#!/usr/bin/env python3

import argparse
import csv
import logging
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

BUCKET = "chillnbite-cameras"
PREFIX = "anon/"
OUT_DIR = Path(".local/source_videos")
LEDGER_PATH = Path(".local/candidate_staging/local_processing.csv")

MIN_SIZE_MB = 50


def _update_ledger(rel_key: str, local_path: str) -> None:
    """Mark a video as downloaded in the local processing ledger."""
    entries: dict[str, dict[str, str]] = {}
    if LEDGER_PATH.exists():
        text = LEDGER_PATH.read_text(encoding="utf-8").strip()
        if text:
            reader = csv.DictReader(LEDGER_PATH.open())
            for row in reader:
                fn = row.get("file_name", "").strip()
                if fn:
                    entries[fn] = dict(row)

    if rel_key not in entries:
        entries[rel_key] = {
            "file_name": rel_key,
            "downloaded": "true",
            "generated": "false",
            "uploaded": "false",
            "local_source_path": local_path,
            "source_etag": "",
            "source_size_bytes": "",
            "last_error": "",
        }
    else:
        entries[rel_key]["downloaded"] = "true"
        entries[rel_key]["local_source_path"] = local_path
        entries[rel_key]["last_error"] = ""

    header = [
        "file_name",
        "downloaded",
        "generated",
        "uploaded",
        "local_source_path",
        "source_etag",
        "source_size_bytes",
        "last_error",
    ]
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER_PATH.with_suffix(".tmp")
    buf: list[str] = []
    buf.append(",".join(header))
    for fn in sorted(entries):
        e = entries[fn]
        buf.append(",".join(e.get(h, "") for h in header))
    tmp.write_text("\n".join(buf) + "\n", encoding="utf-8")
    tmp.rename(LEDGER_PATH)
    logger.info("Ledger updated: %s -> downloaded", rel_key)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download source videos from S3 in deterministic batches."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of clips to download in this batch (default: 10).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many clips from the sorted list (default: 0).",
    )
    parser.add_argument(
        "--min-size-mb",
        type=float,
        default=MIN_SIZE_MB,
        help=f"Minimum clip size in MB (default: {MIN_SIZE_MB}).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all .mp4 files regardless of size.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    clips: list[tuple[str, int]] = []

    print(f"Searching s3://{BUCKET}/{PREFIX}")

    try:
        for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                size_bytes = obj["Size"]
                size_mb = size_bytes / (1024 * 1024)

                if key.lower().endswith(".mp4") and (args.all or size_mb >= args.min_size_mb):
                    clips.append((key, size_bytes))
    except (BotoCoreError, ClientError) as exc:
        raise SystemExit(f"Could not list S3 objects: {exc}") from exc

    clips.sort(key=lambda c: c[0])
    size_label = "all sizes" if args.all else f">= {args.min_size_mb} MB"
    print(f"Found {len(clips)} clips ({size_label})")

    if not clips:
        raise SystemExit("No matching clips found. Use --all or lower --min-size-mb.")

    already_downloaded = {
        p.name for p in OUT_DIR.rglob("*.mp4") if p.is_file() and not str(p).endswith(".part")
    }
    available = [
        (key, size_bytes) for key, size_bytes in clips if Path(key).name not in already_downloaded
    ]

    if not available:
        print("All matching clips already downloaded.")
        raise SystemExit(0)

    batch = available[args.offset : args.offset + args.count]

    if not batch:
        print(
            f"No clips at offset {args.offset}. Available: {len(available)}, total: {len(clips)}."
        )
        raise SystemExit(1)

    print(f"\nSelected {len(batch)} clip(s) for download (offset={args.offset}):")
    for key, size_bytes in batch:
        print(f"- {size_bytes / (1024 * 1024):.1f} MB | {key}")

    print("\nDownloading...")

    downloaded = 0
    failed = 0
    for index, (key, size_bytes) in enumerate(batch, start=1):
        filename = Path(key).name
        out_path = OUT_DIR / filename
        temporary_path = out_path.with_suffix(out_path.suffix + ".part")

        rel_key = key.removeprefix(PREFIX)

        print(
            f"[{index}/{len(batch)}] Downloading {filename} ({size_bytes / (1024 * 1024):.1f} MB)"
        )

        temporary_path.unlink(missing_ok=True)

        try:
            s3.download_file(BUCKET, key, str(temporary_path))
            temporary_path.replace(out_path)
            _update_ledger(rel_key, str(out_path))
            downloaded += 1
            print(f"Saved: {out_path}")
        except (BotoCoreError, ClientError, OSError) as exc:
            temporary_path.unlink(missing_ok=True)
            print(f"Failed to download {key}: {exc}")
            failed += 1
            continue

    print(f"\nDone. {downloaded} downloaded, {failed} failed. Files in: {OUT_DIR}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
