#!/usr/bin/env python3
"""Rewrite S3 candidate video paths back to local paths in VLM annotation metadata.

Reverse of rewrite_local_paths_to_s3.py. Reuses the PathRewriter class with
direction control so the same code path handles both conversions.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

LOCAL_PATH_PREFIX: Final = ".local/candidate_staging/candidates/"
S3_PATH_PREFIX: Final = "s3://chillnbite-cameras/anon/candidates/videos/"
LOCAL_ROOT: Final = Path(".local")


@dataclass
class RewriteStats:
    """Accumulates rewrite statistics."""

    files_processed: int = 0
    files_modified: int = 0
    files_skipped: int = 0
    replacements: int = 0
    errors: int = 0

    def summary(self) -> str:
        lines = [
            f"  Files scanned:   {self.files_processed}",
            f"  Files modified:  {self.files_modified}",
            f"  Files skipped:   {self.files_skipped}",
            f"  Replacements:    {self.replacements}",
            f"  Errors:          {self.errors}",
        ]
        return "\n".join(lines)


class PathRewriter:
    """Handles bidirectional path replacement logic."""

    def __init__(
        self,
        old_prefix: str = S3_PATH_PREFIX,
        new_prefix: str = LOCAL_PATH_PREFIX,
    ) -> None:
        self.old_prefix = old_prefix
        self.new_prefix = new_prefix

    def rewrite_string(self, value: str) -> tuple[str, int]:
        """Replace old path prefix in a string. Returns (new_value, count)."""
        count = value.count(self.old_prefix)
        if count > 0:
            return value.replace(self.old_prefix, self.new_prefix), count
        return value, 0

    def rewrite_json_file(self, path: Path, field_name: str | None = None) -> int:
        """Rewrite paths in a JSON file. Returns replacement count."""
        data = json.loads(path.read_text(encoding="utf-8"))
        count = self._rewrite_json_value(data, field_name)
        if count > 0:
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return count

    def _rewrite_json_value(self, value: object, target_field: str | None) -> int:
        """Recursively rewrite paths in a JSON value."""
        if isinstance(value, str):
            new_val, count = self.rewrite_string(value)
            return count
        elif isinstance(value, dict):
            total = 0
            for key, val in value.items():
                if target_field and key == target_field:
                    if isinstance(val, str):
                        new_val, count = self.rewrite_string(val)
                        if count > 0:
                            value[key] = new_val
                            total += count
                    elif isinstance(val, list):
                        total += self._rewrite_list(val, None)
                elif not target_field:
                    total += self._rewrite_json_value(val, None)
            return total
        elif isinstance(value, list):
            return self._rewrite_list(value, target_field)
        return 0

    def _rewrite_list(self, lst: list[object], target_field: str | None) -> int:
        """Rewrite paths in a list of values."""
        total = 0
        for item in lst:
            total += self._rewrite_json_value(item, target_field)
        return total

    def rewrite_csv_file(self, path: Path) -> int:
        """Rewrite paths in a CSV file. Returns replacement count."""
        text = path.read_text(encoding="utf-8")
        new_text, count = self.rewrite_string(text)
        if count > 0:
            path.write_text(new_text, encoding="utf-8")
        return count


class FileDiscovery:
    """Discovers files that need path rewriting."""

    TARGET_DIRS: Final = [
        ("vlm_annotations/normalized", "json", "video_path"),
        ("vlm_annotations/raw", "json", "video_path"),
        ("task_7_review", "csv", None),
        ("task_7_vlm", "csv", None),
        ("candidate_staging/candidates", "json", "candidate_key"),
    ]

    @classmethod
    def discover(cls) -> list[tuple[Path, str, str | None]]:
        """Discover all files that may need rewriting."""
        targets: list[tuple[Path, str, str | None]] = []
        for rel_dir, file_type, field_name in cls.TARGET_DIRS:
            base_dir = LOCAL_ROOT / rel_dir
            if not base_dir.exists():
                logger.warning("Directory not found: %s", base_dir)
                continue
            if file_type == "json":
                for p in sorted(base_dir.rglob("*.json")):
                    if not p.is_symlink():
                        targets.append((p, "json", field_name))
            elif file_type == "csv":
                for p in sorted(base_dir.rglob("*.csv")):
                    if not p.is_symlink():
                        targets.append((p, "csv", None))
        return targets


class RewriteSession:
    """Orchestrates the path rewriting process."""

    def __init__(self, dry_run: bool = False, backup: bool = True) -> None:
        self.dry_run = dry_run
        self.backup = backup
        self.rewriter = PathRewriter()
        self.stats = RewriteStats()

    def run(self) -> None:
        """Execute the rewrite process."""
        targets = FileDiscovery.discover()
        if not targets:
            print("No target files found.")
            return

        print(f"Discovered {len(targets)} file(s) to scan.\n")
        for path, file_type, field_name in targets:
            self._process_target(path, file_type, field_name)

        print("\n=== Rewrite Summary ===")
        print(self.stats.summary())
        if self.stats.errors > 0:
            raise SystemExit(1)

    def _process_target(self, path: Path, file_type: str, field_name: str | None) -> None:
        """Process a single file target."""
        self.stats.files_processed += 1
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            self.stats.files_skipped += 1
            return

        if self.rewriter.old_prefix not in content:
            self.stats.files_skipped += 1
            return

        if self.dry_run:
            print(f"  [DRY-RUN] Would rewrite: {path}")
            self.stats.files_modified += 1
            return

        try:
            if self.backup:
                backup_path = path.with_suffix(path.suffix + ".bak")
                shutil.copy2(path, backup_path)

            if file_type == "json":
                count = self.rewriter.rewrite_json_file(path, field_name)
            else:
                count = self.rewriter.rewrite_csv_file(path)

            if count > 0:
                self.stats.files_modified += 1
                self.stats.replacements += count
                logger.info("Rewrote %d path(s) in %s", count, path)
            else:
                self.stats.files_skipped += 1
        except Exception as exc:
            self.stats.errors += 1
            logger.error("Failed to process %s: %s", path, exc)
            print(f"  ERROR: {path}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite S3 candidate video paths back to local paths in VLM annotation metadata."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating .bak backup files.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    print(f"Old prefix: {S3_PATH_PREFIX}")
    print(f"New prefix: {LOCAL_PATH_PREFIX}")
    if args.dry_run:
        print("[DRY-RUN MODE - no files will be modified]\n")
    else:
        print(f"Backup: {'disabled' if args.no_backup else 'enabled'}\n")

    session = RewriteSession(dry_run=args.dry_run, backup=not args.no_backup)
    session.run()


if __name__ == "__main__":
    main()
