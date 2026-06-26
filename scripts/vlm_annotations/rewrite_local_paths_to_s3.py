#!/usr/bin/env python3
"""Rewrite local candidate video paths to S3 paths in VLM annotation metadata."""

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
class FileTarget:
    """Describes a file to process and how to rewrite its paths."""

    path: Path
    file_type: str  # "json" or "csv"
    field_name: str | None = None  # JSON field to rewrite (None = all string values)
    csv_column: str | None = None  # CSV column name to rewrite

    def contains_old_path(self) -> bool:
        """Check if file contains the old path prefix."""
        try:
            content = self.path.read_text(encoding="utf-8")
            return LOCAL_PATH_PREFIX in content
        except (OSError, UnicodeDecodeError):
            return False


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
    """Handles the actual path replacement logic."""

    def __init__(self, old_prefix: str = LOCAL_PATH_PREFIX, new_prefix: str = S3_PATH_PREFIX) -> None:
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

    def rewrite_csv_file(self, path: Path, column_name: str | None = None) -> int:
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
    def discover(cls) -> list[FileTarget]:
        """Discover all files that may need rewriting."""
        targets: list[FileTarget] = []

        for rel_dir, file_type, field_name in cls.TARGET_DIRS:
            base_dir = LOCAL_ROOT / rel_dir
            if not base_dir.exists():
                logger.warning("Directory not found: %s", base_dir)
                continue

            if file_type == "json":
                targets.extend(cls._discover_json_files(base_dir, field_name))
            elif file_type == "csv":
                targets.extend(cls._discover_csv_files(base_dir))

        return targets

    @classmethod
    def _discover_json_files(cls, base_dir: Path, field_name: str | None) -> list[FileTarget]:
        """Find all JSON files under base_dir."""
        targets: list[FileTarget] = []
        for json_path in sorted(base_dir.rglob("*.json")):
            if json_path.is_symlink():
                logger.debug("Skipping symlink: %s", json_path)
                continue
            targets.append(FileTarget(path=json_path, file_type="json", field_name=field_name))
        return targets

    @classmethod
    def _discover_csv_files(cls, base_dir: Path) -> list[FileTarget]:
        """Find all CSV files under base_dir."""
        targets: list[FileTarget] = []
        for csv_path in sorted(base_dir.rglob("*.csv")):
            if csv_path.is_symlink():
                logger.debug("Skipping symlink: %s", csv_path)
                continue
            targets.append(FileTarget(path=csv_path, file_type="csv"))
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

        for target in targets:
            self._process_target(target)

        print("\n=== Rewrite Summary ===")
        print(self.stats.summary())

        if self.stats.errors > 0:
            raise SystemExit(1)

    def _process_target(self, target: FileTarget) -> None:
        """Process a single file target."""
        self.stats.files_processed += 1

        if not target.contains_old_path():
            self.stats.files_skipped += 1
            return

        if self.dry_run:
            print(f"  [DRY-RUN] Would rewrite: {target.path}")
            self.stats.files_modified += 1
            return

        try:
            if self.backup:
                backup_path = target.path.with_suffix(target.path.suffix + ".bak")
                shutil.copy2(target.path, backup_path)

            if target.file_type == "json":
                count = self.rewriter.rewrite_json_file(target.path, target.field_name)
            else:
                count = self.rewriter.rewrite_csv_file(target.path, target.csv_column)

            if count > 0:
                self.stats.files_modified += 1
                self.stats.replacements += count
                logger.info("Rewrote %d path(s) in %s", count, target.path)
            else:
                self.stats.files_skipped += 1
        except Exception as exc:
            self.stats.errors += 1
            logger.error("Failed to process %s: %s", target.path, exc)
            print(f"  ERROR: {target.path}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite local candidate video paths to S3 paths in VLM annotation metadata."
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

    print(f"Old prefix: {LOCAL_PATH_PREFIX}")
    print(f"New prefix: {S3_PATH_PREFIX}")
    if args.dry_run:
        print("[DRY-RUN MODE - no files will be modified]\n")
    else:
        print(f"Backup: {'disabled' if args.no_backup else 'enabled'}\n")

    session = RewriteSession(dry_run=args.dry_run, backup=not args.no_backup)
    session.run()


if __name__ == "__main__":
    main()
