"""Local processing ledger — CSV-based record of source video download/generate/upload state."""

from __future__ import annotations

import csv
import logging
import shutil
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

logger = logging.getLogger(__name__)

_LOCAL_LEDGER_HEADER = [
    "file_name",
    "downloaded",
    "generated",
    "uploaded",
    "local_source_path",
    "source_etag",
    "source_size_bytes",
    "last_error",
]

_LEGACY_HEADER = ["file_name", "processed"]


@dataclass
class LocalLedgerEntry:
    file_name: str
    downloaded: bool = False
    generated: bool = False
    uploaded: bool = False
    local_source_path: str = ""
    source_etag: str = ""
    source_size_bytes: str = ""
    last_error: str = ""


class LocalProcessingLedger:
    """Manages local_processing.csv on local disk."""

    def __init__(self, ledger_path: Path) -> None:
        self.ledger_path = ledger_path
        self.entries: dict[str, LocalLedgerEntry] = {}

    # ------------------------------------------------------------------
    # Read / Write
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load existing ledger from disk. Creates empty ledger if not found."""
        self.entries.clear()
        if not self.ledger_path.exists():
            logger.info(
                "No existing local ledger at %s — will create on next save", self.ledger_path
            )
            return

        text = self.ledger_path.read_text(encoding="utf-8").strip()
        if not text:
            return

        reader = csv.DictReader(StringIO(text))
        headers = reader.fieldnames or []

        is_legacy = _is_legacy_header(headers)

        for row in reader:
            fn = row.get("file_name", "").strip()
            if not fn:
                continue

            entry = _migrate_legacy_row(row, fn) if is_legacy else _parse_row(row, fn)
            self.entries[fn] = entry

        logger.info("Local ledger loaded: %d entries", len(self.entries))

    def save(self) -> None:
        """Persist ledger to disk atomically."""
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger_path.with_suffix(".tmp")
        buf = StringIO()
        writer = csv.DictWriter(buf, fieldnames=_LOCAL_LEDGER_HEADER, lineterminator="\n")
        writer.writeheader()
        for entry in sorted(self.entries.values(), key=lambda e: e.file_name):
            writer.writerow(
                {
                    "file_name": entry.file_name,
                    "downloaded": str(entry.downloaded).lower(),
                    "generated": str(entry.generated).lower(),
                    "uploaded": str(entry.uploaded).lower(),
                    "local_source_path": entry.local_source_path,
                    "source_etag": entry.source_etag,
                    "source_size_bytes": entry.source_size_bytes,
                    "last_error": entry.last_error,
                }
            )
        tmp.write_text(buf.getvalue(), encoding="utf-8")
        shutil.move(str(tmp), str(self.ledger_path))
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        logger.info("Local ledger saved: %d entries", len(self.entries))

    # ------------------------------------------------------------------
    # Sync & selection
    # ------------------------------------------------------------------

    def sync_with_discovery(self, discovered: list[str]) -> None:
        """Add newly discovered video keys. Preserve existing flags."""
        before = len(self.entries)
        for rel_key in discovered:
            if rel_key not in self.entries:
                self.entries[rel_key] = LocalLedgerEntry(file_name=rel_key)
        added = len(self.entries) - before
        if added:
            logger.info("Local ledger sync: added %d new entries", added)

    def select_not_downloaded(self, target_count: int) -> list[LocalLedgerEntry]:
        """Return up to target_count not-downloaded entries in sorted order."""
        candidates = sorted(
            [e for e in self.entries.values() if not e.downloaded],
            key=lambda e: e.file_name,
        )
        return candidates[:target_count]

    def select_ready_for_generation(self, target_count: int) -> list[LocalLedgerEntry]:
        """Return entries where downloaded=true, generated=false."""
        candidates = sorted(
            [e for e in self.entries.values() if e.downloaded and not e.generated],
            key=lambda e: e.file_name,
        )
        return candidates[:target_count]

    def select_ready_for_upload(self, target_count: int) -> list[LocalLedgerEntry]:
        """Return entries where generated=true, uploaded=false."""
        candidates = sorted(
            [e for e in self.entries.values() if e.generated and not e.uploaded],
            key=lambda e: e.file_name,
        )
        return candidates[:target_count]

    def mark_downloaded(
        self,
        file_name: str,
        local_source_path: str = "",
        source_etag: str = "",
        source_size_bytes: str = "",
    ) -> None:
        """Mark a source as downloaded=true."""
        entry = self.entries.get(file_name)
        if entry:
            entry.downloaded = True
            entry.local_source_path = local_source_path
            entry.source_etag = source_etag
            entry.source_size_bytes = source_size_bytes
            entry.last_error = ""
            logger.info("Local ledger: marked %s as downloaded", file_name)

    def mark_generated(self, file_name: str) -> None:
        """Mark a source as generated=true."""
        entry = self.entries.get(file_name)
        if entry:
            assert entry.downloaded, f"Cannot mark {file_name} generated without downloaded"
            entry.generated = True
            entry.last_error = ""
            logger.info("Local ledger: marked %s as generated", file_name)

    def mark_uploaded(self, file_name: str) -> None:
        """Mark a source as uploaded=true."""
        entry = self.entries.get(file_name)
        if entry:
            assert entry.generated, f"Cannot mark {file_name} uploaded without generated"
            entry.uploaded = True
            entry.last_error = ""
            logger.info("Local ledger: marked %s as uploaded", file_name)

    def set_error(self, file_name: str, error: str) -> None:
        """Record a concise failure reason."""
        entry = self.entries.get(file_name)
        if entry:
            entry.last_error = error[:500]

    def get_entry(self, file_name: str) -> LocalLedgerEntry | None:
        return self.entries.get(file_name)

    @property
    def downloaded_count(self) -> int:
        return sum(1 for e in self.entries.values() if e.downloaded)

    @property
    def not_downloaded_count(self) -> int:
        return sum(1 for e in self.entries.values() if not e.downloaded)

    @property
    def generated_count(self) -> int:
        return sum(1 for e in self.entries.values() if e.generated)

    @property
    def uploaded_count(self) -> int:
        return sum(1 for e in self.entries.values() if e.uploaded)

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def reconcile_with_disk(
        self,
        local_source_dir: Path,
        current_s3_info: dict[str, dict[str, str]] | None = None,
        refresh_changed: bool = False,
    ) -> list[str]:
        """Reconcile ledger state with actual disk files.

        Returns list of warnings produced.
        """
        warnings: list[str] = []

        for fn, entry in list(self.entries.items()):
            local_path = local_source_dir / fn

            if entry.downloaded:
                if entry.local_source_path:
                    local_path = Path(entry.local_source_path)

                if not local_path.exists() or not local_path.is_file():
                    entry.downloaded = False
                    w = f"Reconcile: {fn} marked downloaded but file missing at {local_path}"
                    warnings.append(w)
                    logger.warning(w)
                    continue

                if current_s3_info and fn in current_s3_info:
                    s3_info = current_s3_info[fn]
                    s3_etag = s3_info.get("etag", "")
                    s3_size = s3_info.get("size", "")
                    if s3_etag and entry.source_etag and s3_etag != entry.source_etag:
                        w = f"Reconcile: {fn} source changed in S3 (ETag {entry.source_etag} -> {s3_etag})"
                        warnings.append(w)
                        logger.warning(w)
                        if not refresh_changed:
                            entry.downloaded = False
                            entry.last_error = "source_changed"
                    elif (
                        s3_size and entry.source_size_bytes and s3_size != entry.source_size_bytes
                    ):
                        w = f"Reconcile: {fn} source changed in S3 (size {entry.source_size_bytes} -> {s3_size})"
                        warnings.append(w)
                        logger.warning(w)
                        if not refresh_changed:
                            entry.downloaded = False
                            entry.last_error = "source_changed"

            elif not entry.downloaded and entry.local_source_path:
                local_p = Path(entry.local_source_path)
                if (
                    local_p.exists()
                    and local_p.is_file()
                    and current_s3_info
                    and fn in current_s3_info
                ):
                    s3_info = current_s3_info[fn]
                    local_size = str(local_p.stat().st_size)
                    if s3_info.get("size") == local_size:
                        entry.downloaded = True
                        entry.source_etag = s3_info.get("etag", entry.source_etag)
                        entry.source_size_bytes = s3_info.get("size", entry.source_size_bytes)
                        logger.info("Reconcile: adopted existing file for %s", fn)

        return warnings


def _is_legacy_header(headers: list[str | None]) -> bool:
    cleaned = [h.strip() if h else "" for h in headers]
    return cleaned == _LEGACY_HEADER or (len(cleaned) == 2 and "processed" in cleaned)


def _migrate_legacy_row(row: dict[str, str], fn: str) -> LocalLedgerEntry:
    proc_str = row.get("processed", "false").strip().lower()
    processed = proc_str in ("true", "1", "yes")
    generated = processed
    uploaded = processed
    return LocalLedgerEntry(
        file_name=fn,
        downloaded=False,
        generated=generated,
        uploaded=uploaded,
    )


def _parse_row(row: dict[str, str], fn: str) -> LocalLedgerEntry:
    def _bool(val: str, default: bool = False) -> bool:
        v = val.strip().lower()
        if v in ("true", "1", "yes"):
            return True
        if v in ("false", "0", "no", ""):
            return False
        return default

    return LocalLedgerEntry(
        file_name=fn,
        downloaded=_bool(row.get("downloaded", "false")),
        generated=_bool(row.get("generated", "false")),
        uploaded=_bool(row.get("uploaded", "false")),
        local_source_path=row.get("local_source_path", "").strip(),
        source_etag=row.get("source_etag", "").strip(),
        source_size_bytes=row.get("source_size_bytes", "").strip(),
        last_error=row.get("last_error", "").strip(),
    )
