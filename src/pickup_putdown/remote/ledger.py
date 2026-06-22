"""Processing ledger — CSV-based record of which source videos have been processed."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LEDGER_HEADER = ["file_name", "processed"]


@dataclass
class LedgerEntry:
    file_name: str
    processed: bool = False


class ProcessingLedger:
    """Manages process_for_candidates.csv on S3 via S3Storage."""

    def __init__(self, storage: Any, ledger_key: str = "process_for_candidates.csv") -> None:
        self.storage = storage
        self.ledger_key = ledger_key
        self.entries: dict[str, LedgerEntry] = {}

    # ------------------------------------------------------------------
    # Read / Write
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load existing ledger from S3. Creates empty ledger if not found."""
        try:
            tmp = Path("/tmp/_ledger_tmp.csv")
            self.storage.download(self.storage.full_key(self.ledger_key), tmp)
            text = tmp.read_text(encoding="utf-8")
            tmp.unlink(missing_ok=True)
        except Exception:
            logger.info("No existing ledger at %s — will create on next save", self.ledger_key)
            text = ""

        self.entries.clear()
        if text.strip():
            reader = csv.DictReader(StringIO(text))
            for row in reader:
                fn = row.get("file_name", "").strip()
                if fn:
                    proc_str = row.get("processed", "false").strip().lower()
                    self.entries[fn] = LedgerEntry(
                        file_name=fn,
                        processed=proc_str in ("true", "1", "yes"),
                    )
        logger.info("Ledger loaded: %d entries", len(self.entries))

    def save(self) -> None:
        """Persist ledger to S3."""
        buf = StringIO()
        writer = csv.DictWriter(buf, fieldnames=_LEDGER_HEADER, lineterminator="\n")
        writer.writeheader()
        for entry in sorted(self.entries.values(), key=lambda e: e.file_name):
            writer.writerow(
                {"file_name": entry.file_name, "processed": str(entry.processed).lower()}
            )
        content = buf.getvalue()
        tmp = Path("/tmp/_ledger_tmp.csv")
        tmp.write_text(content, encoding="utf-8")
        self.storage.upload(tmp, self.storage.full_key(self.ledger_key))
        tmp.unlink(missing_ok=True)
        logger.info("Ledger saved: %d entries", len(self.entries))

    # ------------------------------------------------------------------
    # Sync & selection
    # ------------------------------------------------------------------

    def sync_with_discovery(self, discovered: list[str]) -> None:
        """Add newly discovered video keys with processed=false. Preserve existing flags."""
        before = len(self.entries)
        for rel_key in discovered:
            if rel_key not in self.entries:
                self.entries[rel_key] = LedgerEntry(file_name=rel_key, processed=False)
        added = len(self.entries) - before
        if added:
            logger.info("Ledger sync: added %d new entries", added)

    def select_unprocessed(self, target_count: int) -> list[LedgerEntry]:
        """Return up to target_count unprocessed entries in sorted file_name order."""
        unprocessed = sorted(
            [e for e in self.entries.values() if not e.processed],
            key=lambda e: e.file_name,
        )
        return unprocessed[:target_count]

    def mark_processed(self, file_name: str) -> None:
        """Mark a source as processed=true."""
        if file_name in self.entries:
            self.entries[file_name].processed = True
            logger.info("Ledger: marked %s as processed", file_name)

    def get_entry(self, file_name: str) -> LedgerEntry | None:
        return self.entries.get(file_name)

    @property
    def processed_count(self) -> int:
        return sum(1 for e in self.entries.values() if e.processed)

    @property
    def unprocessed_count(self) -> int:
        return sum(1 for e in self.entries.values() if not e.processed)
