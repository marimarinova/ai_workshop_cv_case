"""Manifest I/O for Track A feature datasets.

This module handles:
- Saving FeatureDataset to parquet files
- Loading FeatureDataset from parquet files
- Appending new records to existing manifests
- Filtering and querying manifests
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    pass

from pickup_putdown.layer1.track_a.contracts import (
    CropGeometry,
    FeatureDataset,
    FeatureRecord,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

MANIFEST_SCHEMA = pa.schema(
    [
        ("crop_id", pa.string()),
        ("clip_id", pa.string()),
        ("candidate_id", pa.string()),
        ("timestamp_s", pa.float64()),
        ("sample_position", pa.string()),
        ("crop_type", pa.string()),
        # Geometry fields (flattened)
        ("geometry_x", pa.int32()),
        ("geometry_y", pa.int32()),
        ("geometry_width", pa.int32()),
        ("geometry_height", pa.int32()),
        # Embedding info
        ("embedding_path", pa.string()),
        ("encoder_name", pa.string()),
        ("encoder_version", pa.string()),
        # Label and split
        ("label", pa.string()),
        ("split", pa.string()),
        # Optional metadata
        ("actor_id", pa.string()),
        ("hand_side", pa.string()),
        ("region_id", pa.string()),
        ("confidence", pa.string()),
        ("hard_case", pa.bool_()),
        ("event_id", pa.string()),
        # Tracking metadata
        ("created_at", pa.timestamp("us")),
        ("batch_id", pa.string()),
    ]
)


# ---------------------------------------------------------------------------
# Save functions
# ---------------------------------------------------------------------------


def save_manifest(
    dataset: FeatureDataset,
    output_path: Path | str,
    batch_id: str | None = None,
) -> Path:
    """Save a FeatureDataset to a parquet manifest file.

    Args:
        dataset: The feature dataset to save.
        output_path: Path to the output parquet file.
        batch_id: Optional batch identifier for tracking.

    Returns:
        Path where the manifest was saved.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert records to table
    table = _records_to_table(dataset.records, batch_id)

    # Write parquet
    pq.write_table(table, output_path)

    logger.info(f"Saved manifest with {len(dataset.records)} records to {output_path}")
    return output_path


def append_to_manifest(
    records: list[FeatureRecord],
    manifest_path: Path | str,
    batch_id: str | None = None,
) -> int:
    """Append new records to an existing manifest.

    Args:
        records: New records to append.
        manifest_path: Path to the existing manifest.
        batch_id: Optional batch identifier for the new records.

    Returns:
        Total number of records after appending.
    """
    manifest_path = Path(manifest_path)

    # Load existing records
    if manifest_path.exists():
        existing_table = pq.read_table(manifest_path)
        existing_ids = set(existing_table.column("crop_id").to_pylist())
    else:
        existing_table = None
        existing_ids = set()

    # Filter out duplicates
    new_records = [r for r in records if r.crop_id not in existing_ids]

    if not new_records:
        logger.info("No new records to append (all duplicates)")
        return len(existing_ids)

    # Convert new records to table
    new_table = _records_to_table(new_records, batch_id)

    # Combine tables
    if existing_table is not None:
        combined_table = pa.concat_tables([existing_table, new_table])
    else:
        combined_table = new_table

    # Write back
    pq.write_table(combined_table, manifest_path)

    total = combined_table.num_rows
    logger.info(f"Appended {len(new_records)} records to manifest (total: {total})")
    return total


def _records_to_table(
    records: list[FeatureRecord],
    batch_id: str | None = None,
) -> pa.Table:
    """Convert FeatureRecords to a PyArrow table."""
    now = datetime.now()

    data = {
        "crop_id": [],
        "clip_id": [],
        "candidate_id": [],
        "timestamp_s": [],
        "sample_position": [],
        "crop_type": [],
        "geometry_x": [],
        "geometry_y": [],
        "geometry_width": [],
        "geometry_height": [],
        "embedding_path": [],
        "encoder_name": [],
        "encoder_version": [],
        "label": [],
        "split": [],
        "actor_id": [],
        "hand_side": [],
        "region_id": [],
        "confidence": [],
        "hard_case": [],
        "event_id": [],
        "created_at": [],
        "batch_id": [],
    }

    for record in records:
        data["crop_id"].append(record.crop_id)
        data["clip_id"].append(record.clip_id)
        data["candidate_id"].append(record.candidate_id)
        data["timestamp_s"].append(record.timestamp_s)
        data["sample_position"].append(record.sample_position)
        data["crop_type"].append(record.crop_type)
        data["geometry_x"].append(record.geometry.x)
        data["geometry_y"].append(record.geometry.y)
        data["geometry_width"].append(record.geometry.width)
        data["geometry_height"].append(record.geometry.height)
        data["embedding_path"].append(str(record.embedding_path))
        data["encoder_name"].append(record.encoder_name)
        data["encoder_version"].append(record.encoder_version)
        data["label"].append(record.label)
        data["split"].append(record.split)
        data["actor_id"].append(record.actor_id)
        data["hand_side"].append(record.hand_side)
        data["region_id"].append(record.region_id)
        data["confidence"].append(record.confidence)
        data["hard_case"].append(record.hard_case)
        data["event_id"].append(record.event_id)
        data["created_at"].append(now)
        data["batch_id"].append(batch_id)

    return pa.Table.from_pydict(data, schema=MANIFEST_SCHEMA)


# ---------------------------------------------------------------------------
# Load functions
# ---------------------------------------------------------------------------


def load_manifest(
    manifest_path: Path | str,
) -> FeatureDataset:
    """Load a FeatureDataset from a parquet manifest.

    Args:
        manifest_path: Path to the manifest parquet file.

    Returns:
        FeatureDataset with loaded records.
    """
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    table = pq.read_table(manifest_path)
    records = _table_to_records(table)

    # Extract encoder info from first record
    encoder_name = ""
    encoder_version = ""
    if records:
        encoder_name = records[0].encoder_name
        encoder_version = records[0].encoder_version

    dataset = FeatureDataset(
        records=records,
        encoder_name=encoder_name,
        encoder_version=encoder_version,
    )
    dataset.compute_stats()

    logger.info(f"Loaded manifest with {len(records)} records from {manifest_path}")
    return dataset


def _table_to_records(table: pa.Table) -> list[FeatureRecord]:
    """Convert a PyArrow table to FeatureRecords."""
    records = []

    for i in range(table.num_rows):
        geometry = CropGeometry(
            x=table.column("geometry_x")[i].as_py(),
            y=table.column("geometry_y")[i].as_py(),
            width=table.column("geometry_width")[i].as_py(),
            height=table.column("geometry_height")[i].as_py(),
        )

        record = FeatureRecord(
            crop_id=table.column("crop_id")[i].as_py(),
            clip_id=table.column("clip_id")[i].as_py(),
            candidate_id=table.column("candidate_id")[i].as_py(),
            timestamp_s=table.column("timestamp_s")[i].as_py(),
            sample_position=table.column("sample_position")[i].as_py(),
            crop_type=table.column("crop_type")[i].as_py(),
            geometry=geometry,
            embedding_path=Path(table.column("embedding_path")[i].as_py()),
            encoder_name=table.column("encoder_name")[i].as_py(),
            encoder_version=table.column("encoder_version")[i].as_py(),
            label=table.column("label")[i].as_py(),
            split=table.column("split")[i].as_py(),
            actor_id=table.column("actor_id")[i].as_py(),
            hand_side=table.column("hand_side")[i].as_py(),
            region_id=table.column("region_id")[i].as_py(),
            confidence=table.column("confidence")[i].as_py(),
            hard_case=table.column("hard_case")[i].as_py() or False,
            event_id=table.column("event_id")[i].as_py(),
        )
        records.append(record)

    return records


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------


def filter_manifest(
    manifest_path: Path | str,
    split: str | None = None,
    label: str | None = None,
    batch_id: str | None = None,
    clip_ids: list[str] | None = None,
) -> FeatureDataset:
    """Load and filter a manifest by various criteria.

    Args:
        manifest_path: Path to the manifest parquet file.
        split: Filter by split ("train", "val", "test").
        label: Filter by label ("pickup", "putdown", "negative").
        batch_id: Filter by batch ID.
        clip_ids: Filter by list of clip IDs.

    Returns:
        FeatureDataset with filtered records.
    """
    manifest_path = Path(manifest_path)

    # Build filter expressions
    filters = []
    if split is not None:
        filters.append(("split", "=", split))
    if label is not None:
        filters.append(("label", "=", label))
    if batch_id is not None:
        filters.append(("batch_id", "=", batch_id))

    # Read with row group filtering if possible
    if filters:
        table = pq.read_table(manifest_path, filters=filters)
    else:
        table = pq.read_table(manifest_path)

    # Apply clip_ids filter (not supported by parquet filters)
    if clip_ids is not None:
        clip_set = set(clip_ids)
        mask = [table.column("clip_id")[i].as_py() in clip_set for i in range(table.num_rows)]
        table = table.filter(mask)

    records = _table_to_records(table)

    encoder_name = ""
    encoder_version = ""
    if records:
        encoder_name = records[0].encoder_name
        encoder_version = records[0].encoder_version

    dataset = FeatureDataset(
        records=records,
        encoder_name=encoder_name,
        encoder_version=encoder_version,
    )
    dataset.compute_stats()

    return dataset


def get_manifest_stats(manifest_path: Path | str) -> dict:
    """Get statistics about a manifest without loading all records.

    Args:
        manifest_path: Path to the manifest parquet file.

    Returns:
        Dict with manifest statistics.
    """
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        return {"exists": False}

    # Read metadata
    parquet_file = pq.ParquetFile(manifest_path)
    metadata = parquet_file.metadata

    # Read just the columns we need for stats
    table = pq.read_table(
        manifest_path,
        columns=["label", "split", "crop_type", "batch_id", "encoder_name"],
    )

    labels = table.column("label").to_pylist()
    splits = table.column("split").to_pylist()
    crop_types = table.column("crop_type").to_pylist()
    batch_ids = table.column("batch_id").to_pylist()
    encoder_names = table.column("encoder_name").to_pylist()

    return {
        "exists": True,
        "total_records": metadata.num_rows,
        "file_size_mb": round(manifest_path.stat().st_size / (1024 * 1024), 2),
        "labels": {
            "pickup": labels.count("pickup"),
            "putdown": labels.count("putdown"),
            "negative": labels.count("negative"),
        },
        "splits": {
            "train": splits.count("train"),
            "val": splits.count("val"),
            "test": splits.count("test"),
        },
        "crop_types": {
            "hand": crop_types.count("hand"),
            "shelf": crop_types.count("shelf"),
        },
        "unique_batches": len(set(b for b in batch_ids if b is not None)),
        "encoder": encoder_names[0] if encoder_names else None,
    }


def list_batches(manifest_path: Path | str) -> list[dict]:
    """List all batches in a manifest with their record counts.

    Args:
        manifest_path: Path to the manifest parquet file.

    Returns:
        List of dicts with batch_id and count.
    """
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        return []

    table = pq.read_table(manifest_path, columns=["batch_id", "created_at"])
    batch_ids = table.column("batch_id").to_pylist()
    created_ats = table.column("created_at").to_pylist()

    # Group by batch
    batches: dict[str, dict] = {}
    for batch_id, created_at in zip(batch_ids, created_ats):
        key = batch_id or "unknown"
        if key not in batches:
            batches[key] = {"batch_id": batch_id, "count": 0, "created_at": created_at}
        batches[key]["count"] += 1

    return sorted(batches.values(), key=lambda x: x["created_at"] or datetime.min)
