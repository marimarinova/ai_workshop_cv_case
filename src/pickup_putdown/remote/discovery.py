"""Source video discovery under S3 prefix with exclusion rules."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def discover_source_videos(storage: object) -> list[str]:
    """List supported source video objects under the configured prefix.

    Excludes:
    - anything under candidates/
    - process_for_candidates.csv

    Returns relative keys (relative to the bucket prefix, e.g. anon/).
    """
    all_objects = storage.list_objects()
    discovered: list[str] = []
    for obj in all_objects:
        full_key = obj["key"]
        rel_key = storage.relative_key(full_key)
        if storage.is_excluded(rel_key):
            continue
        if not storage.is_video(rel_key):
            continue
        discovered.append(rel_key)
    discovered.sort()
    logger.info("Discovered %d source video(s) under %s", len(discovered), storage.prefix)
    return discovered
