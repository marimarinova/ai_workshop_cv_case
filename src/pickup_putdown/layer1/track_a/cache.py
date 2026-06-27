"""Caching layer for Track A crops and embeddings.

This module handles:
- Computing cache keys from video/timestamp/geometry/encoder
- Saving and loading crop images
- Saving and loading embedding vectors
- Cache invalidation via video checksums and encoder versions
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import cv2
import numpy as np

from pickup_putdown.layer1.track_a.contracts import CropGeometry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache key computation
# ---------------------------------------------------------------------------


def get_video_checksum(video_path: Path | str, quick: bool = True) -> str:
    """Compute a checksum for a video file.

    Args:
        video_path: Path to the video file.
        quick: If True, only hash first 1MB + file size (faster).
                If False, hash entire file (slower but more accurate).

    Returns:
        Hex string checksum.
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    file_size = video_path.stat().st_size
    hasher = hashlib.sha256()

    if quick:
        # Quick mode: hash first 1MB + file size
        with open(video_path, "rb") as f:
            chunk = f.read(1024 * 1024)  # 1MB
            hasher.update(chunk)
        hasher.update(str(file_size).encode())
    else:
        # Full mode: hash entire file
        with open(video_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)

    return hasher.hexdigest()[:16]


def compute_crop_cache_key(
    video_checksum: str,
    timestamp_s: float,
    geometry: CropGeometry,
    crop_type: str,
) -> str:
    """Compute a cache key for a crop image.

    Args:
        video_checksum: Checksum of the source video.
        timestamp_s: Timestamp in seconds.
        geometry: Crop geometry (x, y, width, height).
        crop_type: "hand" or "shelf".

    Returns:
        Cache key string.
    """
    # Convert timestamp to microseconds for precision
    timestamp_us = int(timestamp_s * 1_000_000)

    key_parts = [
        video_checksum,
        str(timestamp_us),
        str(geometry.x),
        str(geometry.y),
        str(geometry.width),
        str(geometry.height),
        crop_type,
    ]

    return "_".join(key_parts)


def compute_embedding_cache_key(
    crop_cache_key: str,
    encoder_name: str,
    encoder_version: str,
) -> str:
    """Compute a cache key for an embedding.

    Args:
        crop_cache_key: Cache key of the source crop.
        encoder_name: Name of the encoder model.
        encoder_version: Version string of the encoder.

    Returns:
        Cache key string.
    """
    return f"{crop_cache_key}_{encoder_name}_{encoder_version}"


# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------


def get_crop_cache_path(
    cache_dir: Path | str,
    crop_cache_key: str,
    extension: str = ".jpg",
) -> Path:
    """Get the file path for a cached crop.

    Args:
        cache_dir: Base cache directory.
        crop_cache_key: Cache key for the crop.
        extension: File extension (.jpg, .png).

    Returns:
        Full path to the crop file.
    """
    cache_dir = Path(cache_dir)
    # Use first 2 chars of key as subdirectory to avoid too many files in one dir
    subdir = crop_cache_key[:2]
    return cache_dir / "crops" / subdir / f"{crop_cache_key}{extension}"


def get_embedding_cache_path(
    cache_dir: Path | str,
    embedding_cache_key: str,
) -> Path:
    """Get the file path for a cached embedding.

    Args:
        cache_dir: Base cache directory.
        embedding_cache_key: Cache key for the embedding.

    Returns:
        Full path to the embedding file (.npy).
    """
    cache_dir = Path(cache_dir)
    # Use first 2 chars of key as subdirectory
    subdir = embedding_cache_key[:2]
    return cache_dir / "embeddings" / subdir / f"{embedding_cache_key}.npy"


# ---------------------------------------------------------------------------
# Cache operations
# ---------------------------------------------------------------------------


def is_crop_cached(
    cache_dir: Path | str,
    crop_cache_key: str,
    extension: str = ".jpg",
) -> bool:
    """Check if a crop is cached.

    Args:
        cache_dir: Base cache directory.
        crop_cache_key: Cache key for the crop.
        extension: File extension.

    Returns:
        True if cached, False otherwise.
    """
    path = get_crop_cache_path(cache_dir, crop_cache_key, extension)
    return path.exists()


def is_embedding_cached(
    cache_dir: Path | str,
    embedding_cache_key: str,
) -> bool:
    """Check if an embedding is cached.

    Args:
        cache_dir: Base cache directory.
        embedding_cache_key: Cache key for the embedding.

    Returns:
        True if cached, False otherwise.
    """
    path = get_embedding_cache_path(cache_dir, embedding_cache_key)
    return path.exists()


def save_crop(
    crop: np.ndarray,
    cache_dir: Path | str,
    crop_cache_key: str,
    extension: str = ".jpg",
    quality: int = 95,
) -> Path:
    """Save a crop image to cache.

    Args:
        crop: Crop image array (H, W, 3) in BGR format.
        cache_dir: Base cache directory.
        crop_cache_key: Cache key for the crop.
        extension: File extension (.jpg or .png).
        quality: JPEG quality (1-100), ignored for PNG.

    Returns:
        Path where the crop was saved.
    """
    path = get_crop_cache_path(cache_dir, crop_cache_key, extension)
    path.parent.mkdir(parents=True, exist_ok=True)

    if extension.lower() == ".jpg":
        cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(str(path), crop)

    logger.debug(f"Saved crop to {path}")
    return path


def load_crop(
    cache_dir: Path | str,
    crop_cache_key: str,
    extension: str = ".jpg",
) -> np.ndarray | None:
    """Load a crop image from cache.

    Args:
        cache_dir: Base cache directory.
        crop_cache_key: Cache key for the crop.
        extension: File extension.

    Returns:
        Crop image array (H, W, 3) in BGR format, or None if not found.
    """
    path = get_crop_cache_path(cache_dir, crop_cache_key, extension)

    if not path.exists():
        return None

    crop = cv2.imread(str(path))
    if crop is None:
        logger.warning(f"Failed to load crop from {path}")
        return None

    return crop


def save_embedding(
    embedding: np.ndarray,
    cache_dir: Path | str,
    embedding_cache_key: str,
) -> Path:
    """Save an embedding to cache.

    Args:
        embedding: Embedding vector (1D numpy array).
        cache_dir: Base cache directory.
        embedding_cache_key: Cache key for the embedding.

    Returns:
        Path where the embedding was saved.
    """
    path = get_embedding_cache_path(cache_dir, embedding_cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.save(path, embedding)

    logger.debug(f"Saved embedding to {path}")
    return path


def load_embedding(
    cache_dir: Path | str,
    embedding_cache_key: str,
) -> np.ndarray | None:
    """Load an embedding from cache.

    Args:
        cache_dir: Base cache directory.
        embedding_cache_key: Cache key for the embedding.

    Returns:
        Embedding vector, or None if not found.
    """
    path = get_embedding_cache_path(cache_dir, embedding_cache_key)

    if not path.exists():
        return None

    try:
        return np.load(path)
    except Exception as e:
        logger.warning(f"Failed to load embedding from {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def get_cache_stats(cache_dir: Path | str) -> dict:
    """Get statistics about the cache.

    Args:
        cache_dir: Base cache directory.

    Returns:
        Dict with cache statistics.
    """
    cache_dir = Path(cache_dir)

    crops_dir = cache_dir / "crops"
    embeddings_dir = cache_dir / "embeddings"

    n_crops = 0
    crops_size = 0
    if crops_dir.exists():
        for f in crops_dir.rglob("*"):
            if f.is_file():
                n_crops += 1
                crops_size += f.stat().st_size

    n_embeddings = 0
    embeddings_size = 0
    if embeddings_dir.exists():
        for f in embeddings_dir.rglob("*.npy"):
            n_embeddings += 1
            embeddings_size += f.stat().st_size

    return {
        "n_crops": n_crops,
        "crops_size_mb": round(crops_size / (1024 * 1024), 2),
        "n_embeddings": n_embeddings,
        "embeddings_size_mb": round(embeddings_size / (1024 * 1024), 2),
        "total_size_mb": round((crops_size + embeddings_size) / (1024 * 1024), 2),
    }


def clear_cache(cache_dir: Path | str, crops: bool = True, embeddings: bool = True) -> None:
    """Clear the cache.

    Args:
        cache_dir: Base cache directory.
        crops: If True, clear crop images.
        embeddings: If True, clear embeddings.
    """
    import shutil

    cache_dir = Path(cache_dir)

    if crops:
        crops_dir = cache_dir / "crops"
        if crops_dir.exists():
            shutil.rmtree(crops_dir)
            logger.info(f"Cleared crops cache: {crops_dir}")

    if embeddings:
        embeddings_dir = cache_dir / "embeddings"
        if embeddings_dir.exists():
            shutil.rmtree(embeddings_dir)
            logger.info(f"Cleared embeddings cache: {embeddings_dir}")
