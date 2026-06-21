"""Extract video metadata and validate basic decoding with FFmpeg."""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

FFPROBE_TIMEOUT_SECONDS = 60
FFMPEG_TIMEOUT_SECONDS = 30
DECODE_FRAME_LIMIT = 3


@dataclass
class ProbeResult:
    """Metadata extracted from a video file via ffprobe."""

    duration_s: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    video_codec: str | None = None
    audio_codec: str | None = None
    probe_fps: float | None = None
    decode_ok: bool = True
    probe_error: str | None = None
    _raw: dict[str, Any] = field(default_factory=dict, repr=False)


def _find_ffprobe() -> str:
    """Return the ffprobe executable path.

    Raises
    ------
    RuntimeError
        If ffprobe cannot be found on PATH.
    """
    path = shutil.which("ffprobe")
    if path is None:
        raise RuntimeError(
            "ffprobe not found on PATH. Install the FFmpeg package with "
            "'sudo apt-get install ffmpeg' on Debian/Ubuntu or "
            "'brew install ffmpeg' on macOS."
        )
    return path


def _find_ffmpeg(ffprobe: str) -> str | None:
    """Return the ffmpeg executable path when available.

    PATH is preferred. If ffmpeg is not on PATH, look next to the resolved
    ffprobe executable.
    """
    path = shutil.which("ffmpeg")
    if path is not None:
        return path

    sibling = Path(ffprobe).with_name("ffmpeg")
    return shutil.which(str(sibling))


def _parse_positive_int(value: Any) -> int | None:
    """Parse a positive integer value."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None

    return parsed if parsed > 0 else None


def _parse_non_negative_float(value: Any) -> float | None:
    """Parse a finite, non-negative floating-point value."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(parsed) or parsed < 0:
        return None

    return parsed


def _parse_frame_rate(value: Any) -> float | None:
    """Parse an ffprobe frame-rate value such as ``30000/1001``."""
    if value is None:
        return None

    try:
        parsed = float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return None

    if not math.isfinite(parsed) or parsed <= 0:
        return None

    return parsed


def probe_video(local_path: str | Path) -> ProbeResult:
    """Probe a local video file and validate basic decoding.

    Metadata extraction is performed with ffprobe. When metadata extraction
    succeeds, FFmpeg attempts to decode up to the first three video frames.

    Metadata remains populated when decode validation fails, but ``decode_ok``
    is set to ``False`` and ``probe_error`` contains the FFmpeg error.

    Parameters
    ----------
    local_path:
        Path to the local video file.

    Returns
    -------
    ProbeResult
        Extracted metadata and decode-validation status.
    """
    path = Path(local_path)

    if not path.exists():
        return ProbeResult(
            decode_ok=False,
            probe_error=f"File not found: {path}",
        )

    if not path.is_file():
        return ProbeResult(
            decode_ok=False,
            probe_error=f"Path is not a regular file: {path}",
        )

    try:
        ffprobe = _find_ffprobe()
    except RuntimeError as exc:
        return ProbeResult(
            decode_ok=False,
            probe_error=str(exc),
        )

    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-hide_banner",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=FFPROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ProbeResult(
            decode_ok=False,
            probe_error=(
                f"ffprobe timed out after {FFPROBE_TIMEOUT_SECONDS} seconds"
            ),
        )
    except OSError as exc:
        return ProbeResult(
            decode_ok=False,
            probe_error=f"Could not execute ffprobe: {exc}",
        )

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "no error output"
        return ProbeResult(
            decode_ok=False,
            probe_error=(
                f"ffprobe returned code {completed.returncode}: {stderr}"
            ),
        )

    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return ProbeResult(
            decode_ok=False,
            probe_error=f"ffprobe output is not valid JSON: {exc}",
        )

    if not isinstance(data, dict):
        return ProbeResult(
            decode_ok=False,
            probe_error="ffprobe JSON output is not an object",
        )

    raw_streams = data.get("streams", [])
    if not isinstance(raw_streams, list):
        return ProbeResult(
            decode_ok=False,
            probe_error="ffprobe JSON contains an invalid streams field",
            _raw=data,
        )

    streams = [
        stream
        for stream in raw_streams
        if isinstance(stream, dict)
    ]

    video_stream = next(
        (
            stream
            for stream in streams
            if stream.get("codec_type") == "video"
        ),
        None,
    )

    if video_stream is None:
        return ProbeResult(
            decode_ok=False,
            probe_error="No video stream found in file",
            _raw=data,
        )

    probe_result = ProbeResult(
        width=_parse_positive_int(video_stream.get("width")),
        height=_parse_positive_int(video_stream.get("height")),
        video_codec=video_stream.get("codec_name"),
        _raw=data,
    )

    fps = _parse_frame_rate(video_stream.get("avg_frame_rate"))
    if fps is None:
        fps = _parse_frame_rate(video_stream.get("r_frame_rate"))

    probe_result.fps = fps
    probe_result.probe_fps = fps

    format_data = data.get("format", {})
    if not isinstance(format_data, dict):
        format_data = {}

    duration = _parse_non_negative_float(format_data.get("duration"))
    if duration is None:
        duration = _parse_non_negative_float(video_stream.get("duration"))

    probe_result.duration_s = duration

    audio_stream = next(
        (
            stream
            for stream in streams
            if stream.get("codec_type") == "audio"
        ),
        None,
    )
    if audio_stream is not None:
        probe_result.audio_codec = audio_stream.get("codec_name")

    probe_result = _validate_decode(
        local_path=path,
        ffprobe=ffprobe,
        result=probe_result,
    )

    if not probe_result.decode_ok:
        logger.debug(
            "Decode validation failed for %s: %s",
            path,
            probe_result.probe_error,
        )

    duration_display = (
        f"{probe_result.duration_s:.2f}"
        if probe_result.duration_s is not None
        else "?"
    )

    logger.debug(
        "Probed %s: %sx%s %s fps duration=%ss codec=%s decode_ok=%s",
        path.name,
        probe_result.width if probe_result.width is not None else "?",
        probe_result.height if probe_result.height is not None else "?",
        probe_result.fps if probe_result.fps is not None else "?",
        duration_display,
        probe_result.video_codec or "?",
        probe_result.decode_ok,
    )

    return probe_result


def _validate_decode(
    local_path: Path,
    ffprobe: str,
    result: ProbeResult,
) -> ProbeResult:
    """Validate decoding by reading up to the first three video frames."""
    ffmpeg = _find_ffmpeg(ffprobe)

    if ffmpeg is None:
        result.decode_ok = False
        result.probe_error = (
            "ffmpeg not found on PATH or next to the ffprobe executable"
        )
        return result

    try:
        completed = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                "-i",
                str(local_path),
                "-map",
                "0:v:0",
                "-frames:v",
                str(DECODE_FRAME_LIMIT),
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        result.decode_ok = False
        result.probe_error = (
            f"Decode validation timed out after "
            f"{FFMPEG_TIMEOUT_SECONDS} seconds"
        )
        return result
    except OSError as exc:
        result.decode_ok = False
        result.probe_error = f"Could not execute ffmpeg: {exc}"
        return result

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "no error output"
        result.decode_ok = False
        result.probe_error = (
            f"Decode validation failed with code "
            f"{completed.returncode}: {stderr}"
        )
        return result

    result.decode_ok = True
    result.probe_error = None
    return result
