"""H.264 candidate encoding and ffprobe validation."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

FFPROBE_TIMEOUT = 30


@dataclass
class EncodingConfig:
    codec: str = "libx264"
    pixel_format: str = "yuv420p"
    preset: str = "fast"
    crf: int = 23
    faststart: bool = True
    retain_audio: bool = False
    keyframe_interval_s: int = 2


@dataclass
class ProbeValidation:
    valid: bool = False
    codec_name: str = ""
    pix_fmt: str = ""
    container: str = ""
    duration_s: float = 0.0
    has_video_stream: bool = False
    error: str = ""


def encode_candidate(
    input_path: Path,
    output_path: Path,
    config: EncodingConfig | None = None,
) -> Path:
    """Encode a candidate clip to H.264/AVC MP4 with fast-start metadata.

    Uses explicit encoding rather than inheriting the source codec.
    """
    if config is None:
        config = EncodingConfig()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH")

    args = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-c:v",
        config.codec,
        "-pix_fmt",
        config.pixel_format,
        "-preset",
        config.preset,
        "-crf",
        str(config.crf),
        "-g",
        str(config.keyframe_interval_s * 30),
        "-force_key_frames",
        f"expr:gte(t,n_forced*{config.keyframe_interval_s})",
    ]

    if config.faststart:
        args.extend(["-movflags", "+faststart"])

    if not config.retain_audio:
        args.append("-an")

    args.append(str(output_path))

    completed = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "no error output"
        raise RuntimeError(f"ffmpeg encoding failed (code {completed.returncode}): {stderr}")

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Encoding produced empty output: {output_path}")

    return output_path


def validate_encoding(output_path: Path) -> ProbeValidation:
    """Validate encoded output with ffprobe.

    Checks:
    - codec_name == h264
    - pix_fmt == yuv420p
    - container includes mp4
    - duration > 0
    - video stream exists
    """
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return ProbeValidation(error="ffprobe not found on PATH")

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
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=FFPROBE_TIMEOUT,
            check=False,
        )

        if completed.returncode != 0:
            return ProbeValidation(error=f"ffprobe failed: {completed.stderr.strip()}")

        data = json.loads(completed.stdout)
        streams = data.get("streams", [])
        fmt = data.get("format", {})

        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        if not video_stream:
            return ProbeValidation(error="No video stream found")

        codec_name = video_stream.get("codec_name", "")
        pix_fmt = video_stream.get("pix_fmt", "")

        duration_str = fmt.get("duration", "0")
        try:
            duration_s = float(duration_str)
        except (TypeError, ValueError):
            duration_s = 0.0

        container = fmt.get("format_name", "")
        errors = []
        if codec_name != "h264":
            errors.append(f"codec_name is {codec_name!r}, expected h264")
        if pix_fmt != "yuv420p":
            errors.append(f"pix_fmt is {pix_fmt!r}, expected yuv420p")
        if "mp4" not in container:
            errors.append(f"container is {container!r}, expected mp4")
        if duration_s <= 0:
            errors.append(f"duration is {duration_s}, expected > 0")

        if errors:
            return ProbeValidation(
                valid=False,
                codec_name=codec_name,
                pix_fmt=pix_fmt,
                container=container,
                duration_s=duration_s,
                has_video_stream=video_stream is not None,
                error="; ".join(errors),
            )

        return ProbeValidation(
            valid=True,
            codec_name=codec_name,
            pix_fmt=pix_fmt,
            container=container,
            duration_s=duration_s,
            has_video_stream=True,
        )

    except subprocess.TimeoutExpired:
        return ProbeValidation(error=f"ffprobe timed out after {FFPROBE_TIMEOUT}s")
    except Exception as exc:
        return ProbeValidation(error=str(exc))
