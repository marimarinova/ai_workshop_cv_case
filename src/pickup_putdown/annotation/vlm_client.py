"""llama.cpp OpenAI-compatible vision client for VLM annotation.

Sends contact sheet images to a local llama.cpp server running a vision model
and parses structured JSON responses into event annotations.

Invalid, empty, or truncated responses are retried. Final failures are returned
with ``status="failed"`` and must not be treated as valid zero-event results.
"""

from __future__ import annotations

import base64
import json
import logging
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, Field

from pickup_putdown.annotation.schemas import ConfidenceLevel, EventLabel

logger = logging.getLogger(__name__)


class VlmClientConfig(BaseModel):
    """Configuration for the llama.cpp VLM client."""

    base_url: str = "http://localhost:8080"
    model: str = ""
    temperature: float = 0.0

    # First-attempt output budget.
    max_tokens: int = Field(default=2048, ge=256)

    # Used after a truncated or malformed first response.
    retry_max_tokens: int = Field(default=4096, ge=256)
    max_attempts: int = Field(default=2, ge=1, le=5)
    retry_delay_s: float = Field(default=1.0, ge=0.0)

    timeout_s: int = Field(default=180, ge=1)

    # Structured annotation should not spend tokens on hidden reasoning.
    disable_thinking: bool = True
    enforce_json_schema: bool = True


class VlmClientError(RuntimeError):
    """Raised when a VLM response cannot be used safely."""


ANNOTATION_JSON_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "maxItems": 20,
            "items": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "enum": ["pickup", "putdown"],
                    },
                    "start_frame": {
                        "type": "integer",
                        "minimum": 0,
                    },
                    "end_frame": {
                        "type": "integer",
                        "minimum": 0,
                    },
                    "item_count": {
                        "type": "integer",
                        "minimum": 1,
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "med", "low"],
                    },
                    "hard_case": {
                        "type": "boolean",
                    },
                    "notes": {
                        "type": "string",
                        "maxLength": 300,
                    },
                },
                "required": [
                    "label",
                    "start_frame",
                    "end_frame",
                    "item_count",
                    "confidence",
                    "hard_case",
                    "notes",
                ],
                "additionalProperties": False,
            },
        },
        "reasoning": {
            "type": "string",
            "maxLength": 600,
        },
    },
    "required": ["events", "reasoning"],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """\
You are a video event annotator for a retail pickup/putdown detection task.

Analyze one contact sheet containing sequential frames from a short candidate
video clip. Each frame is labeled with its frame number.

Event definitions:

pickup:
A person removes an item from a shelf or surface and takes control of it, so the
item leaves its resting place.

putdown:
A person places an item they are holding onto a shelf or surface and releases
it, so the item remains resting there.

Do not annotate:
- touching or inspecting an item without removing it
- reaching past an item
- browsing or standing near shelves
- hand motion without an object transfer
- clips with no visible person
- ambiguous motion where no object transfer can be observed

For every valid event return:
- label: "pickup" or "putdown"
- start_frame: first numbered frame where the transfer action begins
- end_frame: last numbered frame where the transfer is complete
- item_count: number of transferred items
- confidence: "high", "med", or "low"
- hard_case: true only when visibility or timing is ambiguous
- notes: one short evidence-based explanation

If no valid event is visible, return an empty events array.

Return exactly one complete JSON object. Do not use Markdown or code fences.
Keep reasoning to at most two short sentences.
"""


def _image_to_base64(image_path: Path) -> str:
    """Read an image file and return a base64-encoded string."""

    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def _failure_result(
    *,
    error: str,
    attempts: int,
    finish_reason: str | None = None,
    raw_response: str = "",
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an explicit failed VLM result."""

    return {
        "status": "failed",
        "events": [],
        "reasoning": "",
        "error": error,
        "attempts": attempts,
        "finish_reason": finish_reason,
        "raw_response": raw_response[:2_000],
        "usage": usage or {},
    }


def _build_payload(
    *,
    image_b64: str,
    frame_count: int,
    fps: float,
    duration_s: float,
    config: VlmClientConfig,
    max_tokens: int,
    attempt: int,
) -> dict[str, Any]:
    """Build one OpenAI-compatible chat-completions request."""

    retry_instruction = ""
    if attempt > 1:
        retry_instruction = (
            " This is a retry because the previous response was incomplete or "
            "invalid. Return a complete JSON object and nothing else."
        )

    user_text = (
        f"The contact sheet contains {frame_count} numbered frames sampled at "
        f"{fps:.1f} review FPS over a {duration_s:.1f}-second candidate window. "
        "Inspect the complete sequence and identify all valid pickup or putdown "
        f"events. Valid frame numbers are 0 through {frame_count - 1}."
        f"{retry_instruction}"
    )

    payload: dict[str, Any] = {
        "model": config.model,
        "stream": False,
        "temperature": config.temperature,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                        },
                    },
                ],
            },
        ],
    }

    if config.disable_thinking:
        payload["chat_template_kwargs"] = {
            "enable_thinking": False,
        }

    if config.enforce_json_schema:
        payload["response_format"] = {
            "type": "json_object",
            "schema": ANNOTATION_JSON_SCHEMA,
        }

    return payload


def _post_json(
    *,
    url: str,
    payload: dict[str, Any],
    timeout_s: int,
) -> dict[str, Any]:
    """POST JSON and decode the JSON response."""

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise VlmClientError(f"VLM HTTP {exc.code}: {error_body[:1_000]}") from exc
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        raise VlmClientError(f"VLM request failed: {exc}") from exc

    try:
        decoded = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise VlmClientError(
            f"llama.cpp returned a non-JSON API response: {response_body[:1_000]!r}"
        ) from exc

    if not isinstance(decoded, dict):
        raise VlmClientError(f"Unexpected top-level API response type: {type(decoded).__name__}")

    return decoded


def _extract_assistant_response(
    response: dict[str, Any],
) -> tuple[str, str | None, str, dict[str, Any]]:
    """Extract content and termination metadata from a chat response."""

    try:
        choice = response["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise VlmClientError(
            "Unexpected VLM response structure: missing choices[0].message"
        ) from exc

    finish_reason_raw = choice.get("finish_reason")
    finish_reason = str(finish_reason_raw) if finish_reason_raw is not None else None

    content_raw = message.get("content")
    content = content_raw if isinstance(content_raw, str) else ""

    reasoning_raw = message.get("reasoning_content")
    reasoning_content = reasoning_raw if isinstance(reasoning_raw, str) else ""

    usage_raw = response.get("usage")
    usage = usage_raw if isinstance(usage_raw, dict) else {}

    logger.debug(
        (
            "VLM completion: finish_reason=%s content_chars=%d "
            "reasoning_chars=%d prompt_tokens=%s completion_tokens=%s"
        ),
        finish_reason,
        len(content),
        len(reasoning_content),
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
    )

    if finish_reason == "length":
        raise VlmClientError("VLM response reached the output-token limit")

    if not content.strip():
        if reasoning_content:
            raise VlmClientError(
                "VLM returned no final content but produced "
                f"{len(reasoning_content)} reasoning characters"
            )

        raise VlmClientError("VLM returned empty assistant content")

    return content, finish_reason, reasoning_content, usage


def _strip_code_fence(content: str) -> str:
    """Remove one optional Markdown code fence."""

    text = content.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()

    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]

    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]

    return "\n".join(lines).strip()


def _coerce_int(value: Any, field_name: str) -> int:
    """Convert a JSON value to an integer without accepting booleans."""

    if isinstance(value, bool):
        raise VlmClientError(f"{field_name} must be an integer")

    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise VlmClientError(f"{field_name} must be an integer, got {value!r}") from exc

    return converted


def _normalize_event(
    event: Any,
    *,
    frame_count: int | None,
) -> dict[str, Any]:
    """Validate and normalize one event object."""

    if not isinstance(event, dict):
        raise VlmClientError(f"Event must be an object, got {type(event).__name__}")

    label = str(event.get("label", "")).strip().lower()
    if label not in {"pickup", "putdown"}:
        raise VlmClientError(f"Unsupported event label: {label!r}")

    start_frame = _coerce_int(event.get("start_frame"), "start_frame")
    end_frame = _coerce_int(event.get("end_frame"), "end_frame")
    item_count = _coerce_int(event.get("item_count", 1), "item_count")

    if start_frame < 0:
        raise VlmClientError("start_frame must be non-negative")

    if end_frame < start_frame:
        raise VlmClientError(f"end_frame {end_frame} is before start_frame {start_frame}")

    if frame_count is not None and frame_count > 0:
        if start_frame >= frame_count or end_frame >= frame_count:
            raise VlmClientError(
                "Event frame range is outside the contact sheet: "
                f"{start_frame}-{end_frame}, frame_count={frame_count}"
            )

    if item_count < 1:
        raise VlmClientError("item_count must be at least 1")

    confidence = str(event.get("confidence", "med")).strip().lower()
    if confidence == "medium":
        confidence = "med"

    if confidence not in {"high", "med", "low"}:
        raise VlmClientError(f"Unsupported confidence value: {confidence!r}")

    hard_case_raw = event.get("hard_case", False)
    if not isinstance(hard_case_raw, bool):
        raise VlmClientError("hard_case must be a boolean")

    return {
        "label": label,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "item_count": item_count,
        "confidence": confidence,
        "hard_case": hard_case_raw,
        "notes": str(event.get("notes", "")).strip(),
    }


def _parse_vlm_response(
    content: str,
    frame_count: int | None = None,
) -> dict[str, Any]:
    """Parse and validate a VLM JSON response.

    ``frame_count`` is optional to preserve compatibility with direct unit
    tests of this function.
    """

    text = _strip_code_fence(content)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise VlmClientError(
            "Failed to parse VLM content as JSON: "
            f"{exc.msg} at line {exc.lineno}, column {exc.colno}"
        ) from exc

    if not isinstance(parsed, dict):
        raise VlmClientError(f"VLM output must be a JSON object, got {type(parsed).__name__}")

    events_raw = parsed.get("events")
    if not isinstance(events_raw, list):
        raise VlmClientError("VLM output field 'events' must be an array")

    reasoning_raw = parsed.get("reasoning", "")
    if not isinstance(reasoning_raw, str):
        raise VlmClientError("VLM output field 'reasoning' must be a string")

    events = [_normalize_event(event, frame_count=frame_count) for event in events_raw]

    return {
        "events": events,
        "reasoning": reasoning_raw.strip(),
    }


def call_vlm(
    contact_sheet_path: Path,
    frame_count: int,
    fps: float,
    duration_s: float,
    config: VlmClientConfig,
) -> dict[str, Any]:
    """Call the VLM through llama.cpp's OpenAI-compatible API.

    Successful result::

        {
            "status": "success",
            "events": [...],
            "reasoning": "...",
            "error": None,
            "attempts": 1,
            "finish_reason": "stop",
            "usage": {...},
        }

    Failed result::

        {
            "status": "failed",
            "events": [],
            "reasoning": "",
            "error": "...",
            "attempts": 2,
            ...
        }

    A failed result is not a valid no-event annotation.
    """

    if not contact_sheet_path.is_file():
        return _failure_result(
            error=f"Contact sheet not found: {contact_sheet_path}",
            attempts=0,
        )

    if frame_count <= 0:
        return _failure_result(
            error=f"frame_count must be positive, got {frame_count}",
            attempts=0,
        )

    image_b64 = _image_to_base64(contact_sheet_path)
    url = f"{config.base_url.rstrip('/')}/v1/chat/completions"

    last_error = "Unknown VLM failure"
    last_raw_response = ""
    last_finish_reason: str | None = None
    last_usage: dict[str, Any] = {}

    for attempt in range(1, config.max_attempts + 1):
        max_tokens = (
            config.max_tokens if attempt == 1 else max(config.max_tokens, config.retry_max_tokens)
        )

        payload = _build_payload(
            image_b64=image_b64,
            frame_count=frame_count,
            fps=fps,
            duration_s=duration_s,
            config=config,
            max_tokens=max_tokens,
            attempt=attempt,
        )

        try:
            response = _post_json(
                url=url,
                payload=payload,
                timeout_s=config.timeout_s,
            )

            content, finish_reason, _, usage = _extract_assistant_response(response)

            last_raw_response = content
            last_finish_reason = finish_reason
            last_usage = usage

            parsed = _parse_vlm_response(
                content,
                frame_count=frame_count,
            )

            return {
                "status": "success",
                "events": parsed["events"],
                "reasoning": parsed["reasoning"],
                "error": None,
                "attempts": attempt,
                "finish_reason": finish_reason,
                "raw_response": "",
                "usage": usage,
            }

        except VlmClientError as exc:
            last_error = str(exc)

            logger.warning(
                (
                    "VLM annotation attempt %d/%d failed for %s "
                    "(max_tokens=%d, finish_reason=%s): %s"
                ),
                attempt,
                config.max_attempts,
                contact_sheet_path.name,
                max_tokens,
                last_finish_reason,
                exc,
            )

        except Exception as exc:
            last_error = f"Unexpected VLM client error: {exc}"

            logger.exception(
                "Unexpected VLM annotation error for %s",
                contact_sheet_path.name,
            )

        if attempt < config.max_attempts and config.retry_delay_s > 0:
            time.sleep(config.retry_delay_s)

    logger.error(
        "VLM annotation failed after %d attempts for %s: %s",
        config.max_attempts,
        contact_sheet_path.name,
        last_error,
    )

    return _failure_result(
        error=last_error,
        attempts=config.max_attempts,
        finish_reason=last_finish_reason,
        raw_response=last_raw_response,
        usage=last_usage,
    )


def vlm_result_to_annotations(
    vlm_response: dict[str, Any],
    fps: float,
    duration_s: float | None = None,
) -> list[dict[str, Any]]:
    """Convert frame-based VLM events to time-based annotations.

    Raises:
        VlmClientError: If the VLM call failed. A failed call must never be
            converted into a valid empty annotation list.
    """

    if vlm_response.get("status") == "failed":
        raise VlmClientError(str(vlm_response.get("error") or "VLM annotation failed"))

    if fps <= 0:
        logger.warning(
            "Invalid review FPS %.3f; falling back to 5.0 FPS",
            fps,
        )
        fps = 5.0

    annotations: list[dict[str, Any]] = []

    confidence_map = {
        "high": ConfidenceLevel.HIGH,
        "med": ConfidenceLevel.MED,
        "medium": ConfidenceLevel.MED,
        "low": ConfidenceLevel.LOW,
    }

    for event in vlm_response.get("events", []):
        start_frame = _coerce_int(
            event.get("start_frame", 0),
            "start_frame",
        )
        end_frame = _coerce_int(
            event.get("end_frame", 0),
            "end_frame",
        )

        start_s = start_frame / fps
        end_s = (end_frame + 1) / fps

        if duration_s is not None:
            start_s = min(start_s, duration_s)
            end_s = min(end_s, duration_s)

        if end_s <= start_s:
            raise VlmClientError(
                f"Invalid event interval after clamping: start_s={start_s}, end_s={end_s}"
            )

        confidence_str = str(event.get("confidence", "med")).strip().lower()

        annotations.append(
            {
                "label": EventLabel(event["label"]),
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "item_count": event.get("item_count", 1),
                "confidence": confidence_map.get(
                    confidence_str,
                    ConfidenceLevel.MED,
                ),
                "hard_case": event.get("hard_case", False),
                "notes": event.get("notes", ""),
            }
        )

    return annotations
