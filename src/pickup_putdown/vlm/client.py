"""Shared VLM client: HTTP transport, retry, and response extraction.

This module is independent of any annotation or layer2 logic.  It only
knows about OpenAI-compatible chat-completions endpoints and returns
structured ``VlmResponse`` objects.

Callers supply their own ``VlmRequest`` payloads and are responsible for
prompt construction, schema enforcement, and post-parsing validation.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

from pickup_putdown.vlm.schemas import (
    VlmClientConfig,
    VlmError,
    VlmRequest,
    VlmResponse,
    VlmUsage,
)

logger = logging.getLogger(__name__)


class VlmClientError(RuntimeError):
    """Raised when a VLM response cannot be used safely."""


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
    except (urllib.error.URLError, TimeoutError) as exc:
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
) -> tuple[str, str | None, str, VlmUsage]:
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
    usage = (
        VlmUsage(
            prompt_tokens=int(usage_raw.get("prompt_tokens", 0))
            if isinstance(usage_raw, dict)
            else 0,
            completion_tokens=int(usage_raw.get("completion_tokens", 0))
            if isinstance(usage_raw, dict)
            else 0,
            total_tokens=int(usage_raw.get("total_tokens", 0))
            if isinstance(usage_raw, dict)
            else 0,
        )
        if isinstance(usage_raw, dict)
        else VlmUsage()
    )

    logger.debug(
        "VLM completion: finish_reason=%s content_chars=%d reasoning_chars=%d prompt_tokens=%s completion_tokens=%s",
        finish_reason,
        len(content),
        len(reasoning_content),
        usage.prompt_tokens,
        usage.completion_tokens,
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


def call_vlm(
    request: VlmRequest,
    config: VlmClientConfig,
    *,
    max_attempts: int | None = None,
) -> VlmResponse:
    """Call the VLM through an OpenAI-compatible API.

    Parameters
    ----------
    request:
        The structured request payload (model, messages, etc.).
    config:
        Client configuration (endpoint, timeout, retry budget).
    max_attempts:
        Override ``config.max_attempts`` for this single call.

    Returns a ``VlmResponse``.  On exhaustion the response has
    ``is_success=False`` and ``error`` set.
    """
    attempts = max_attempts if max_attempts is not None else config.max_attempts
    last_error: VlmError | None = None
    last_finish_reason: str | None = None
    last_usage = VlmUsage()
    last_raw_response = ""

    for attempt in range(1, attempts + 1):
        payload = request.model_dump(exclude_none=False)
        # Ensure we send the model field even if empty
        payload.setdefault("model", config.model)

        try:
            raw = _post_json(
                url=config.endpoint_url,
                payload=payload,
                timeout_s=config.timeout_s,
            )
            content, finish_reason, reasoning, usage = _extract_assistant_response(raw)
            last_raw_response = content
            last_finish_reason = finish_reason
            last_usage = usage

            return VlmResponse(
                content=content,
                finish_reason=finish_reason,
                reasoning_content=reasoning,
                usage=usage,
                raw_response="",
            )

        except VlmClientError as exc:
            last_error = VlmError(message=str(exc))
            logger.warning(
                "VLM attempt %d/%d failed (max_tokens=%d, finish_reason=%s): %s",
                attempt,
                attempts,
                request.max_tokens,
                last_finish_reason,
                exc,
            )

        except Exception as exc:
            last_error = VlmError(message=f"Unexpected VLM client error: {exc}")
            logger.exception("Unexpected VLM error on attempt %d", attempt)

        if attempt < attempts and config.retry_delay_s > 0:
            time.sleep(config.retry_delay_s)

    logger.error(
        "VLM failed after %d attempts: %s",
        attempts,
        last_error.message if last_error else "unknown",
    )
    return VlmResponse(
        content="",
        finish_reason=last_finish_reason,
        reasoning_content="",
        usage=last_usage,
        raw_response=last_raw_response,
        error=last_error,
    )
