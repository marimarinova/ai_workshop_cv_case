"""Shared VLM request/response/audit schemas.

These Pydantic models define the contract between the VLM client and any
caller (annotation, layer2, etc.).  They are deliberately free of
annotation-specific or layer2-specific concepts.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class VlmUsage(BaseModel):
    """Token-usage metadata returned by the VLM API."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class VlmRequest(BaseModel):
    """One OpenAI-compatible chat-completions request payload."""

    model: str
    stream: bool = False
    temperature: float = 0.0
    max_tokens: int = Field(default=2048, ge=256)
    messages: list[dict[str, Any]]
    chat_template_kwargs: dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None


class VlmError(BaseModel):
    """Structured error from a failed VLM call."""

    message: str
    http_status: int | None = None


class VlmResponse(BaseModel):
    """Parsed VLM API response with audit metadata."""

    content: str = ""
    finish_reason: str | None = None
    reasoning_content: str = ""
    usage: VlmUsage = Field(default_factory=VlmUsage)
    raw_response: str = ""
    error: VlmError | None = None

    @property
    def is_success(self) -> bool:
        return self.error is None and bool(self.content.strip())


class VlmClientConfig(BaseModel):
    """Configuration for the llama.cpp OpenAI-compatible VLM client."""

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

    @property
    def endpoint_url(self) -> str:
        """Full chat-completions endpoint URL."""
        return f"{self.base_url.rstrip('/')}/v1/chat/completions"
