"""Shared VLM client package for vision-language model inference.

This package provides a reusable OpenAI-compatible VLM client that handles
endpoint configuration, request execution, timeout, retry, structured
response parsing, and audit metadata.

Usage::

    from pickup_putdown.vlm import VlmClientConfig, call_vlm
    from pickup_putdown.vlm.schemas import VlmRequest, VlmResponse

    config = VlmClientConfig(base_url="http://localhost:8080", model="llamacpp/Qwen3.6-27B-UD-Q4_K_XL")
    result = call_vlm(image_b64="...", request_schema=..., config=config)
"""

from pickup_putdown.vlm.client import (
    VlmClientConfig,
    VlmClientError,
    call_vlm,
)
from pickup_putdown.vlm.schemas import (
    VlmError,
    VlmRequest,
    VlmResponse,
    VlmUsage,
)

__all__ = [
    "VlmClientConfig",
    "VlmClientError",
    "VlmRequest",
    "VlmResponse",
    "VlmUsage",
    "VlmError",
    "call_vlm",
]
