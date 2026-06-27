"""Tests for the shared VLM client refactor.

Covers:
- Shared schemas (VlmRequest, VlmResponse, VlmUsage, VlmError, VlmClientConfig)
- Shared client (call_vlm) with mocked HTTP
- Annotation wrapper still works (backward-compatible public API)
- Model selection is configurable
- Request metadata and raw responses are preserved
- Errors and retries still work
- Tests require no live VLM endpoint
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pickup_putdown.annotation.schemas import ConfidenceLevel, EventLabel
from pickup_putdown.annotation.vlm_client import (
    ANNOTATION_JSON_SCHEMA,
    SYSTEM_PROMPT,
    VlmClientConfig,
    VlmClientError,
    call_vlm,
    vlm_result_to_annotations,
)
from pickup_putdown.vlm import (
    VlmClientConfig as SharedVlmClientConfig,
)
from pickup_putdown.vlm import (
    VlmClientError as SharedVlmClientError,
)
from pickup_putdown.vlm import (
    VlmError,
    VlmRequest,
    VlmResponse,
    VlmUsage,
)
from pickup_putdown.vlm import (
    call_vlm as shared_call_vlm,
)
from pickup_putdown.vlm.schemas import VlmClientConfig as SchemasVlmClientConfig

# ---------------------------------------------------------------------------
# Shared schema tests
# ---------------------------------------------------------------------------


class TestSharedSchemas:
    """Test that shared schemas are correct and independent."""

    def test_vlm_usage_defaults(self):
        usage = VlmUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_vlm_usage_from_dict(self):
        usage = VlmUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_vlm_request_builds(self):
        req = VlmRequest(
            model="test-model",
            temperature=0.0,
            max_tokens=2048,
            messages=[{"role": "user", "content": "hello"}],
        )
        assert req.model == "test-model"
        assert req.stream is False
        assert req.temperature == 0.0
        assert req.max_tokens == 2048

    def test_vlm_request_serializes(self):
        req = VlmRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
        )
        d = req.model_dump()
        assert d["model"] == "test-model"
        assert d["messages"] == [{"role": "user", "content": "hello"}]
        assert d["stream"] is False

    def test_vlm_response_success(self):
        resp = VlmResponse(content="some output", finish_reason="stop")
        assert resp.is_success is True
        assert resp.error is None

    def test_vlm_response_failure(self):
        resp = VlmResponse(error=VlmError(message="timeout"))
        assert resp.is_success is False
        assert resp.error is not None
        assert resp.error.message == "timeout"

    def test_vlm_response_empty_content_is_failure(self):
        resp = VlmResponse(content="")
        assert resp.is_success is False

    def test_vlm_error_defaults(self):
        err = VlmError(message="test")
        assert err.message == "test"
        assert err.http_status is None

    def test_vlm_client_config_default_endpoint(self):
        cfg = VlmClientConfig()
        assert cfg.endpoint_url == "http://localhost:8080/v1/chat/completions"

    def test_vlm_client_config_custom_base(self):
        cfg = VlmClientConfig(base_url="http://myhost:9000")
        assert cfg.endpoint_url == "http://myhost:9000/v1/chat/completions"

    def test_vlm_client_config_trailing_slash(self):
        cfg = VlmClientConfig(base_url="http://myhost:9000/")
        assert cfg.endpoint_url == "http://myhost:9000/v1/chat/completions"

    def test_shared_and_schema_configs_are_same_class(self):
        """VlmClientConfig from vlm/__init__ and vlm/schemas are the same class."""
        assert SharedVlmClientConfig is SchemasVlmClientConfig


# ---------------------------------------------------------------------------
# Shared client tests (mocked HTTP)
# ---------------------------------------------------------------------------


def _make_mock_response(
    content: str = "{'events': [], 'reasoning': 'none'}",
    finish_reason: str = "stop",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                    "reasoning_content": "",
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class TestSharedClient:
    """Test the shared call_vlm with mocked HTTP."""

    def test_success_returns_content(self):
        mock_resp = _make_mock_response(content="hello world", finish_reason="stop")
        with patch("pickup_putdown.vlm.client._post_json", return_value=mock_resp):
            request = VlmRequest(
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
            )
            config = VlmClientConfig(max_attempts=1)
            response = shared_call_vlm(request, config)

        assert response.is_success is True
        assert response.content == "hello world"
        assert response.finish_reason == "stop"
        assert response.usage.prompt_tokens == 100
        assert response.usage.completion_tokens == 50

    def test_failure_returns_error(self):
        with patch(
            "pickup_putdown.vlm.client._post_json",
            side_effect=VlmClientError("connection refused"),
        ):
            request = VlmRequest(
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
            )
            config = VlmClientConfig(max_attempts=1)
            response = shared_call_vlm(request, config)

        assert response.is_success is False
        assert response.error is not None
        assert "connection refused" in response.error.message

    def test_retry_on_failure(self):
        call_count = 0

        def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise VlmClientError("first fail")
            return _make_mock_response(content="recovered")

        with patch("pickup_putdown.vlm.client._post_json", side_effect=_side_effect):
            request = VlmRequest(
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
            )
            config = VlmClientConfig(max_attempts=3, retry_delay_s=0.0)
            response = shared_call_vlm(request, config)

        assert response.is_success is True
        assert response.content == "recovered"
        assert call_count == 2

    def test_exhausts_retries(self):
        with patch(
            "pickup_putdown.vlm.client._post_json",
            side_effect=VlmClientError("always fails"),
        ):
            request = VlmRequest(
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
            )
            config = VlmClientConfig(max_attempts=3, retry_delay_s=0.0)
            response = shared_call_vlm(request, config)

        assert response.is_success is False
        assert response.error is not None
        assert "always fails" in response.error.message

    def test_model_is_passed_to_request(self):
        captured_payload = {}

        def _capture(*args, **kwargs):
            captured_payload["payload"] = kwargs.get("payload", args[1] if len(args) > 1 else {})
            return _make_mock_response()

        with patch("pickup_putdown.vlm.client._post_json", side_effect=_capture):
            request = VlmRequest(
                model="llamacpp/Qwen3.6-27B-UD-Q4_K_XL",
                messages=[{"role": "user", "content": "test"}],
            )
            config = VlmClientConfig(max_attempts=1)
            shared_call_vlm(request, config)

        assert captured_payload["payload"]["model"] == "llamacpp/Qwen3.6-27B-UD-Q4_K_XL"

    def test_empty_content_is_failure(self):
        mock_resp = _make_mock_response(content="")
        with patch("pickup_putdown.vlm.client._post_json", return_value=mock_resp):
            request = VlmRequest(
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
            )
            config = VlmClientConfig(max_attempts=1)
            response = shared_call_vlm(request, config)

        assert response.is_success is False

    def test_reasoning_content_preserved(self):
        mock_resp = _make_mock_response(
            content="events here",
            finish_reason="stop",
        )
        # Override reasoning_content in the mock
        mock_resp["choices"][0]["message"]["reasoning_content"] = "thinking..."

        with patch("pickup_putdown.vlm.client._post_json", return_value=mock_resp):
            request = VlmRequest(
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
            )
            config = VlmClientConfig(max_attempts=1)
            response = shared_call_vlm(request, config)

        assert response.reasoning_content == "thinking..."

    def test_http_error_is_caught(self):
        import urllib.error

        http_err = urllib.error.HTTPError(
            "http://localhost:8080/v1/chat/completions",
            500,
            "Internal Server Error",
            {},
            None,
        )
        http_err.fp = MagicMock(read=lambda: b'{"error": "server error"}')

        def _raise_http(*args, **kwargs):
            raise http_err

        with patch("pickup_putdown.vlm.client._post_json", side_effect=_raise_http):
            request = VlmRequest(
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
            )
            config = VlmClientConfig(max_attempts=1)
            response = shared_call_vlm(request, config)

        assert response.is_success is False
        assert "500" in response.error.message

    def test_non_json_response_raises(self):
        def _non_json(*args, **kwargs):
            raise VlmClientError("not JSON")

        with patch("pickup_putdown.vlm.client._post_json", side_effect=_non_json):
            request = VlmRequest(
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
            )
            config = VlmClientConfig(max_attempts=1)
            response = shared_call_vlm(request, config)

        assert response.is_success is False


# ---------------------------------------------------------------------------
# Annotation wrapper tests (backward compatibility)
# ---------------------------------------------------------------------------


class TestAnnotationWrapper:
    """Test that the annotation wrapper still works with the shared client."""

    def test_vlm_client_config_is_reexported(self):
        """VlmClientConfig from annotation.vlm_client is the shared class."""
        assert VlmClientConfig is SharedVlmClientConfig

    def test_vlm_client_error_is_reexported(self):
        """VlmClientError from annotation.vlm_client is the shared class."""
        assert VlmClientError is SharedVlmClientError

    def test_annotation_schema_unchanged(self):
        """ANNOTATION_JSON_SCHEMA has the expected structure."""
        assert "events" in ANNOTATION_JSON_SCHEMA["properties"]
        assert "reasoning" in ANNOTATION_JSON_SCHEMA["properties"]
        assert ANNOTATION_JSON_SCHEMA["required"] == ["events", "reasoning"]
        event_schema = ANNOTATION_JSON_SCHEMA["properties"]["events"]["items"]
        assert event_schema["required"] == [
            "label",
            "start_frame",
            "end_frame",
            "item_count",
            "confidence",
            "hard_case",
            "notes",
        ]

    def test_system_prompt_unchanged(self):
        """SYSTEM_PROMPT contains key event definitions."""
        assert "pickup" in SYSTEM_PROMPT.lower()
        assert "putdown" in SYSTEM_PROMPT.lower()
        assert "shelf" in SYSTEM_PROMPT.lower()

    def test_call_vlm_success(self, tmp_path):
        """call_vlm returns success dict when shared client succeeds."""
        contact_sheet = tmp_path / "sheet.jpg"
        contact_sheet.write_bytes(b"fake image")

        mock_response = _make_mock_response(
            content='{"events": [{"label": "pickup", "start_frame": 0, "end_frame": 2, "item_count": 1, "confidence": "high", "hard_case": false, "notes": "test"}], "reasoning": "seen it"}'
        )

        with patch("pickup_putdown.vlm.client._post_json", return_value=mock_response):
            config = VlmClientConfig(max_attempts=1)
            result = call_vlm(
                contact_sheet_path=contact_sheet,
                frame_count=10,
                fps=5.0,
                duration_s=2.0,
                config=config,
            )

        assert result["status"] == "success"
        assert len(result["events"]) == 1
        assert result["events"][0]["label"] == "pickup"
        assert result["attempts"] == 1

    def test_call_vlm_file_not_found(self, tmp_path):
        """call_vlm returns failure when contact sheet is missing."""
        config = VlmClientConfig(max_attempts=1)
        result = call_vlm(
            contact_sheet_path=tmp_path / "nonexistent.jpg",
            frame_count=10,
            fps=5.0,
            duration_s=2.0,
            config=config,
        )
        assert result["status"] == "failed"
        assert "not found" in result["error"].lower()

    def test_call_vlm_invalid_frame_count(self, tmp_path):
        """call_vlm returns failure when frame_count <= 0."""
        contact_sheet = tmp_path / "sheet.jpg"
        contact_sheet.write_bytes(b"fake")
        config = VlmClientConfig(max_attempts=1)
        result = call_vlm(
            contact_sheet_path=contact_sheet,
            frame_count=0,
            fps=5.0,
            duration_s=2.0,
            config=config,
        )
        assert result["status"] == "failed"

    def test_vlm_result_to_annotations_success(self):
        vlm_response = {
            "status": "success",
            "events": [
                {
                    "label": "pickup",
                    "start_frame": 5,
                    "end_frame": 8,
                    "item_count": 1,
                    "confidence": "high",
                    "hard_case": False,
                    "notes": "test event",
                }
            ],
            "reasoning": "test",
            "error": None,
            "attempts": 1,
            "finish_reason": "stop",
            "raw_response": "",
            "usage": {},
        }
        annotations = vlm_result_to_annotations(vlm_response, fps=5.0)
        assert len(annotations) == 1
        assert annotations[0]["label"] == EventLabel.PICKUP
        assert annotations[0]["start_s"] == pytest.approx(1.0)
        assert annotations[0]["end_s"] == pytest.approx(1.8)
        assert annotations[0]["confidence"] == ConfidenceLevel.HIGH

    def test_vlm_result_to_annotations_failed_raises(self):
        vlm_response = {
            "status": "failed",
            "events": [],
            "reasoning": "",
            "error": "connection refused",
            "attempts": 2,
        }
        with pytest.raises(VlmClientError, match="connection refused"):
            vlm_result_to_annotations(vlm_response, fps=5.0)

    def test_vlm_result_to_annotations_zero_fps_fallback(self):
        vlm_response = {
            "status": "success",
            "events": [
                {
                    "label": "putdown",
                    "start_frame": 0,
                    "end_frame": 4,
                    "item_count": 1,
                    "confidence": "med",
                    "hard_case": False,
                    "notes": "",
                }
            ],
            "reasoning": "",
            "error": None,
            "attempts": 1,
            "finish_reason": "stop",
            "raw_response": "",
            "usage": {},
        }
        with patch("pickup_putdown.annotation.vlm_client.logger") as mock_logger:
            annotations = vlm_result_to_annotations(vlm_response, fps=0.0)
        assert len(annotations) == 1
        assert annotations[0]["start_s"] == pytest.approx(0.0)
        assert annotations[0]["end_s"] == pytest.approx(1.0)
        mock_logger.warning.assert_called_once()

    def test_call_vlm_preserves_metadata(self, tmp_path):
        """call_vlm preserves usage, finish_reason, and raw_response in result."""
        contact_sheet = tmp_path / "sheet.jpg"
        contact_sheet.write_bytes(b"fake")

        mock_response = _make_mock_response(
            content='{"events": [], "reasoning": "none"}',
            finish_reason="stop",
            prompt_tokens=200,
            completion_tokens=30,
        )

        with patch("pickup_putdown.vlm.client._post_json", return_value=mock_response):
            config = VlmClientConfig(max_attempts=1)
            result = call_vlm(
                contact_sheet_path=contact_sheet,
                frame_count=5,
                fps=5.0,
                duration_s=1.0,
                config=config,
            )

        assert result["status"] == "success"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 200
        assert result["usage"]["completion_tokens"] == 30

    def test_call_vlm_retry_increases_attempts(self, tmp_path):
        """call_vlm retries and reports correct attempt count on second try."""
        contact_sheet = tmp_path / "sheet.jpg"
        contact_sheet.write_bytes(b"fake")

        call_count = 0

        def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise VlmClientError("first fail")
            return _make_mock_response('{"events": [], "reasoning": "ok"}')

        # The annotation wrapper handles its own retry loop; the shared
        # client is called with max_attempts=1 so it does not double-retry.
        with patch("pickup_putdown.vlm.client._post_json", side_effect=_side_effect):
            config = VlmClientConfig(max_attempts=2, retry_delay_s=0.0)
            result = call_vlm(
                contact_sheet_path=contact_sheet,
                frame_count=5,
                fps=5.0,
                duration_s=1.0,
                config=config,
            )

        assert result["status"] == "success"
        assert result["attempts"] == 2

    def test_model_selection_configurable_27b(self):
        config = VlmClientConfig(model="llamacpp/Qwen3.6-27B-UD-Q4_K_XL")
        assert config.model == "llamacpp/Qwen3.6-27B-UD-Q4_K_XL"

    def test_model_selection_configurable_35b(self):
        config = VlmClientConfig(model="llamacpp/Qwen3.6-35B-A3B-UD-Q4_K_XL")
        assert config.model == "llamacpp/Qwen3.6-35B-A3B-UD-Q4_K_XL"

    def test_model_selection_empty_default(self):
        config = VlmClientConfig()
        assert config.model == ""

    def test_vlm_result_to_annotations_with_duration_clamping(self):
        vlm_response = {
            "status": "success",
            "events": [
                {
                    "label": "pickup",
                    "start_frame": 40,
                    "end_frame": 50,
                    "item_count": 1,
                    "confidence": "high",
                    "hard_case": False,
                    "notes": "",
                }
            ],
            "reasoning": "",
            "error": None,
            "attempts": 1,
            "finish_reason": "stop",
            "raw_response": "",
            "usage": {},
        }
        # fps=5.0, so frame 40 = 8.0s, frame 50 = 10.2s
        # With duration_s=9.0, end_s should be clamped to 9.0
        annotations = vlm_result_to_annotations(vlm_response, fps=5.0, duration_s=9.0)
        assert len(annotations) == 1
        assert annotations[0]["start_s"] == pytest.approx(8.0)
        assert annotations[0]["end_s"] == pytest.approx(9.0)

    def test_vlm_result_to_annotations_medium_confidence_normalized(self):
        vlm_response = {
            "status": "success",
            "events": [
                {
                    "label": "pickup",
                    "start_frame": 0,
                    "end_frame": 4,
                    "item_count": 1,
                    "confidence": "medium",
                    "hard_case": False,
                    "notes": "",
                }
            ],
            "reasoning": "",
            "error": None,
            "attempts": 1,
            "finish_reason": "stop",
            "raw_response": "",
            "usage": {},
        }
        annotations = vlm_result_to_annotations(vlm_response, fps=5.0)
        assert annotations[0]["confidence"] == ConfidenceLevel.MED

    def test_annotation_config_has_all_fields(self):
        """VlmClientConfig has all expected fields."""
        config = VlmClientConfig(
            base_url="http://test:8080",
            model="test-model",
            temperature=0.5,
            max_tokens=1024,
            retry_max_tokens=2048,
            max_attempts=3,
            retry_delay_s=0.5,
            timeout_s=60,
            disable_thinking=False,
            enforce_json_schema=False,
        )
        assert config.base_url == "http://test:8080"
        assert config.model == "test-model"
        assert config.temperature == 0.5
        assert config.max_tokens == 1024
        assert config.retry_max_tokens == 2048
        assert config.max_attempts == 3
        assert config.retry_delay_s == 0.5
        assert config.timeout_s == 60
        assert config.disable_thinking is False
        assert config.enforce_json_schema is False
