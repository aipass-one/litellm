"""
Tests for the shared FalAIBaseVideoConfig.

Per-model behavior (param mapping, endpoint slug) is tested in
``test_seedance.py``. This file exercises the queue lifecycle plumbing
every Fal video config inherits.
"""
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.fal_ai.video_generation.base import FalAIBaseVideoConfig


class _StubConfig(FalAIBaseVideoConfig):
    """Minimal subclass for direct base-class testing."""

    VIDEO_ENDPOINT = "stub/test-video-model"


# ---------------------------------------------------------------- env+url


class TestEnvAndUrl:
    def setup_method(self):
        self.config = _StubConfig()

    def test_get_complete_url_default_queue_base(self):
        url = self.config.get_complete_url(model="x", api_base=None, litellm_params={})
        assert url == "https://queue.fal.run/stub/test-video-model"

    def test_get_complete_url_strips_trailing_slash(self):
        url = self.config.get_complete_url(
            model="x",
            api_base="https://custom.queue.fal.run/",
            litellm_params={},
        )
        assert url == "https://custom.queue.fal.run/stub/test-video-model"

    def test_get_complete_url_falls_back_to_model_when_endpoint_empty(self):
        class _NoEndpoint(FalAIBaseVideoConfig):
            VIDEO_ENDPOINT = ""

        url = _NoEndpoint().get_complete_url(
            model="fal_ai/some/other/endpoint",
            api_base=None,
            litellm_params={},
        )
        assert url == "https://queue.fal.run/some/other/endpoint"

    def test_validate_environment_uses_fal_key(self, monkeypatch):
        monkeypatch.setenv("FAL_AI_API_KEY", "test-key")
        headers = self.config.validate_environment(headers={}, model="x", api_key=None)
        assert headers["Authorization"] == "Key test-key"
        assert headers["Content-Type"] == "application/json"

    def test_validate_environment_explicit_api_key_wins(self, monkeypatch):
        monkeypatch.setenv("FAL_AI_API_KEY", "env-key")
        headers = self.config.validate_environment(
            headers={}, model="x", api_key="explicit-key"
        )
        assert headers["Authorization"] == "Key explicit-key"

    def test_validate_environment_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("FAL_AI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="FAL_AI_API_KEY"):
            self.config.validate_environment(headers={}, model="x", api_key=None)


# --------------------------------------------------------------- request


class TestCreateRequest:
    def setup_method(self):
        self.config = _StubConfig()

    def test_request_body_carries_prompt_and_image_url(self):
        body, files, url = self.config.transform_video_create_request(
            model="fal_ai/stub/test-video-model",
            prompt="a cat blinking",
            api_base="https://queue.fal.run/stub/test-video-model",
            video_create_optional_request_params={"image_url": "https://x/cat.jpg"},
            litellm_params=MagicMock(),
            headers={},
        )
        assert body == {"prompt": "a cat blinking", "image_url": "https://x/cat.jpg"}
        assert files == []
        assert url == "https://queue.fal.run/stub/test-video-model"

    def test_request_includes_extra_fal_params(self):
        body, _, _ = self.config.transform_video_create_request(
            model="fal_ai/stub/test-video-model",
            prompt="x",
            api_base="https://queue.fal.run/stub/test-video-model",
            video_create_optional_request_params={
                "image_url": "https://x/y.jpg",
                "resolution": "1080p",
                "duration": "5",
                "aspect_ratio": "16:9",
                "generate_audio": False,
                "seed": 42,
            },
            litellm_params=MagicMock(),
            headers={},
        )
        assert body["resolution"] == "1080p"
        assert body["duration"] == "5"
        assert body["aspect_ratio"] == "16:9"
        assert body["generate_audio"] is False
        assert body["seed"] == 42

    def test_request_raises_without_image_url(self):
        with pytest.raises(ValueError, match="image_url"):
            self.config.transform_video_create_request(
                model="fal_ai/stub/test-video-model",
                prompt="x",
                api_base="https://queue.fal.run/stub/test-video-model",
                video_create_optional_request_params={},
                litellm_params=MagicMock(),
                headers={},
            )


# ------------------------------------------------------------- response


def _build_response(json_body, headers=None):
    raw = MagicMock(spec=httpx.Response)
    raw.json.return_value = json_body
    raw.headers = httpx.Headers(headers or {})
    raw.status_code = 200
    return raw


class TestCreateResponse:
    def setup_method(self):
        self.config = _StubConfig()

    def test_parses_request_id_and_status(self):
        raw = _build_response({"request_id": "abc-123", "status": "IN_QUEUE"})
        obj = self.config.transform_video_create_response(
            model="fal_ai/stub/test-video-model",
            raw_response=raw,
            logging_obj=MagicMock(),
            custom_llm_provider=None,
        )
        # No custom_llm_provider → id remains raw request_id (not encoded)
        assert obj.id == "abc-123"
        assert obj.status == "queued"
        assert obj.model == "fal_ai/stub/test-video-model"

    def test_encodes_id_with_provider_when_given(self):
        raw = _build_response({"request_id": "abc-123", "status": "IN_QUEUE"})
        obj = self.config.transform_video_create_response(
            model="fal_ai/stub/test-video-model",
            raw_response=raw,
            logging_obj=MagicMock(),
            custom_llm_provider="fal_ai",
        )
        # encode_video_id_with_provider produces "video_<base64>"
        assert obj.id.startswith("video_")
        assert obj.id != "abc-123"

    def test_captures_billable_units_header(self):
        raw = _build_response(
            {"request_id": "x", "status": "COMPLETED"},
            headers={"x-fal-billable-units": "5.0"},
        )
        obj = self.config.transform_video_create_response(
            model="fal_ai/stub/test-video-model",
            raw_response=raw,
            logging_obj=MagicMock(),
        )
        assert obj._hidden_params.get("fal_billable_units") == 5.0

    def test_missing_header_leaves_hidden_params_clean(self):
        raw = _build_response({"request_id": "x", "status": "IN_QUEUE"})
        obj = self.config.transform_video_create_response(
            model="fal_ai/stub/test-video-model",
            raw_response=raw,
            logging_obj=MagicMock(),
        )
        assert "fal_billable_units" not in (obj._hidden_params or {})


# --------------------------------------------------------- status mapping


class TestStatusMapping:
    def setup_method(self):
        self.config = _StubConfig()

    @pytest.mark.parametrize(
        "fal_status,expected",
        [
            ("IN_QUEUE", "queued"),
            ("in_queue", "queued"),
            ("IN_PROGRESS", "in_progress"),
            ("COMPLETED", "completed"),
            ("FAILED", "failed"),
            ("ERROR", "failed"),
            ("CANCELLED", "failed"),
            (None, "queued"),
            ("", "queued"),
            ("UNKNOWN_STATE", "queued"),
        ],
    )
    def test_map_fal_status(self, fal_status, expected):
        assert self.config._map_fal_status(fal_status) == expected


# --------------------------------------------------------- status retrieve


class TestBaseAppId:
    """Fal queue API uses {owner}/{alias} for status/result URLs, not the
    full submit path. Collapse to 2-segment base."""

    def setup_method(self):
        self.config = _StubConfig()

    @pytest.mark.parametrize(
        "slug,expected",
        [
            ("bytedance/seedance-2.0", "bytedance/seedance-2.0"),
            (
                "bytedance/seedance-2.0/fast/image-to-video",
                "bytedance/seedance-2.0",
            ),
            (
                "bytedance/seedance-2.0/image-to-video",
                "bytedance/seedance-2.0",
            ),
            ("fal-ai/clarity-upscaler", "fal-ai/clarity-upscaler"),
            ("openai/gpt-image-2/edit", "openai/gpt-image-2"),
            ("bare", "bare"),
            ("", ""),
            ("/extra/leading/", "extra/leading"),
        ],
    )
    def test_collapse(self, slug, expected):
        assert self.config._base_app_id(slug) == expected


class TestStatusRetrieve:
    def setup_method(self):
        self.config = _StubConfig()

    def test_status_url_uses_encoded_model(self):
        # Simulate the post-create id (encoded with provider+model)
        from litellm.types.videos.utils import encode_video_id_with_provider

        encoded_id = encode_video_id_with_provider(
            "abc-123", "fal_ai", "fal_ai/stub/test-video-model"
        )
        url, params = self.config.transform_video_status_retrieve_request(
            video_id=encoded_id,
            api_base="https://queue.fal.run",
            litellm_params=MagicMock(),
            headers={},
        )
        assert url == "https://queue.fal.run/stub/test-video-model/requests/abc-123/status"
        assert params == {}

    def test_status_url_handles_full_create_url_as_api_base(self):
        # Regression: LiteLLM's video handler echoes the full create URL
        # back as api_base for status/content requests. We must strip the
        # path to scheme+host or the endpoint gets doubled and Fal returns
        # 405 with empty body, which then crashes raw_response.json().
        from litellm.types.videos.utils import encode_video_id_with_provider

        encoded_id = encode_video_id_with_provider(
            "abc-123", "fal_ai", "fal_ai/stub/test-video-model"
        )
        url, _ = self.config.transform_video_status_retrieve_request(
            video_id=encoded_id,
            api_base="https://queue.fal.run/stub/test-video-model",
            litellm_params=MagicMock(),
            headers={},
        )
        assert url == (
            "https://queue.fal.run/stub/test-video-model/requests/abc-123/status"
        )
        assert "stub/test-video-model/stub/test-video-model" not in url

    def test_status_response_progress_from_queue_position(self):
        raw = _build_response(
            {"request_id": "abc", "status": "IN_QUEUE", "queue_position": 3}
        )
        obj = self.config.transform_video_status_retrieve_response(
            raw_response=raw, logging_obj=MagicMock()
        )
        assert obj.status == "queued"
        assert obj.progress == 97


# --------------------------------------------------------- content fetch


class TestContentRequest:
    def setup_method(self):
        self.config = _StubConfig()

    def test_content_url_is_request_result_endpoint(self):
        from litellm.types.videos.utils import encode_video_id_with_provider

        encoded_id = encode_video_id_with_provider(
            "abc-123", "fal_ai", "fal_ai/stub/test-video-model"
        )
        url, params = self.config.transform_video_content_request(
            video_id=encoded_id,
            api_base="https://queue.fal.run",
            litellm_params=MagicMock(),
            headers={},
        )
        assert url == "https://queue.fal.run/stub/test-video-model/requests/abc-123"
        assert params == {}

    def test_content_url_handles_full_create_url_as_api_base(self):
        # Same regression as status: api_base may carry the create URL.
        from litellm.types.videos.utils import encode_video_id_with_provider

        encoded_id = encode_video_id_with_provider(
            "abc-123", "fal_ai", "fal_ai/stub/test-video-model"
        )
        url, _ = self.config.transform_video_content_request(
            video_id=encoded_id,
            api_base="https://queue.fal.run/stub/test-video-model",
            litellm_params=MagicMock(),
            headers={},
        )
        assert url == "https://queue.fal.run/stub/test-video-model/requests/abc-123"
        assert "stub/test-video-model/stub/test-video-model" not in url

    def test_extract_video_url(self):
        url = self.config._extract_video_url(
            {"video": {"url": "https://cdn/x.mp4", "content_type": "video/mp4"}}
        )
        assert url == "https://cdn/x.mp4"

    def test_extract_video_url_missing_raises(self):
        with pytest.raises(ValueError, match="Video URL not found"):
            self.config._extract_video_url({})


# ----------------------------------------------------- unsupported methods


class TestUnsupported:
    def setup_method(self):
        self.config = _StubConfig()

    def test_remix_raises(self):
        with pytest.raises(NotImplementedError, match="remix"):
            self.config.transform_video_remix_request(
                video_id="x",
                prompt="y",
                api_base="z",
                litellm_params=MagicMock(),
                headers={},
            )

    def test_list_raises(self):
        with pytest.raises(NotImplementedError, match="list"):
            self.config.transform_video_list_request(
                api_base="z", litellm_params=MagicMock(), headers={}
            )
