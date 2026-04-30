"""
Tests for the shared FalAIImageEditConfig base class.

Per-subclass behavior (param mapping, response shape) is tested in each
config's own file. This file exercises the generic plumbing that every
fal_ai image-edit config inherits.
"""
import base64
from io import BytesIO
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.fal_ai.image_edit.base import FalAIImageEditConfig


class _SingleStubConfig(FalAIImageEditConfig):
    """Minimal subclass for single-image, singular-response Fal endpoints."""

    EDIT_ENDPOINT = "fal-ai/_stub/single"
    SUPPORTED_PARAMS = []
    BODY_IMAGE_KEY = "image_url"
    RESPONSE_IMAGE_KEY = "image"


class _SingleStampConfig(FalAIImageEditConfig):
    """Single-image variant that stamps output size for per-pixel pricing."""

    EDIT_ENDPOINT = "fal-ai/_stub/stamp"
    BODY_IMAGE_KEY = "image_url"
    RESPONSE_IMAGE_KEY = "image"
    STAMP_OUTPUT_SIZE = True


class _MultiStubConfig(FalAIImageEditConfig):
    """Multi-image input/output, like nano-banana-pro/edit."""

    EDIT_ENDPOINT = "fal-ai/_stub/multi"
    SUPPORTED_PARAMS = ["n", "size"]
    PARAM_MAPPING = {"n": "num_images", "size": "image_size"}
    BODY_IMAGE_KEY = "image_urls"
    RESPONSE_IMAGE_KEY = "images"
    SUPPORTS_MULTI_IMAGE = True


class _NoPromptConfig(FalAIImageEditConfig):
    """Pure image→image like birefnet."""

    EDIT_ENDPOINT = "fal-ai/_stub/no-prompt"
    ACCEPTS_PROMPT = False


# ---------------------------------------------------------------- env+url


class TestEnvAndUrl:
    def setup_method(self):
        self.config = _SingleStubConfig()

    def test_get_complete_url(self):
        url = self.config.get_complete_url(model="x", api_base=None, litellm_params={})
        assert url == "https://fal.run/fal-ai/_stub/single"

    def test_get_complete_url_strips_trailing_slash(self):
        url = self.config.get_complete_url(
            model="x",
            api_base="https://custom.fal.run/",
            litellm_params={},
        )
        assert url == "https://custom.fal.run/fal-ai/_stub/single"

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

    def test_use_multipart_form_data_returns_false(self):
        assert self.config.use_multipart_form_data() is False


# --------------------------------------------------------------- request


class TestRequestBuild:
    def test_single_image_url_passthrough(self):
        config = _SingleStubConfig()
        body, files = config.transform_image_edit_request(
            model="x",
            prompt="hello",
            image="https://example.com/in.png",
            image_edit_optional_request_params={},
            litellm_params=MagicMock(),
            headers={},
        )
        assert files == []
        assert body == {"image_url": "https://example.com/in.png", "prompt": "hello"}

    def test_single_image_bytes_get_base64_encoded(self):
        config = _SingleStubConfig()
        raw_bytes = b"\x00\x11\x22stub"
        body, _ = config.transform_image_edit_request(
            model="x",
            prompt=None,
            image=BytesIO(raw_bytes),
            image_edit_optional_request_params={},
            litellm_params=MagicMock(),
            headers={},
        )
        assert body["image_url"].startswith("data:")
        encoded = body["image_url"].split("base64,", 1)[1]
        assert base64.b64decode(encoded) == raw_bytes

    def test_single_image_rejects_multi_image_input(self):
        config = _SingleStubConfig()
        with pytest.raises(ValueError, match="exactly one input image"):
            config.transform_image_edit_request(
                model="x",
                prompt=None,
                image=[BytesIO(b"a"), BytesIO(b"b")],
                image_edit_optional_request_params={},
                litellm_params=MagicMock(),
                headers={},
            )

    def test_single_image_requires_image(self):
        config = _SingleStubConfig()
        with pytest.raises(ValueError, match="requires exactly one input image"):
            config.transform_image_edit_request(
                model="x",
                prompt=None,
                image=None,
                image_edit_optional_request_params={},
                litellm_params=MagicMock(),
                headers={},
            )

    def test_multi_image_packs_list(self):
        config = _MultiStubConfig()
        body, _ = config.transform_image_edit_request(
            model="x",
            prompt="combine these",
            image=["https://a.png", "https://b.png"],
            image_edit_optional_request_params={"num_images": 2},
            litellm_params=MagicMock(),
            headers={},
        )
        assert body["image_urls"] == ["https://a.png", "https://b.png"]
        assert body["prompt"] == "combine these"
        assert body["num_images"] == 2

    def test_multi_image_requires_at_least_one(self):
        config = _MultiStubConfig()
        with pytest.raises(ValueError, match="requires at least one input image"):
            config.transform_image_edit_request(
                model="x",
                prompt="hi",
                image=None,
                image_edit_optional_request_params={},
                litellm_params=MagicMock(),
                headers={},
            )

    def test_no_prompt_config_drops_prompt(self):
        config = _NoPromptConfig()
        body, _ = config.transform_image_edit_request(
            model="x",
            prompt="this gets dropped",
            image="https://a.png",
            image_edit_optional_request_params={},
            litellm_params=MagicMock(),
            headers={},
        )
        assert "prompt" not in body

    def test_extra_body_params_pass_through(self):
        config = _SingleStubConfig()
        body, _ = config.transform_image_edit_request(
            model="x",
            prompt=None,
            image="https://a.png",
            image_edit_optional_request_params={
                "creativity": 0.5,
                "upscale_factor": 4,
            },
            litellm_params=MagicMock(),
            headers={},
        )
        assert body["creativity"] == 0.5
        assert body["upscale_factor"] == 4


# -------------------------------------------------------------- map params


class TestMapOpenAIParams:
    def test_drops_unsupported_params(self):
        config = _SingleStubConfig()  # SUPPORTED_PARAMS = []
        mapped = config.map_openai_params(
            image_edit_optional_params={"n": 1, "size": "1024x1024"},  # type: ignore[arg-type]
            model="x",
            drop_params=True,
        )
        assert mapped == {}

    def test_renames_via_param_mapping(self):
        config = _MultiStubConfig()
        mapped = config.map_openai_params(
            image_edit_optional_params={"n": 2, "size": "1024x1024"},  # type: ignore[arg-type]
            model="x",
            drop_params=True,
        )
        assert mapped == {"num_images": 2, "image_size": "1024x1024"}


# -------------------------------------------------------------- response


class TestResponseTransform:
    def test_singular_image_dict_parsed(self):
        config = _SingleStubConfig()
        raw = httpx.Response(
            status_code=200,
            json={
                "image": {
                    "url": "https://out.png",
                    "width": 2048,
                    "height": 2048,
                }
            },
        )
        response = config.transform_image_edit_response(
            model="fal_ai/fal-ai/_stub/single",
            raw_response=raw,
            logging_obj=MagicMock(),
        )
        assert response.model == "fal_ai/fal-ai/_stub/single"
        assert len(response.data) == 1
        assert response.data[0].url == "https://out.png"

    def test_singular_image_skips_size_stamp_when_disabled(self):
        config = _SingleStubConfig()
        raw = httpx.Response(
            status_code=200,
            json={"image": {"url": "https://out.png", "width": 1024, "height": 1024}},
        )
        response = config.transform_image_edit_response(
            model="x", raw_response=raw, logging_obj=MagicMock()
        )
        assert response.size is None

    def test_singular_image_stamps_size_when_enabled(self):
        config = _SingleStampConfig()
        raw = httpx.Response(
            status_code=200,
            json={"image": {"url": "https://out.png", "width": 4096, "height": 2048}},
        )
        response = config.transform_image_edit_response(
            model="x", raw_response=raw, logging_obj=MagicMock()
        )
        assert response.size == "4096-x-2048"

    def test_list_response_parses_each_image(self):
        config = _MultiStubConfig()
        raw = httpx.Response(
            status_code=200,
            json={
                "images": [
                    {"url": "https://a.png", "width": 1024, "height": 1024},
                    {"url": "https://b.png"},
                ]
            },
        )
        response = config.transform_image_edit_response(
            model="x", raw_response=raw, logging_obj=MagicMock()
        )
        assert len(response.data) == 2
        assert response.data[0].url == "https://a.png"
        assert response.data[1].url == "https://b.png"

    def test_string_url_in_list(self):
        config = _MultiStubConfig()
        raw = httpx.Response(
            status_code=200,
            json={"images": ["https://just-a-url.png"]},
        )
        response = config.transform_image_edit_response(
            model="x", raw_response=raw, logging_obj=MagicMock()
        )
        assert len(response.data) == 1
        assert response.data[0].url == "https://just-a-url.png"

    def test_model_field_always_stamped(self):
        """
        Regression test: pre-base-class versions of these configs forgot to
        set ``model_response.model``, causing
        ``cannot override response model; missing 'model' attribute``
        in common_request_processing.py.
        """
        config = _SingleStubConfig()
        raw = httpx.Response(
            status_code=200, json={"image": {"url": "https://x.png"}}
        )
        response = config.transform_image_edit_response(
            model="fal_ai/fal-ai/_stub/single",
            raw_response=raw,
            logging_obj=MagicMock(),
        )
        assert response.model == "fal_ai/fal-ai/_stub/single"
