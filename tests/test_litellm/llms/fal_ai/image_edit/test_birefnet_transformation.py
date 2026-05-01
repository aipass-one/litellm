import base64
from io import BytesIO
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.fal_ai.image_edit.birefnet_transformation import (
    FalAIBirefnetEditConfig,
)


class TestFalAIBirefnetEditTransformation:
    def setup_method(self) -> None:
        self.config = FalAIBirefnetEditConfig()
        self.model = "fal_ai/fal-ai/birefnet"
        self.logging_obj = MagicMock()

    def test_get_complete_url(self) -> None:
        url = self.config.get_complete_url(
            model=self.model, api_base=None, litellm_params={}
        )
        assert url == "https://fal.run/fal-ai/birefnet"

    def test_validate_environment_uses_fal_key(self, monkeypatch) -> None:
        monkeypatch.setenv("FAL_AI_API_KEY", "test-key")
        headers = self.config.validate_environment(
            headers={}, model=self.model, api_key=None
        )
        assert headers["Authorization"] == "Key test-key"
        assert headers["Content-Type"] == "application/json"

    def test_validate_environment_raises_without_key(self, monkeypatch) -> None:
        monkeypatch.delenv("FAL_AI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="FAL_AI_API_KEY"):
            self.config.validate_environment(
                headers={}, model=self.model, api_key=None
            )

    def test_use_multipart_form_data_returns_false(self) -> None:
        assert self.config.use_multipart_form_data() is False

    def test_map_openai_params_drops_all(self) -> None:
        mapped = self.config.map_openai_params(
            image_edit_optional_params={  # type: ignore[arg-type]
                "n": 1,
                "size": "1024x1024",
                "quality": "high",
                "response_format": "b64_json",
                "background": "transparent",
            },
            model=self.model,
            drop_params=True,
        )
        assert mapped == {}

    def test_transform_image_edit_request_single_image(self) -> None:
        image_bytes = b"birefnet_input_bytes"
        image = BytesIO(image_bytes)

        request_body, files = self.config.transform_image_edit_request(
            model=self.model,
            prompt=None,
            image=image,
            image_edit_optional_request_params={},
            litellm_params=MagicMock(),
            headers={},
        )

        assert files == []
        # birefnet doesn't take a prompt — segmentation is image-driven.
        assert "prompt" not in request_body
        assert request_body["image_url"].startswith("data:")
        encoded = request_body["image_url"].split("base64,", 1)[1]
        assert base64.b64decode(encoded) == image_bytes

    def test_transform_image_edit_request_passes_url_through(self) -> None:
        external_url = "https://example.com/input.png"
        request_body, _ = self.config.transform_image_edit_request(
            model=self.model,
            prompt=None,
            image=external_url,
            image_edit_optional_request_params={},
            litellm_params=MagicMock(),
            headers={},
        )
        assert request_body["image_url"] == external_url

    def test_transform_image_edit_request_drops_prompt(self) -> None:
        # birefnet ignores prompts — confirm we don't accidentally include one.
        request_body, _ = self.config.transform_image_edit_request(
            model=self.model,
            prompt="this should be ignored",
            image="https://example.com/input.png",
            image_edit_optional_request_params={},
            litellm_params=MagicMock(),
            headers={},
        )
        assert "prompt" not in request_body

    def test_transform_image_edit_request_rejects_multi_image(self) -> None:
        with pytest.raises(ValueError, match="exactly one input image"):
            self.config.transform_image_edit_request(
                model=self.model,
                prompt=None,
                image=[BytesIO(b"a"), BytesIO(b"b")],
                image_edit_optional_request_params={},
                litellm_params=MagicMock(),
                headers={},
            )

    def test_transform_image_edit_request_requires_image(self) -> None:
        with pytest.raises(ValueError, match="requires exactly one input image"):
            self.config.transform_image_edit_request(
                model=self.model,
                prompt=None,
                image=None,
                image_edit_optional_request_params={},
                litellm_params=MagicMock(),
                headers={},
            )

    def test_transform_image_edit_response_returns_image_url(self) -> None:
        raw = httpx.Response(
            status_code=200,
            json={
                "image": {
                    "url": "https://v3b.fal.media/files/cutout.png",
                    "width": 1024,
                    "height": 1024,
                    "content_type": "image/png",
                }
            },
        )
        response = self.config.transform_image_edit_response(
            model=self.model, raw_response=raw, logging_obj=self.logging_obj
        )
        assert len(response.data) == 1
        assert response.data[0].url == "https://v3b.fal.media/files/cutout.png"


def test_fal_ai_birefnet_flat_cost_lookup(monkeypatch):
    """birefnet pricing is flat per-image (output_cost_per_image)."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import importlib

    import litellm

    importlib.reload(litellm)

    from litellm.litellm_core_utils.llm_cost_calc.utils import CostCalculatorUtils
    from litellm.types.utils import ImageObject, ImageResponse

    response = ImageResponse(
        data=[ImageObject(url="https://example.com/img.png", b64_json=None)]
    )

    cost = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="fal_ai/fal-ai/birefnet",
        completion_response=response,
        custom_llm_provider="fal_ai",
        quality=None,
        size=None,
        n=1,
    )
    # JSON entry: output_cost_per_image = 0.004 (~$0.00111/sec × ~3s avg)
    assert abs(cost - 0.004) < 1e-9, f"expected $0.004, got ${cost}"
