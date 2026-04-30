import base64
from io import BytesIO
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.fal_ai.image_edit.clarity_upscaler_transformation import (
    FalAIClarityUpscalerEditConfig,
)


class TestFalAIClarityUpscalerEditTransformation:
    def setup_method(self) -> None:
        self.config = FalAIClarityUpscalerEditConfig()
        self.model = "fal_ai/fal-ai/clarity-upscaler"
        self.logging_obj = MagicMock()

    def test_get_complete_url(self) -> None:
        url = self.config.get_complete_url(
            model=self.model, api_base=None, litellm_params={}
        )
        assert url == "https://fal.run/fal-ai/clarity-upscaler"

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
        # No OpenAI image-edit params apply to clarity-upscaler.
        mapped = self.config.map_openai_params(
            image_edit_optional_params={  # type: ignore[arg-type]
                "n": 1,
                "size": "1024x1024",
                "quality": "high",
                "response_format": "b64_json",
            },
            model=self.model,
            drop_params=True,
        )
        assert mapped == {}

    def test_transform_image_edit_request_single_image(self) -> None:
        image_bytes = b"clarity_upscale_input_bytes"
        image = BytesIO(image_bytes)

        request_body, files = self.config.transform_image_edit_request(
            model=self.model,
            prompt="masterpiece, best quality",
            image=image,
            image_edit_optional_request_params={},
            litellm_params=MagicMock(),
            headers={},
        )

        assert files == []
        assert request_body["prompt"] == "masterpiece, best quality"
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

    def test_transform_image_edit_response_stamps_size(self) -> None:
        raw = httpx.Response(
            status_code=200,
            json={
                "image": {
                    "url": "https://v3b.fal.media/files/upscaled.png",
                    "width": 2048,
                    "height": 2048,
                    "content_type": "image/png",
                },
                "seed": 12345,
                "timings": {},
            },
        )
        response = self.config.transform_image_edit_response(
            model=self.model, raw_response=raw, logging_obj=self.logging_obj
        )
        assert len(response.data) == 1
        assert response.data[0].url == "https://v3b.fal.media/files/upscaled.png"
        assert response.size == "2048-x-2048"

    def test_transform_image_edit_response_no_dims_no_size_stamp(self) -> None:
        raw = httpx.Response(
            status_code=200,
            json={"image": {"url": "https://v3b.fal.media/files/x.png"}},
        )
        response = self.config.transform_image_edit_response(
            model=self.model, raw_response=raw, logging_obj=self.logging_obj
        )
        assert len(response.data) == 1
        # Without dims, leave size unset (cost calc would fall back to default).
        assert response.size in (None, "")


@pytest.mark.parametrize(
    "size,expected_cost",
    [
        ("1024-x-1024", 1024 * 1024 * 3e-8),  # 1 MP → $0.0314...
        ("2048-x-2048", 2048 * 2048 * 3e-8),  # 4 MP → $0.1258...
        ("4096-x-4096", 4096 * 4096 * 3e-8),  # 16 MP → $0.5033...
        ("1920-x-1080", 1920 * 1080 * 3e-8),  # ~2.07 MP → $0.0622...
    ],
)
def test_fal_ai_clarity_upscaler_pixel_cost_lookup(size, expected_cost, monkeypatch):
    """Pixel-based pricing — cost = input_cost_per_pixel × width × height × n.
    The transformation stamps response.size as the OUTPUT dimensions, so the
    cost calc multiplies the per-pixel rate by the output area."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import importlib

    import litellm

    importlib.reload(litellm)

    from litellm.litellm_core_utils.llm_cost_calc.utils import CostCalculatorUtils
    from litellm.types.utils import ImageObject, ImageResponse

    response = ImageResponse(
        data=[ImageObject(url="https://example.com/img.png", b64_json=None)]
    )
    response.size = size

    cost = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="fal_ai/fal-ai/clarity-upscaler",
        completion_response=response,
        custom_llm_provider="fal_ai",
        quality=None,
        size=size,
        n=1,
    )
    assert abs(cost - expected_cost) < 1e-9, (
        f"expected ${expected_cost:.6f} for {size}, got ${cost:.6f}"
    )
