import base64
from io import BytesIO
from typing import Dict
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.fal_ai.image_edit.transformation import FalAIGptImage2EditConfig


class TestFalAIGptImage2EditTransformation:
    def setup_method(self) -> None:
        self.config = FalAIGptImage2EditConfig()
        self.model = "fal_ai/openai/gpt-image-2/edit"
        self.prompt = "Replace the background with a rainy Tokyo street."
        self.logging_obj = MagicMock()

    def test_get_complete_url(self) -> None:
        url = self.config.get_complete_url(
            model=self.model, api_base=None, litellm_params={}
        )
        assert url == "https://fal.run/openai/gpt-image-2/edit"

    def test_validate_environment_uses_fal_key(self, monkeypatch) -> None:
        monkeypatch.setenv("FAL_AI_API_KEY", "test-key")
        headers = self.config.validate_environment(
            headers={}, model=self.model, api_key=None
        )
        assert headers["Authorization"] == "Key test-key"
        assert headers["Content-Type"] == "application/json"

    def test_use_multipart_form_data_returns_false(self) -> None:
        assert self.config.use_multipart_form_data() is False

    def test_map_openai_params(self) -> None:
        optional_params: Dict[str, object] = {
            "n": 2,
            "size": "1024x1024",
            "quality": "high",
            "response_format": "png",
        }
        mapped = self.config.map_openai_params(
            image_edit_optional_params=optional_params,  # type: ignore[arg-type]
            model=self.model,
            drop_params=False,
        )
        assert mapped["num_images"] == 2
        assert mapped["image_size"] == "square_hd"
        assert mapped["quality"] == "high"
        assert mapped["output_format"] == "png"

    def test_map_openai_params_unknown_size_returns_dict(self) -> None:
        mapped = self.config.map_openai_params(
            image_edit_optional_params={"size": "1500x900"},  # type: ignore[arg-type]
            model=self.model,
            drop_params=False,
        )
        assert mapped["image_size"] == {"width": 1500, "height": 900}

    def test_map_openai_params_passes_auto_through(self) -> None:
        mapped = self.config.map_openai_params(
            image_edit_optional_params={"size": "auto"},  # type: ignore[arg-type]
            model=self.model,
            drop_params=False,
        )
        assert mapped["image_size"] == "auto"

    def test_transform_image_edit_request_single_image(self) -> None:
        image_bytes = b"fake_image_data"
        image = BytesIO(image_bytes)

        request_body, files = self.config.transform_image_edit_request(
            model=self.model,
            prompt=self.prompt,
            image=image,
            image_edit_optional_request_params={"quality": "low"},
            litellm_params=MagicMock(),
            headers={},
        )

        assert files == []
        assert request_body["prompt"] == self.prompt
        assert request_body["quality"] == "low"
        assert len(request_body["image_urls"]) == 1
        assert request_body["image_urls"][0].startswith("data:")
        # Decode and confirm bytes round-trip
        encoded = request_body["image_urls"][0].split("base64,", 1)[1]
        assert base64.b64decode(encoded) == image_bytes

    def test_transform_image_edit_request_multiple_images(self) -> None:
        image_one = BytesIO(b"image_one")
        image_two = BytesIO(b"image_two")

        request_body, _ = self.config.transform_image_edit_request(
            model=self.model,
            prompt=self.prompt,
            image=[image_one, image_two],
            image_edit_optional_request_params={},
            litellm_params=MagicMock(),
            headers={},
        )

        assert len(request_body["image_urls"]) == 2
        assert all(u.startswith("data:") for u in request_body["image_urls"])

    def test_transform_image_edit_request_passes_url_through(self) -> None:
        external_url = "https://example.com/photo.png"

        request_body, _ = self.config.transform_image_edit_request(
            model=self.model,
            prompt=self.prompt,
            image=external_url,
            image_edit_optional_request_params={},
            litellm_params=MagicMock(),
            headers={},
        )

        assert request_body["image_urls"] == [external_url]

    def test_transform_image_edit_request_requires_image(self) -> None:
        with pytest.raises(ValueError, match="requires at least one input image"):
            self.config.transform_image_edit_request(
                model=self.model,
                prompt=self.prompt,
                image=None,
                image_edit_optional_request_params={},
                litellm_params=MagicMock(),
                headers={},
            )

    def test_transform_image_edit_response_stamps_size_and_quality(self) -> None:
        raw = httpx.Response(
            status_code=200,
            json={
                "images": [
                    {
                        "url": "https://v3b.fal.media/files/x.png",
                        "width": 1024,
                        "height": 1024,
                        "content_type": "image/png",
                    }
                ]
            },
        )
        response = self.config.transform_image_edit_response(
            model=self.model, raw_response=raw, logging_obj=self.logging_obj
        )
        assert len(response.data) == 1
        assert response.data[0].url == "https://v3b.fal.media/files/x.png"
        assert response.size == "1024-x-1024"
        assert response.quality == "high"


@pytest.mark.parametrize(
    "quality,size,expected_cost",
    [
        ("high", "1024-x-1024", 0.22),
        ("medium", "1024-x-1024", 0.06),
        ("low", "1024-x-1024", 0.01),
        ("high", "3840-x-2160", 0.41),
        ("low", "1024-x-768", 0.01),
        ("medium", "1920-x-1080", 0.04),
    ],
)
def test_fal_ai_gpt_image_2_edit_tiered_cost_lookup(
    quality, size, expected_cost, monkeypatch
):
    """Edit endpoint pricing keys (`{quality}/fal_ai/{size}/openai/gpt-image-2/edit`)
    resolve through the same composite-key chain as text-to-image."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    # Re-import to pick up forced-local cost map
    import importlib
    import litellm

    importlib.reload(litellm)

    from litellm.litellm_core_utils.llm_cost_calc.utils import CostCalculatorUtils
    from litellm.types.utils import ImageObject, ImageResponse

    response = ImageResponse(
        data=[ImageObject(url="https://example.com/img.png", b64_json=None)]
    )
    response.size = size
    response.quality = quality

    cost = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="fal_ai/openai/gpt-image-2/edit",
        completion_response=response,
        custom_llm_provider="fal_ai",
        quality=quality,
        size=size,
        n=1,
    )
    assert abs(cost - expected_cost) < 1e-9, (
        f"expected ${expected_cost} for {quality}/{size}, got ${cost}"
    )
