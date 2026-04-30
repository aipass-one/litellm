import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm import aimage_generation


@pytest.mark.parametrize(
    "model,expected_endpoint",
    [
        ("fal_ai/fal-ai/flux-pro/v1.1-ultra", "fal-ai/flux-pro/v1.1-ultra"),
        (
            "fal_ai/fal-ai/stable-diffusion-v35-medium",
            "fal-ai/stable-diffusion-v35-medium",
        ),
        ("fal_ai/openai/gpt-image-2", "openai/gpt-image-2"),
    ],
)
@pytest.mark.asyncio
async def test_fal_ai_image_generation_basic(model, expected_endpoint):
    """
    Test that fal_ai image generation constructs correct request body and URL.

    Validates:
    - Correct API endpoint URL construction
    - Proper request body format with prompt
    - Correct Authorization header format
    """
    captured_url = None
    captured_json_data = None
    captured_headers = None

    def capture_post_call(*args, **kwargs):
        nonlocal captured_url, captured_json_data, captured_headers

        captured_url = args[0] if args else kwargs.get("url")
        captured_json_data = kwargs.get("json")
        captured_headers = kwargs.get("headers")

        # Mock response with fal.ai format
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "images": [
                {
                    "url": "https://example.com/generated-image.png",
                    "width": 1024,
                    "height": 768,
                    "content_type": "image/jpeg",
                }
            ],
            "seed": 42,
        }

        return mock_response

    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post") as mock_post:
        mock_post.side_effect = capture_post_call

        test_api_key = "test-fal-ai-key-12345"
        test_prompt = "A cute baby sea otter"

        response = await aimage_generation(
            model=model,
            prompt=test_prompt,
            api_key=test_api_key,
        )

        # Validate response
        assert response is not None
        assert hasattr(response, "data")
        assert response.data is not None
        assert len(response.data) > 0

        # Validate URL
        assert captured_url is not None
        assert "fal.run" in captured_url
        assert expected_endpoint in captured_url
        print(f"Validated URL: {captured_url}")

        # Validate headers
        assert captured_headers is not None
        assert "Authorization" in captured_headers
        assert captured_headers["Authorization"] == f"Key {test_api_key}"
        print(f"Validated headers: {captured_headers}")

        # Validate request body
        assert captured_json_data is not None
        assert captured_json_data["prompt"] == test_prompt
        print(f"Validated request body: {captured_json_data}")


@pytest.mark.asyncio
async def test_fal_ai_gpt_image_2_param_mapping():
    """gpt-image-2 maps OpenAI params (n, size, quality, response_format) to
    Fal's request shape (num_images, image_size, quality, output_format)."""
    captured_json_data = None

    def capture_post_call(*args, **kwargs):
        nonlocal captured_json_data
        captured_json_data = kwargs.get("json")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "images": [
                {
                    "url": "https://example.com/img.png",
                    "width": 1024,
                    "height": 1024,
                    "content_type": "image/png",
                }
            ]
        }
        return mock_response

    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post") as mock_post:
        mock_post.side_effect = capture_post_call

        await aimage_generation(
            model="fal_ai/openai/gpt-image-2",
            prompt="a teacup",
            api_key="test-key",
            n=2,
            size="1024x1024",
            quality="high",
            response_format="png",
        )

        assert captured_json_data["prompt"] == "a teacup"
        assert captured_json_data["num_images"] == 2
        # 1024x1024 maps to the square_hd preset
        assert captured_json_data["image_size"] == "square_hd"
        assert captured_json_data["quality"] == "high"
        assert captured_json_data["output_format"] == "png"


@pytest.mark.asyncio
async def test_fal_ai_gpt_image_2_response_stamps_size_and_quality():
    """The response transformer must populate ImageResponse.size and .quality
    so the cost calculator's composite-key lookup can find the right entry."""

    def capture_post_call(*args, **kwargs):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "images": [
                {
                    "url": "https://example.com/img.png",
                    "width": 1920,
                    "height": 1080,
                    "content_type": "image/png",
                }
            ]
        }
        return mock_response

    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post") as mock_post:
        mock_post.side_effect = capture_post_call

        response = await aimage_generation(
            model="fal_ai/openai/gpt-image-2",
            prompt="a teacup",
            api_key="test-key",
            quality="medium",
        )

        assert response.size == "1920-x-1080"
        assert response.quality == "medium"


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
def test_fal_ai_gpt_image_2_tiered_cost_lookup(quality, size, expected_cost):
    """The {quality}/{size}/{model} composite key resolves to the matching
    pricing entry in model_prices_and_context_window.json."""
    from litellm.litellm_core_utils.llm_cost_calc.utils import (
        CostCalculatorUtils,
    )
    from litellm.types.utils import ImageObject, ImageResponse

    response = ImageResponse(
        data=[ImageObject(url="https://example.com/img.png", b64_json=None)]
    )
    response.size = size
    response.quality = quality

    cost = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="fal_ai/openai/gpt-image-2",
        completion_response=response,
        custom_llm_provider="fal_ai",
        quality=quality,
        size=size,
        n=1,
    )

    assert abs(cost - expected_cost) < 1e-9, (
        f"expected ${expected_cost} for quality={quality} size={size}, got ${cost}"
    )
