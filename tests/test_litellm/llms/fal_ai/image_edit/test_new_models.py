"""
Lean coverage for the 9 new image_edit configs introduced alongside the
``FalAIImageEditConfig`` base class. Per-config behavior shared with the
base (env validation, image encoding, response parsing) is covered in
``test_base.py``; this file just exercises the model-specific glue
(endpoint URL + flat-cost or per-pixel-cost lookup).
"""
import importlib
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.fal_ai.image_edit import (
    FalAIAuraSREditConfig,
    FalAIBenV2EditConfig,
    FalAIBirefnetV2EditConfig,
    FalAIEsrganEditConfig,
    FalAINanoBanana2EditConfig,
    FalAINanoBananaProEditConfig,
    FalAIRecraftUpscaleCreativeEditConfig,
    FalAIRecraftUpscaleCrispEditConfig,
    FalAITopazUpscaleEditConfig,
)


@pytest.mark.parametrize(
    "config_class,expected_url",
    [
        (FalAIAuraSREditConfig, "https://fal.run/fal-ai/aura-sr"),
        (FalAIEsrganEditConfig, "https://fal.run/fal-ai/esrgan"),
        (FalAITopazUpscaleEditConfig, "https://fal.run/fal-ai/topaz/upscale/image"),
        (
            FalAIRecraftUpscaleCrispEditConfig,
            "https://fal.run/fal-ai/recraft/upscale/crisp",
        ),
        (
            FalAIRecraftUpscaleCreativeEditConfig,
            "https://fal.run/fal-ai/recraft/upscale/creative",
        ),
        (FalAIBirefnetV2EditConfig, "https://fal.run/fal-ai/birefnet/v2"),
        (FalAIBenV2EditConfig, "https://fal.run/fal-ai/ben/v2/image"),
        (
            FalAINanoBananaProEditConfig,
            "https://fal.run/fal-ai/nano-banana-pro/edit",
        ),
        (
            FalAINanoBanana2EditConfig,
            "https://fal.run/fal-ai/nano-banana-2/edit",
        ),
    ],
)
def test_endpoint_url_construction(config_class, expected_url):
    config = config_class()
    url = config.get_complete_url(model="x", api_base=None, litellm_params={})
    assert url == expected_url


@pytest.mark.parametrize(
    "config_class,no_prompt",
    [
        (FalAIAuraSREditConfig, True),
        (FalAIEsrganEditConfig, True),
        (FalAITopazUpscaleEditConfig, True),
        (FalAIRecraftUpscaleCrispEditConfig, True),
        (FalAIRecraftUpscaleCreativeEditConfig, True),
        (FalAIBirefnetV2EditConfig, True),
        (FalAIBenV2EditConfig, True),
        (FalAINanoBananaProEditConfig, False),
        (FalAINanoBanana2EditConfig, False),
    ],
)
def test_prompt_handling_per_model(config_class, no_prompt):
    """
    Upscalers + bg-removal configs drop the prompt (image-only); nano-banana
    configs keep it.
    """
    config = config_class()
    if config.SUPPORTS_MULTI_IMAGE:
        body, _ = config.transform_image_edit_request(
            model="x",
            prompt="hello",
            image=["https://a.png"],
            image_edit_optional_request_params={},
            litellm_params=MagicMock(),
            headers={},
        )
    else:
        body, _ = config.transform_image_edit_request(
            model="x",
            prompt="hello",
            image="https://a.png",
            image_edit_optional_request_params={},
            litellm_params=MagicMock(),
            headers={},
        )

    if no_prompt:
        assert "prompt" not in body
    else:
        assert body.get("prompt") == "hello"


def test_nano_banana_pro_supports_multi_image():
    config = FalAINanoBananaProEditConfig()
    body, _ = config.transform_image_edit_request(
        model="x",
        prompt="merge",
        image=["https://a.png", "https://b.png"],
        image_edit_optional_request_params={"num_images": 2},
        litellm_params=MagicMock(),
        headers={},
    )
    assert body["image_urls"] == ["https://a.png", "https://b.png"]
    assert body["prompt"] == "merge"


def test_nano_banana_pro_response_parses_images_list():
    config = FalAINanoBananaProEditConfig()
    raw = httpx.Response(
        status_code=200,
        json={
            "images": [
                {"url": "https://o1.png"},
                {"url": "https://o2.png"},
            ]
        },
    )
    response = config.transform_image_edit_response(
        model="fal_ai/fal-ai/nano-banana-pro/edit",
        raw_response=raw,
        logging_obj=MagicMock(),
    )
    assert len(response.data) == 2
    assert response.model == "fal_ai/fal-ai/nano-banana-pro/edit"


def test_aura_sr_stamps_output_size_for_pixel_pricing():
    """aura-sr is per-pixel, so size must land on the response."""
    config = FalAIAuraSREditConfig()
    raw = httpx.Response(
        status_code=200,
        json={"image": {"url": "https://o.png", "width": 4096, "height": 4096}},
    )
    response = config.transform_image_edit_response(
        model="fal_ai/fal-ai/aura-sr",
        raw_response=raw,
        logging_obj=MagicMock(),
    )
    assert response.size == "4096-x-4096"


# ------------------------------------------------------------- cost lookup


@pytest.fixture(scope="module")
def litellm_with_local_costs(monkeypatch_module=None):
    """Reload litellm with the local cost map enabled (one-time per module)."""
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    import litellm

    importlib.reload(litellm)
    yield litellm


@pytest.mark.parametrize(
    "model,expected_cost",
    [
        ("fal_ai/fal-ai/esrgan", 0.001),
        ("fal_ai/fal-ai/topaz/upscale/image", 0.05),
        ("fal_ai/fal-ai/recraft/upscale/crisp", 0.04),
        ("fal_ai/fal-ai/recraft/upscale/creative", 0.04),
        ("fal_ai/fal-ai/birefnet/v2", 0.005),
        ("fal_ai/fal-ai/ben/v2/image", 0.005),
        ("fal_ai/fal-ai/nano-banana-pro/edit", 0.04),
        ("fal_ai/fal-ai/nano-banana-2/edit", 0.04),
    ],
)
def test_flat_per_image_cost(litellm_with_local_costs, model, expected_cost):
    """Flat-priced edit models read ``output_cost_per_image`` from JSON."""
    from litellm.litellm_core_utils.llm_cost_calc.utils import CostCalculatorUtils
    from litellm.types.utils import ImageObject, ImageResponse

    response = ImageResponse(
        data=[ImageObject(url="https://example.com/img.png", b64_json=None)]
    )
    cost = CostCalculatorUtils.route_image_generation_cost_calculator(
        model=model,
        completion_response=response,
        custom_llm_provider="fal_ai",
        quality=None,
        size=None,
        n=1,
    )
    assert abs(cost - expected_cost) < 1e-9, (
        f"{model}: expected ${expected_cost}, got ${cost}"
    )


@pytest.mark.parametrize(
    "size,expected_cost",
    [
        # input_cost_per_pixel = 1e-8, so 4MP = $0.04, 16MP = $0.16
        ("1024-x-1024", 1024 * 1024 * 1e-8),
        ("2048-x-2048", 2048 * 2048 * 1e-8),
        ("4096-x-4096", 4096 * 4096 * 1e-8),
    ],
)
def test_aura_sr_pixel_cost_lookup(litellm_with_local_costs, size, expected_cost):
    """aura-sr uses input_cost_per_pixel — width × height × rate."""
    from litellm.litellm_core_utils.llm_cost_calc.utils import CostCalculatorUtils
    from litellm.types.utils import ImageObject, ImageResponse

    response = ImageResponse(
        data=[ImageObject(url="https://example.com/img.png", b64_json=None)],
    )
    response.size = size  # the transformation stamps this for per-pixel pricing

    cost = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="fal_ai/fal-ai/aura-sr",
        completion_response=response,
        custom_llm_provider="fal_ai",
        quality=None,
        size=size,
        n=1,
    )
    assert abs(cost - expected_cost) < 1e-9
