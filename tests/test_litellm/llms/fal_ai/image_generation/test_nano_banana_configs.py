"""
Tests for the two new Fal AI image-generation configs (nano-banana-pro,
nano-banana-2). These extend the existing ``FalAIBaseConfig`` and add
nothing exotic — just endpoint + flat-per-image cost lookup.
"""
import importlib
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.fal_ai.image_generation import (
    FalAINanoBanana2Config,
    FalAINanoBananaProConfig,
    get_fal_ai_image_generation_config,
)


@pytest.mark.parametrize(
    "model,expected_class",
    [
        ("fal-ai/nano-banana-pro", FalAINanoBananaProConfig),
        ("fal_ai/fal-ai/nano-banana-pro", FalAINanoBananaProConfig),
        ("fal-ai/nano-banana-2", FalAINanoBanana2Config),
        ("fal_ai/fal-ai/nano-banana-2", FalAINanoBanana2Config),
    ],
)
def test_dispatcher_routes_nano_banana(model, expected_class):
    config = get_fal_ai_image_generation_config(model)
    assert isinstance(config, expected_class)


@pytest.mark.parametrize(
    "config_class,expected_url",
    [
        (FalAINanoBananaProConfig, "https://fal.run/fal-ai/nano-banana-pro"),
        (FalAINanoBanana2Config, "https://fal.run/fal-ai/nano-banana-2"),
    ],
)
def test_endpoint_url(config_class, expected_url):
    config = config_class()
    url = config.get_complete_url(
        api_base=None,
        api_key="dummy",
        model="x",
        optional_params={},
        litellm_params={},
    )
    assert url == expected_url


@pytest.mark.parametrize(
    "config_class", [FalAINanoBananaProConfig, FalAINanoBanana2Config]
)
def test_validate_environment_uses_fal_key(config_class, monkeypatch):
    monkeypatch.setenv("FAL_AI_API_KEY", "test-key")
    config = config_class()
    headers = config.validate_environment(
        headers={},
        model="x",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key=None,
    )
    assert headers["Authorization"] == "Key test-key"


@pytest.mark.parametrize(
    "config_class", [FalAINanoBananaProConfig, FalAINanoBanana2Config]
)
def test_request_body_includes_prompt(config_class):
    config = config_class()
    body = config.transform_image_generation_request(
        model="x",
        prompt="a cat in a bowtie",
        optional_params={"num_images": 2},
        litellm_params={},
        headers={},
    )
    assert body == {"prompt": "a cat in a bowtie", "num_images": 2}


@pytest.mark.parametrize(
    "config_class", [FalAINanoBananaProConfig, FalAINanoBanana2Config]
)
def test_response_parses_images_list(config_class):
    config = config_class()
    raw = httpx.Response(
        status_code=200,
        json={"images": [{"url": "https://o.png", "width": 1024, "height": 1024}]},
    )
    from litellm.types.utils import ImageResponse

    response = ImageResponse()
    out = config.transform_image_generation_response(
        model="x",
        raw_response=raw,
        model_response=response,
        logging_obj=MagicMock(),
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    assert len(out.data) == 1
    assert out.data[0].url == "https://o.png"


@pytest.mark.parametrize(
    "model,expected_cost",
    [
        ("fal_ai/fal-ai/nano-banana-pro", 0.15),
        ("fal_ai/fal-ai/nano-banana-2", 0.08),
    ],
)
def test_flat_per_image_cost(monkeypatch, model, expected_cost):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import litellm

    importlib.reload(litellm)

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
    assert abs(cost - expected_cost) < 1e-9
