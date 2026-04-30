"""
Dispatcher tests — every supported model must route to the right config
class. This catches substring collisions (e.g., ``birefnet/v2`` accidentally
matching the ``birefnet`` branch).
"""
import pytest

from litellm.llms.fal_ai.image_edit import (
    FalAIAuraSREditConfig,
    FalAIBenV2EditConfig,
    FalAIBirefnetEditConfig,
    FalAIBirefnetV2EditConfig,
    FalAIClarityUpscalerEditConfig,
    FalAIEsrganEditConfig,
    FalAIGptImage2EditConfig,
    FalAINanoBanana2EditConfig,
    FalAINanoBananaProEditConfig,
    FalAIRecraftUpscaleCreativeEditConfig,
    FalAIRecraftUpscaleCrispEditConfig,
    FalAITopazUpscaleEditConfig,
    get_fal_ai_image_edit_config,
)


@pytest.mark.parametrize(
    "model,expected_class",
    [
        # Nano Banana family — must take precedence over any "nano-banana" prefix.
        ("fal-ai/nano-banana-pro/edit", FalAINanoBananaProEditConfig),
        ("fal_ai/fal-ai/nano-banana-pro/edit", FalAINanoBananaProEditConfig),
        ("fal-ai/nano-banana-2/edit", FalAINanoBanana2EditConfig),
        ("fal_ai/fal-ai/nano-banana-2/edit", FalAINanoBanana2EditConfig),
        # gpt-image-2/edit (composite-key tiered pricing).
        ("fal-ai/openai/gpt-image-2/edit", FalAIGptImage2EditConfig),
        ("fal_ai/openai/gpt-image-2/edit", FalAIGptImage2EditConfig),
        # Recraft pair.
        ("fal-ai/recraft/upscale/crisp", FalAIRecraftUpscaleCrispEditConfig),
        ("fal_ai/fal-ai/recraft/upscale/crisp", FalAIRecraftUpscaleCrispEditConfig),
        ("fal-ai/recraft/upscale/creative", FalAIRecraftUpscaleCreativeEditConfig),
        (
            "fal_ai/fal-ai/recraft/upscale/creative",
            FalAIRecraftUpscaleCreativeEditConfig,
        ),
        # Topaz.
        ("fal-ai/topaz/upscale/image", FalAITopazUpscaleEditConfig),
        ("fal_ai/fal-ai/topaz/upscale/image", FalAITopazUpscaleEditConfig),
        # BiRefNet — v2 must NOT fall through to v1.
        ("fal-ai/birefnet/v2", FalAIBirefnetV2EditConfig),
        ("fal_ai/fal-ai/birefnet/v2", FalAIBirefnetV2EditConfig),
        ("fal-ai/birefnet", FalAIBirefnetEditConfig),
        ("fal_ai/fal-ai/birefnet", FalAIBirefnetEditConfig),
        # BEN.
        ("fal-ai/ben/v2/image", FalAIBenV2EditConfig),
        ("fal_ai/fal-ai/ben/v2/image", FalAIBenV2EditConfig),
        # Single-name models.
        ("fal-ai/aura-sr", FalAIAuraSREditConfig),
        ("fal_ai/fal-ai/aura-sr", FalAIAuraSREditConfig),
        ("fal-ai/esrgan", FalAIEsrganEditConfig),
        ("fal_ai/fal-ai/esrgan", FalAIEsrganEditConfig),
        ("fal-ai/clarity-upscaler", FalAIClarityUpscalerEditConfig),
        ("fal_ai/fal-ai/clarity-upscaler", FalAIClarityUpscalerEditConfig),
    ],
)
def test_dispatcher_routes_to_correct_class(model, expected_class):
    config = get_fal_ai_image_edit_config(model)
    assert isinstance(config, expected_class), (
        f"model={model!r} routed to {type(config).__name__}, "
        f"expected {expected_class.__name__}"
    )


def test_dispatcher_raises_for_unknown_model():
    with pytest.raises(NotImplementedError, match="No fal_ai image-edit config"):
        get_fal_ai_image_edit_config("fal-ai/unknown-model")
