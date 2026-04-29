from typing import Any

import litellm
from litellm.types.utils import ImageResponse


def cost_calculator(
    model: str,
    image_response: Any,
) -> float:
    """
    fal.ai image generation cost calculator.

    Most Fal models are flat-priced per image. ``openai/gpt-image-2`` is
    tiered by quality + size and is resolved through the shared
    ``default_image_cost_calculator`` composite-key lookup
    (``cost_calculator.py:1938``). Quality and size are read off the
    response object — the transformation stamps them in
    ``transform_image_generation_response``.
    """
    if not isinstance(image_response, ImageResponse):
        raise ValueError(
            f"image_response must be of type ImageResponse got type={type(image_response)}"
        )

    num_images: int = len(image_response.data) if image_response.data else 0

    if "gpt-image-2" in model.lower():
        from litellm.cost_calculator import default_image_cost_calculator

        return default_image_cost_calculator(
            model=model,
            quality=image_response.quality,
            custom_llm_provider=litellm.LlmProviders.FAL_AI.value,
            n=num_images or 1,
            size=image_response.size,
        )

    _model_info = litellm.get_model_info(
        model=model,
        custom_llm_provider=litellm.LlmProviders.FAL_AI.value,
    )
    output_cost_per_image: float = _model_info.get("output_cost_per_image") or 0.0
    return output_cost_per_image * num_images
