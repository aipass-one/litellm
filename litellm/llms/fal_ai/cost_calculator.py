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
    if "gpt-image-2" in model.lower():
        from litellm.cost_calculator import default_image_cost_calculator

        num_images = (
            len(image_response.data)
            if isinstance(image_response, ImageResponse) and image_response.data
            else 1
        )
        return default_image_cost_calculator(
            model=model,
            quality=image_response.quality,
            custom_llm_provider=litellm.LlmProviders.FAL_AI.value,
            n=num_images,
            size=image_response.size,
        )

    _model_info = litellm.get_model_info(
        model=model,
        custom_llm_provider=litellm.LlmProviders.FAL_AI.value,
    )
    output_cost_per_image: float = _model_info.get("output_cost_per_image") or 0.0
    num_images: int = 0
    if isinstance(image_response, ImageResponse):
        if image_response.data:
            num_images = len(image_response.data)
        return output_cost_per_image * num_images
    else:
        raise ValueError(
            f"image_response must be of type ImageResponse got type={type(image_response)}"
        )
