from typing import Any

import litellm
from litellm.types.utils import ImageResponse


def cost_calculator(
    model: str,
    image_response: Any,
) -> float:
    """
    fal.ai image generation cost calculator.

    Most Fal models are flat-priced per image. Two exceptions are
    resolved via the shared ``default_image_cost_calculator``:

    - ``openai/gpt-image-2`` is tiered by quality + size; quality and
      size are stamped on the response by the transformation and feed
      the composite-key lookup.
    - ``fal-ai/clarity-upscaler`` uses pixel-based pricing
      (Fal bills $0.03/MP output); the transformation stamps
      ``image_response.size`` from the output dimensions, then
      ``input_cost_per_pixel`` × width × height returns the cost.
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

    if "clarity-upscaler" in model.lower():
        from litellm.cost_calculator import default_image_cost_calculator

        num_images = (
            len(image_response.data)
            if isinstance(image_response, ImageResponse) and image_response.data
            else 1
        )
        return default_image_cost_calculator(
            model=model,
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
