from .base import FalAIImageEditConfig


class FalAITopazUpscaleEditConfig(FalAIImageEditConfig):
    """
    fal-ai/topaz/upscale/image — premium professional upscaler.

    Endpoint: https://fal.run/fal-ai/topaz/upscale/image
    Pricing entry: ``fal_ai/fal-ai/topaz/upscale/image``
    (output_cost_per_image).
    Docs: https://fal.ai/models/fal-ai/topaz/upscale/image
    """

    EDIT_ENDPOINT = "fal-ai/topaz/upscale/image"
    ACCEPTS_PROMPT = False
