from .base import FalAIImageEditConfig


class FalAIRecraftUpscaleCrispEditConfig(FalAIImageEditConfig):
    """
    fal-ai/recraft/upscale/crisp — Recraft's faithful upscale variant.

    Endpoint: https://fal.run/fal-ai/recraft/upscale/crisp
    Pricing entry: ``fal_ai/fal-ai/recraft/upscale/crisp``
    (output_cost_per_image).
    Docs: https://fal.ai/models/fal-ai/recraft/upscale/crisp
    """

    EDIT_ENDPOINT = "fal-ai/recraft/upscale/crisp"
    ACCEPTS_PROMPT = False
