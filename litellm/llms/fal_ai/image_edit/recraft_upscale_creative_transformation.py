from .base import FalAIImageEditConfig


class FalAIRecraftUpscaleCreativeEditConfig(FalAIImageEditConfig):
    """
    fal-ai/recraft/upscale/creative — Recraft's quality-enhancement variant.

    Endpoint: https://fal.run/fal-ai/recraft/upscale/creative
    Pricing entry: ``fal_ai/fal-ai/recraft/upscale/creative``
    (output_cost_per_image).
    Docs: https://fal.ai/models/fal-ai/recraft/upscale/creative

    Sharpens and refines while increasing resolution — slightly more
    aggressive than ``crisp``, less hallucinatory than ``clarity-upscaler``.
    """

    EDIT_ENDPOINT = "fal-ai/recraft/upscale/creative"
    ACCEPTS_PROMPT = False
