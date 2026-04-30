from .base import FalAIImageEditConfig


class FalAIAuraSREditConfig(FalAIImageEditConfig):
    """
    fal-ai/aura-sr — faithful (non-creative) 4× super-resolution.

    Endpoint: https://fal.run/fal-ai/aura-sr
    Pricing entry: ``fal_ai/fal-ai/aura-sr`` (input_cost_per_pixel).
    Docs: https://fal.ai/models/fal-ai/aura-sr

    Unlike clarity-upscaler, aura-sr preserves the source image — no
    Stable-Diffusion hallucination. Pricing is per output megapixel, so we
    stamp ``model_response.size`` from the response dimensions.
    """

    EDIT_ENDPOINT = "fal-ai/aura-sr"
    ACCEPTS_PROMPT = False
    STAMP_OUTPUT_SIZE = True
