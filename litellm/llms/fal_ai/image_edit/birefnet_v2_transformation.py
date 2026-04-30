from .base import FalAIImageEditConfig


class FalAIBirefnetV2EditConfig(FalAIImageEditConfig):
    """
    fal-ai/birefnet/v2 — latest BiRefNet for background removal.

    Endpoint: https://fal.run/fal-ai/birefnet/v2
    Pricing entry: ``fal_ai/fal-ai/birefnet/v2`` (output_cost_per_image).
    Docs: https://fal.ai/models/fal-ai/birefnet/v2

    Variants are selected via Fal's ``model`` body param, surfaced through
    ``extra_body``: ``"General Use (Light)"``, ``"General Use (Light 2K)"``,
    ``"General Use (Heavy)"``, ``"Portrait"``, ``"Matting"``,
    ``"General Use (Dynamic)"``. Pricing is flat per image regardless of
    variant.
    """

    EDIT_ENDPOINT = "fal-ai/birefnet/v2"
    ACCEPTS_PROMPT = False
