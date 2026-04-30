from .base import FalAIImageEditConfig


class FalAIBenV2EditConfig(FalAIImageEditConfig):
    """
    fal-ai/ben/v2/image — BEN v2 background removal.

    Endpoint: https://fal.run/fal-ai/ben/v2/image
    Pricing entry: ``fal_ai/fal-ai/ben/v2/image`` (output_cost_per_image).
    Docs: https://fal.ai/models/fal-ai/ben/v2/image

    Lightweight, fast alternative to BiRefNet with simpler variant selection.
    """

    EDIT_ENDPOINT = "fal-ai/ben/v2/image"
    ACCEPTS_PROMPT = False
