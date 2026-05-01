from .base import FalAIImageEditConfig


class FalAIBenV2EditConfig(FalAIImageEditConfig):
    """
    fal-ai/ben/v2/image — BEN v2 background removal.

    Endpoint: https://fal.run/fal-ai/ben/v2/image
    Pricing entry: ``fal_ai/fal-ai/ben/v2/image`` (input_cost_per_pixel —
    Fal bills $0.025/megapixel).
    Docs: https://fal.ai/models/fal-ai/ben/v2/image

    Lightweight, fast alternative to BiRefNet with simpler variant selection.
    Per-megapixel pricing means we stamp ``model_response.size`` from the
    response so the cost calc multiplies width × height × per-pixel rate.
    """

    EDIT_ENDPOINT = "fal-ai/ben/v2/image"
    ACCEPTS_PROMPT = False
    STAMP_OUTPUT_SIZE = True
