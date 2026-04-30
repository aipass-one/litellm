from .base import FalAIImageEditConfig


class FalAIBirefnetEditConfig(FalAIImageEditConfig):
    """
    fal-ai/birefnet — flat-priced background removal.

    Endpoint: https://fal.run/fal-ai/birefnet
    Pricing entry: ``fal_ai/fal-ai/birefnet`` (output_cost_per_image).
    Docs: https://fal.ai/models/fal-ai/birefnet

    Segmentation is fully image-driven, so we drop ``prompt`` and any OpenAI
    image-edit params. Fal-specific knobs (``model``, ``operating_resolution``,
    ``output_format``, ``refine_foreground``, ``output_mask``) flow through
    via ``extra_body``.
    """

    EDIT_ENDPOINT = "fal-ai/birefnet"
    ACCEPTS_PROMPT = False
