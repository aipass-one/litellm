from .base import FalAIImageEditConfig


class FalAIClarityUpscalerEditConfig(FalAIImageEditConfig):
    """
    fal-ai/clarity-upscaler — Stable-Diffusion-based creative upscaler.

    Endpoint: https://fal.run/fal-ai/clarity-upscaler
    Pricing entry: ``fal_ai/fal-ai/clarity-upscaler`` (input_cost_per_pixel).
    Docs: https://fal.ai/models/fal-ai/clarity-upscaler

    Pricing is by output megapixels — we stamp ``model_response.size`` from
    the response width/height so ``default_image_cost_calculator`` can
    multiply width × height × input_cost_per_pixel.
    """

    EDIT_ENDPOINT = "fal-ai/clarity-upscaler"
    STAMP_OUTPUT_SIZE = True
