from .base import FalAIImageEditConfig


class FalAIEsrganEditConfig(FalAIImageEditConfig):
    """
    fal-ai/esrgan — classic Real-ESRGAN faithful upscaler.

    Endpoint: https://fal.run/fal-ai/esrgan
    Pricing entry: ``fal_ai/fal-ai/esrgan`` (output_cost_per_image).
    Docs: https://fal.ai/models/fal-ai/esrgan
    """

    EDIT_ENDPOINT = "fal-ai/esrgan"
    ACCEPTS_PROMPT = False
