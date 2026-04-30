from .base import FalAIImageEditConfig


class FalAINanoBananaProEditConfig(FalAIImageEditConfig):
    """
    fal-ai/nano-banana-pro/edit — Google Gemini 2.5 Flash Image (Pro tier).

    Endpoint: https://fal.run/fal-ai/nano-banana-pro/edit
    Pricing entry: ``fal_ai/fal-ai/nano-banana-pro/edit``
    (output_cost_per_image).
    Docs: https://fal.ai/models/fal-ai/nano-banana-pro/edit

    Multi-image input via ``image_urls`` array; multi-image output via
    ``images`` list. Accepts ``num_images`` and ``image_size``.
    """

    EDIT_ENDPOINT = "fal-ai/nano-banana-pro/edit"
    SUPPORTED_PARAMS = ["n", "size"]
    PARAM_MAPPING = {"n": "num_images", "size": "image_size"}
    BODY_IMAGE_KEY = "image_urls"
    RESPONSE_IMAGE_KEY = "images"
    SUPPORTS_MULTI_IMAGE = True
