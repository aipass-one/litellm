from .base import FalAIImageEditConfig


class FalAINanoBanana2EditConfig(FalAIImageEditConfig):
    """
    fal-ai/nano-banana-2/edit — Google Gemini 2.5 Flash Image (next iter).

    Endpoint: https://fal.run/fal-ai/nano-banana-2/edit
    Pricing entry: ``fal_ai/fal-ai/nano-banana-2/edit``
    (output_cost_per_image).
    Docs: https://fal.ai/models/fal-ai/nano-banana-2/edit

    Multi-image input/output, same shape as nano-banana-pro/edit.
    Fal-specific knobs (``aspect_ratio``, ``resolution``, ``output_format``,
    ``safety_tolerance``) flow through via ``extra_body``.
    """

    EDIT_ENDPOINT = "fal-ai/nano-banana-2/edit"
    SUPPORTED_PARAMS = ["n", "size"]
    PARAM_MAPPING = {"n": "num_images", "size": "image_size"}
    BODY_IMAGE_KEY = "image_urls"
    RESPONSE_IMAGE_KEY = "images"
    SUPPORTS_MULTI_IMAGE = True
