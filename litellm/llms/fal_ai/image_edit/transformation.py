from typing import Any, Dict, List, Optional

import httpx

from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.utils import ImageResponse

from .base import FalAIImageEditConfig


class FalAIGptImage2EditConfig(FalAIImageEditConfig):
    """
    OpenAI GPT Image 2 edit endpoint served via Fal AI.

    Endpoint: https://fal.run/openai/gpt-image-2/edit
    Pricing entries: ``{quality}/fal_ai/{W}-x-{H}/openai/gpt-image-2/edit``
    in ``model_prices_and_context_window.json`` — same composite-key matrix
    as the text-to-image variant.
    Docs: https://fal.ai/models/openai/gpt-image-2/edit

    Special handling vs. the rest of the Fal edit family:

    - Multi-image input (``image_urls`` array) and multi-image output
      (``images`` list).
    - OpenAI ``size`` strings are mapped to Fal presets where possible
      (``square_hd``, ``landscape_4_3``, ...) or fall through to a
      ``{"width": int, "height": int}`` object.
    - OpenAI ``response_format=b64_json|url`` maps to Fal
      ``output_format=png``.
    - Cost-calc lookup needs ``quality`` and ``{w}-x-{h}`` size on the
      response. We re-parse the outgoing request body so we stamp what was
      *requested* (auth source of truth) rather than the response payload.
    """

    EDIT_ENDPOINT = "openai/gpt-image-2/edit"
    SUPPORTED_PARAMS = ["n", "size", "quality", "response_format"]
    PARAM_MAPPING = {
        "n": "num_images",
        "response_format": "output_format",
        "size": "image_size",
        "quality": "quality",
    }
    BODY_IMAGE_KEY = "image_urls"
    RESPONSE_IMAGE_KEY = "images"
    SUPPORTS_MULTI_IMAGE = True

    _PRESET_TO_SIZE = {
        "square_hd": (1024, 1024),
        "square": (512, 512),
        "landscape_4_3": (1024, 768),
        "landscape_16_9": (1024, 576),
        "portrait_4_3": (768, 1024),
        "portrait_16_9": (576, 1024),
    }

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict[str, Any]:
        supported = self.get_supported_openai_params(model)
        mapped: Dict[str, Any] = {}
        for k, v in image_edit_optional_params.items():
            if k not in supported:
                continue
            mapped_key = self.PARAM_MAPPING.get(k, k)
            value: Any = v

            if k == "response_format" and value in ("b64_json", "url"):
                value = "png"
            elif k == "size":
                value = self._map_image_size(value)

            mapped[mapped_key] = value
        return mapped

    def _map_image_size(self, size: Any) -> Any:
        if isinstance(size, dict):
            return size
        if not isinstance(size, str):
            return size
        if size in self._PRESET_TO_SIZE or size == "auto":
            return size
        if "x" in size:
            try:
                width_str, height_str = size.lower().split("x")
                width = int(width_str.strip())
                height = int(height_str.strip())
                for preset, (pw, ph) in self._PRESET_TO_SIZE.items():
                    if pw == width and ph == height:
                        return preset
                return {"width": width, "height": height}
            except (ValueError, AttributeError):
                pass
        return size

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: Any,
    ) -> ImageResponse:
        # Base class parses ``images`` list, stamps model. We layer on the
        # gpt-image-2 cost-calc requirements (quality + size from the
        # outgoing request).
        model_response = super().transform_image_edit_response(
            model=model, raw_response=raw_response, logging_obj=logging_obj
        )

        first_width: Optional[int] = None
        first_height: Optional[int] = None
        if model_response.data:
            first = model_response.data[0]
            # ImageObject is a TypedDict-like; width/height aren't standard
            # attrs, so we re-extract from the JSON if present.
            try:
                response_json = raw_response.json()
            except Exception:
                response_json = {}
            images = response_json.get("images") or []
            if images and isinstance(images[0], dict):
                first_width = images[0].get("width")
                first_height = images[0].get("height")

        request_payload = self._parse_request_payload(raw_response)
        request_size = self._dims_from_request(
            request_payload.get("image_size"), first_width, first_height
        )
        if request_size:
            model_response.size = request_size
        # ``quality`` defaults to Fal's documented "high" if the caller didn't
        # specify — keeps the cost-calc composite-key lookup deterministic.
        model_response.quality = request_payload.get("quality") or "high"

        return model_response

    @staticmethod
    def _parse_request_payload(raw_response: httpx.Response) -> Dict[str, Any]:
        # ``raw_response.request`` raises RuntimeError when the response was
        # constructed without a request (unit tests sometimes do this).
        try:
            request = raw_response.request
        except (RuntimeError, AttributeError):
            return {}
        content = getattr(request, "content", None)
        if not content:
            return {}
        try:
            import json

            return json.loads(content)
        except (ValueError, TypeError):
            return {}

    def _dims_from_request(
        self,
        image_size: Any,
        response_width: Optional[int],
        response_height: Optional[int],
    ) -> Optional[str]:
        if isinstance(image_size, dict):
            w = image_size.get("width")
            h = image_size.get("height")
            if w and h:
                return f"{w}-x-{h}"
        elif isinstance(image_size, str) and image_size in self._PRESET_TO_SIZE:
            w, h = self._PRESET_TO_SIZE[image_size]
            return f"{w}-x-{h}"
        if response_width and response_height:
            return f"{response_width}-x-{response_height}"
        return None
