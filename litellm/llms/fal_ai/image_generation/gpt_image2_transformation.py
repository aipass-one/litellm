from typing import TYPE_CHECKING, Any, List, Optional

import httpx

from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageObject, ImageResponse

from .transformation import FalAIBaseConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class FalAIGptImage2Config(FalAIBaseConfig):
    """
    Configuration for OpenAI GPT Image 2 served via Fal AI.

    Endpoint: https://fal.run/openai/gpt-image-2
    Documentation: https://fal.ai/models/openai/gpt-image-2

    Pricing is tiered by quality (low / medium / high) and size.
    See ``fal_ai/{quality}/{W}-x-{H}/openai/gpt-image-2`` entries in
    ``model_prices_and_context_window.json``.
    """

    IMAGE_GENERATION_ENDPOINT: str = "openai/gpt-image-2"

    _PRESET_TO_SIZE = {
        "square_hd": (1024, 1024),
        "square": (512, 512),
        "landscape_4_3": (1024, 768),
        "landscape_16_9": (1024, 576),
        "portrait_4_3": (768, 1024),
        "portrait_16_9": (576, 1024),
    }

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        return [
            "n",
            "response_format",
            "size",
            "quality",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)

        param_mapping = {
            "n": "num_images",
            "response_format": "output_format",
            "size": "image_size",
            "quality": "quality",
        }

        for k, v in non_default_params.items():
            if k in optional_params:
                continue
            if k not in supported_params:
                if drop_params:
                    continue
                raise ValueError(
                    f"Parameter {k} is not supported for model {model}. "
                    f"Supported parameters are {supported_params}. "
                    "Set drop_params=True to drop unsupported parameters."
                )

            mapped_key = param_mapping.get(k, k)
            mapped_value = v

            if k == "response_format":
                if mapped_value in ("b64_json", "url"):
                    mapped_value = "png"
            elif k == "size":
                mapped_value = self._map_image_size(mapped_value)

            optional_params[mapped_key] = mapped_value

        return optional_params

    def _map_image_size(self, size: Any) -> Any:
        """Translate OpenAI ``size`` to Fal ``image_size``.

        Accepts:
        - Fal preset name (``"landscape_4_3"``) → returned as-is
        - ``{"width": W, "height": H}`` dict → returned as-is
        - ``"WIDTHxHEIGHT"`` → mapped to a preset if it matches, else a dict
        """
        if isinstance(size, dict):
            return size
        if not isinstance(size, str):
            return size
        if size in self._PRESET_TO_SIZE:
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

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        return {"prompt": prompt, **optional_params}

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ImageResponse:
        try:
            response_data = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Error transforming image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        if not model_response.data:
            model_response.data = []

        first_width: Optional[int] = None
        first_height: Optional[int] = None

        for image_data in response_data.get("images", []) or []:
            if isinstance(image_data, dict):
                model_response.data.append(
                    ImageObject(
                        url=image_data.get("url"),
                        b64_json=image_data.get("b64_json"),
                    )
                )
                if first_width is None:
                    first_width = image_data.get("width")
                    first_height = image_data.get("height")
            elif isinstance(image_data, str):
                model_response.data.append(ImageObject(url=image_data, b64_json=None))

        # Stamp size and quality on the response so the cost calculator can
        # find the matching {quality}/{size}/{model} entry. Prefer dimensions
        # the API actually returned; fall back to the request payload.
        if first_width is None or first_height is None:
            first_width, first_height = self._dims_from_request(
                request_data.get("image_size")
            )

        if first_width and first_height:
            model_response.size = f"{first_width}-x-{first_height}"

        model_response.quality = request_data.get("quality", "high")

        # Capture Fal's authoritative billing quantity. Mirrors the logic in
        # FalAIBaseConfig.transform_image_generation_response — kept here too
        # because this class fully overrides that method instead of calling
        # super(). cost_calculator multiplies the units by the static
        # unit_price for gpt-image-2 ($1/unit).
        units_header = raw_response.headers.get("x-fal-billable-units")
        if units_header is not None:
            try:
                hidden = model_response._hidden_params or {}
                hidden["fal_billable_units"] = float(units_header)
                model_response._hidden_params = hidden
            except (TypeError, ValueError):
                pass

        return model_response

    def _dims_from_request(self, image_size: Any) -> tuple:
        if isinstance(image_size, dict):
            return image_size.get("width"), image_size.get("height")
        if isinstance(image_size, str) and image_size in self._PRESET_TO_SIZE:
            return self._PRESET_TO_SIZE[image_size]
        # Fal's documented default
        return self._PRESET_TO_SIZE["landscape_4_3"]
