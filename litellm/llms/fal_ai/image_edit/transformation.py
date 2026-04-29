import base64
from io import BufferedReader, BytesIO
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

import httpx
from httpx._types import RequestFiles

from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageObject, ImageResponse, OpenAIImage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class FalAIGptImage2EditConfig(BaseImageEditConfig):
    """
    Configuration for OpenAI GPT Image 2 edit endpoint served via Fal AI.

    Endpoint: https://fal.run/openai/gpt-image-2/edit
    Documentation: https://fal.ai/models/openai/gpt-image-2/edit

    Pricing matches the text-to-image variant exactly (same model, same matrix).
    See ``{quality}/fal_ai/{W}-x-{H}/openai/gpt-image-2/edit`` entries in
    ``model_prices_and_context_window.json``.
    """

    DEFAULT_BASE_URL: str = "https://fal.run"
    EDIT_ENDPOINT: str = "openai/gpt-image-2/edit"
    SUPPORTED_PARAMS: List[str] = ["n", "size", "quality", "response_format"]

    _PRESET_TO_SIZE = {
        "square_hd": (1024, 1024),
        "square": (512, 512),
        "landscape_4_3": (1024, 768),
        "landscape_16_9": (1024, 576),
        "portrait_4_3": (768, 1024),
        "portrait_16_9": (576, 1024),
    }

    def get_supported_openai_params(self, model: str) -> List[str]:
        return list(self.SUPPORTED_PARAMS)

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict[str, Any]:
        supported_params = self.get_supported_openai_params(model)
        param_mapping = {
            "n": "num_images",
            "response_format": "output_format",
            "size": "image_size",
            "quality": "quality",
        }

        mapped_params: Dict[str, Any] = {}
        for k, v in image_edit_optional_params.items():
            if k not in supported_params:
                continue
            mapped_key = param_mapping.get(k, k)
            mapped_value = v

            if k == "response_format":
                if mapped_value in ("b64_json", "url"):
                    mapped_value = "png"
            elif k == "size":
                mapped_value = self._map_image_size(mapped_value)

            mapped_params[mapped_key] = mapped_value

        return mapped_params

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

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[dict] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        final_api_key: Optional[str] = api_key or get_secret_str("FAL_AI_API_KEY")
        if not final_api_key:
            raise ValueError("FAL_AI_API_KEY is not set")
        headers["Authorization"] = f"Key {final_api_key}"
        headers["Content-Type"] = "application/json"
        return headers

    def use_multipart_form_data(self) -> bool:
        return False

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        base_url = (
            api_base or get_secret_str("FAL_AI_API_BASE") or self.DEFAULT_BASE_URL
        ).rstrip("/")
        return f"{base_url}/{self.EDIT_ENDPOINT}"

    def transform_image_edit_request(  # type: ignore[override]
        self,
        model: str,
        prompt: Optional[str],
        image: Optional[FileTypes],
        image_edit_optional_request_params: Dict[str, Any],
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict[str, Any], Optional[RequestFiles]]:
        image_urls = self._build_image_urls(image)
        if not image_urls:
            raise ValueError(
                "openai/gpt-image-2/edit requires at least one input image."
            )

        request_body: Dict[str, Any] = {"image_urls": image_urls}
        if prompt:
            request_body["prompt"] = prompt
        request_body.update(image_edit_optional_request_params)

        empty_files = cast(RequestFiles, [])
        return request_body, empty_files

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: Any,
    ) -> ImageResponse:
        try:
            response_json = raw_response.json()
        except Exception as exc:
            raise self.get_error_class(
                error_message=f"Error transforming image edit response: {exc}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        model_response = ImageResponse()
        data_list: List[ImageObject] = []
        first_width: Optional[int] = None
        first_height: Optional[int] = None

        for image_data in response_json.get("images", []) or []:
            if isinstance(image_data, dict):
                data_list.append(
                    ImageObject(
                        url=image_data.get("url"),
                        b64_json=image_data.get("b64_json"),
                    )
                )
                if first_width is None:
                    first_width = image_data.get("width")
                    first_height = image_data.get("height")
            elif isinstance(image_data, str):
                data_list.append(ImageObject(url=image_data, b64_json=None))

        model_response.data = cast(List[OpenAIImage], data_list)

        # Stamp quality + size from the request we sent (parsed from the
        # outgoing httpx Request body) so the cost calc can resolve the
        # matching {quality}/{size}/{model} entry. Falling back to response
        # dimensions if the request didn't pin a size, and to "high" (Fal's
        # documented default) if quality wasn't specified.
        request_payload = self._parse_request_payload(raw_response)
        request_size = self._dims_from_request(
            request_payload.get("image_size"), first_width, first_height
        )
        if request_size:
            model_response.size = request_size
        model_response.quality = request_payload.get("quality") or "high"

        return model_response

    @staticmethod
    def _parse_request_payload(raw_response: httpx.Response) -> Dict[str, Any]:
        # ``Response.request`` raises RuntimeError when the response was
        # constructed without a request (e.g. unit tests), so guard for that.
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

    def _build_image_urls(
        self, image: Union[FileTypes, List[FileTypes], None]
    ) -> List[str]:
        if image is None:
            return []
        images: List[FileTypes] = image if isinstance(image, list) else [image]
        urls: List[str] = []
        for img in images:
            if img is None:
                continue
            if isinstance(img, str) and (
                img.startswith("http://")
                or img.startswith("https://")
                or img.startswith("data:")
            ):
                urls.append(img)
                continue
            mime_type = ImageEditRequestUtils.get_image_content_type(img)
            image_bytes = self._read_all_bytes(img)
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            urls.append(f"data:{mime_type};base64,{encoded}")
        return urls

    def _read_all_bytes(self, image: FileTypes) -> bytes:
        if isinstance(image, bytes):
            return image
        if isinstance(image, BytesIO):
            current_pos = image.tell()
            image.seek(0)
            data = image.read()
            image.seek(current_pos)
            return data
        if isinstance(image, BufferedReader):
            current_pos = image.tell()
            image.seek(0)
            data = image.read()
            image.seek(current_pos)
            return data
        raise ValueError(
            "Unsupported image type for openai/gpt-image-2/edit. "
            "Expected bytes, BytesIO, BufferedReader, or URL string."
        )
