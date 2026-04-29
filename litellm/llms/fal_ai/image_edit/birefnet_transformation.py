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


class FalAIBirefnetEditConfig(BaseImageEditConfig):
    """
    Configuration for fal-ai/birefnet (background removal) served via Fal AI.

    Endpoint: https://fal.run/fal-ai/birefnet
    Documentation: https://fal.ai/models/fal-ai/birefnet

    Pricing is flat per image. See ``fal_ai/fal-ai/birefnet`` in
    ``model_prices_and_context_window.json``
    (``output_cost_per_image``).
    """

    DEFAULT_BASE_URL: str = "https://fal.run"
    EDIT_ENDPOINT: str = "fal-ai/birefnet"
    SUPPORTED_PARAMS: List[str] = []

    def get_supported_openai_params(self, model: str) -> List[str]:
        return list(self.SUPPORTED_PARAMS)

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict[str, Any]:
        # No OpenAI image-edit params (n, size, quality, response_format,
        # background, mask) apply to birefnet. Fal-specific knobs
        # (``model``, ``operating_resolution``, ``output_format``,
        # ``refine_foreground``, ``output_mask``) use Fal's own defaults.
        return {}

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
        image_url = self._build_image_url(image)
        if not image_url:
            raise ValueError(
                "fal-ai/birefnet requires exactly one input image."
            )

        # birefnet doesn't take a prompt — segmentation is fully image-driven.
        request_body: Dict[str, Any] = {"image_url": image_url}
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
        image = response_json.get("image")

        if isinstance(image, dict):
            data_list.append(
                ImageObject(
                    url=image.get("url"),
                    b64_json=image.get("b64_json"),
                )
            )
        elif isinstance(image, str):
            data_list.append(ImageObject(url=image, b64_json=None))

        model_response.data = cast(List[OpenAIImage], data_list)
        return model_response

    def _build_image_url(
        self, image: Union[FileTypes, List[FileTypes], None]
    ) -> Optional[str]:
        if image is None:
            return None
        if isinstance(image, list):
            if len(image) != 1:
                raise ValueError(
                    "fal-ai/birefnet accepts exactly one input image."
                )
            image = image[0]
        if image is None:
            return None
        if isinstance(image, str) and (
            image.startswith("http://")
            or image.startswith("https://")
            or image.startswith("data:")
        ):
            return image
        mime_type = ImageEditRequestUtils.get_image_content_type(image)
        image_bytes = self._read_all_bytes(image)
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

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
            "Unsupported image type for fal-ai/birefnet. "
            "Expected bytes, BytesIO, BufferedReader, or URL string."
        )
