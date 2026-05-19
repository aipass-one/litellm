import base64
from io import BufferedReader, BytesIO
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import httpx
from httpx._types import RequestFiles

from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.llms.fal_ai.error_utils import classify_fal_ai_error
from litellm.secret_managers.main import get_secret_str
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageObject, ImageResponse, OpenAIImage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as _LiteLLMLoggingObj,
    )

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class FalAIImageEditConfig(BaseImageEditConfig):
    """
    Shared base for Fal AI image-edit transformations.

    Concrete subclasses just declare class attributes (endpoint, supported
    params, response shape) and inherit URL building, environment validation,
    request/response transformation, and base64 image handling.

    Subclass overrides
    ------------------
    EDIT_ENDPOINT : Fal endpoint slug (REQUIRED), e.g. ``fal-ai/aura-sr``.
    SUPPORTED_PARAMS : OpenAI image-edit params the model accepts. Anything
        outside this list is dropped silently in ``map_openai_params``.
    PARAM_MAPPING : OpenAI param name → Fal request body key
        (e.g. ``{"n": "num_images"}``).
    BODY_IMAGE_KEY : ``"image_url"`` (singular, default) or ``"image_urls"``
        (list).
    RESPONSE_IMAGE_KEY : ``"image"`` (singular, default) or ``"images"``
        (list). Most Fal models return ``image`` (single output); gpt-image-2
        and nano-banana return ``images``.
    SUPPORTS_MULTI_IMAGE : whether the endpoint accepts multiple input images.
    ACCEPTS_PROMPT : whether ``prompt`` is forwarded. False for pure
        image→image models like birefnet.
    STAMP_OUTPUT_SIZE : whether the response carries width/height that should
        be stamped onto ``model_response.size``. Required for per-pixel-priced
        models so the cost calculator can multiply ``input_cost_per_pixel``
        by ``width * height``.
    """

    DEFAULT_BASE_URL: ClassVar[str] = "https://fal.run"

    EDIT_ENDPOINT: ClassVar[str] = ""
    SUPPORTED_PARAMS: ClassVar[List[str]] = []
    PARAM_MAPPING: ClassVar[Dict[str, str]] = {}
    BODY_IMAGE_KEY: ClassVar[str] = "image_url"
    RESPONSE_IMAGE_KEY: ClassVar[str] = "image"
    SUPPORTS_MULTI_IMAGE: ClassVar[bool] = False
    ACCEPTS_PROMPT: ClassVar[bool] = True
    STAMP_OUTPUT_SIZE: ClassVar[bool] = False

    # ------------------------------------------------------------------ env

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

    # --------------------------------------------------------------- errors

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],
    ) -> BaseLLMException:
        return classify_fal_ai_error(error_message, status_code, headers)

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

    # --------------------------------------------------------------- params

    def get_supported_openai_params(self, model: str) -> List[str]:
        return list(self.SUPPORTED_PARAMS)

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
            mapped[self.PARAM_MAPPING.get(k, k)] = v
        return mapped

    # --------------------------------------------------------------- request

    def transform_image_edit_request(  # type: ignore[override]
        self,
        model: str,
        prompt: Optional[str],
        image: Optional[FileTypes],
        image_edit_optional_request_params: Dict[str, Any],
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict[str, Any], Optional[RequestFiles]]:
        if self.SUPPORTS_MULTI_IMAGE:
            urls = self._build_image_urls(image)
            if not urls:
                raise ValueError(
                    f"{self.EDIT_ENDPOINT} requires at least one input image."
                )
            request_body: Dict[str, Any] = {self.BODY_IMAGE_KEY: urls}
        else:
            url = self._build_image_url(image)
            if not url:
                raise ValueError(
                    f"{self.EDIT_ENDPOINT} requires exactly one input image."
                )
            request_body = {self.BODY_IMAGE_KEY: url}

        if self.ACCEPTS_PROMPT and prompt:
            request_body["prompt"] = prompt
        request_body.update(image_edit_optional_request_params)

        empty_files = cast(RequestFiles, [])
        return request_body, empty_files

    # -------------------------------------------------------------- response

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
        # Stamp the model name so downstream consumers (cost calc, LiteLLM
        # response surfacing) don't trip on a missing ``model`` attribute —
        # the previous implementations forgot this and the
        # ``cannot override response model`` runtime error in
        # ``common_request_processing.py:410`` traces back here.
        model_response.model = model

        items = self._extract_response_items(response_json)
        data_list: List[ImageObject] = []
        first_width: Optional[int] = None
        first_height: Optional[int] = None

        for item in items:
            if isinstance(item, dict):
                data_list.append(
                    ImageObject(
                        url=item.get("url"),
                        b64_json=item.get("b64_json"),
                    )
                )
                if first_width is None:
                    first_width = item.get("width")
                    first_height = item.get("height")
            elif isinstance(item, str):
                data_list.append(ImageObject(url=item, b64_json=None))

        model_response.data = cast(List[OpenAIImage], data_list)

        if self.STAMP_OUTPUT_SIZE and first_width and first_height:
            # Per-pixel-priced models historically read ``model_response.size``
            # in the matrix-based cost calculator. Kept stamping the size for
            # backward compatibility / debugging visibility, but the new
            # cost path prefers the ``x-fal-billable-units`` response header
            # below; this is only a fallback path.
            model_response.size = f"{first_width}-x-{first_height}"

        # Capture Fal's authoritative billing quantity (header-emitting
        # endpoints). Multiplied by the static unit_price in cost_calculator.
        units_header = raw_response.headers.get("x-fal-billable-units")
        if units_header is not None:
            try:
                hidden = model_response._hidden_params or {}
                hidden["fal_billable_units"] = float(units_header)
                model_response._hidden_params = hidden
            except (TypeError, ValueError):
                pass

        return model_response

    def _extract_response_items(self, response_json: Dict[str, Any]) -> List[Any]:
        """
        Pull a flat list of image entries out of the Fal response, regardless
        of whether the model puts them under ``image`` (singular) or
        ``images`` (list). Subclasses can override for more exotic shapes.
        """
        raw = response_json.get(self.RESPONSE_IMAGE_KEY)
        if raw is None:
            return []
        if isinstance(raw, list):
            return [item for item in raw if item is not None]
        return [raw]

    # ---------------------------------------------------------- image inputs

    def _build_image_url(
        self, image: Union[FileTypes, List[FileTypes], None]
    ) -> Optional[str]:
        """Single-input variant. Rejects multi-image lists with a clear error."""
        if image is None:
            return None
        if isinstance(image, list):
            if len(image) != 1:
                raise ValueError(
                    f"{self.EDIT_ENDPOINT} accepts exactly one input image."
                )
            image = image[0]
        if image is None:
            return None
        return self._encode_single(image)

    def _build_image_urls(
        self, image: Union[FileTypes, List[FileTypes], None]
    ) -> List[str]:
        """Multi-input variant. Always returns a list (possibly empty)."""
        if image is None:
            return []
        images: List[FileTypes] = image if isinstance(image, list) else [image]
        return [self._encode_single(img) for img in images if img is not None]

    def _encode_single(self, image: FileTypes) -> str:
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
            f"Unsupported image type for {self.EDIT_ENDPOINT}. "
            "Expected bytes, BytesIO, BufferedReader, or URL string."
        )
