"""
Shared base for Fal AI video-generation transformations.

Fal exposes video models via its **queue API**:

* ``POST  https://queue.fal.run/{endpoint}``                          → submit
* ``GET   https://queue.fal.run/{endpoint}/requests/{id}/status``     → poll
* ``GET   https://queue.fal.run/{endpoint}/requests/{id}``            → fetch

Submit returns ``{request_id, response_url, status_url, cancel_url}``; polling
returns ``{status, request_id, queue_position, logs}``; the result fetch (when
status is ``COMPLETED``) returns the model-specific payload, which for
seedance is ``{"video": {"url": "...", "content_type": "video/mp4", ...},
"seed": 42}``.

This base implements the full lifecycle plus PR #22's ``x-fal-billable-units``
header capture, so cost_calculator.py can reuse the same billing math we
already use for image generation. Concrete subclasses just override
``VIDEO_ENDPOINT`` and (optionally) ``SUPPORTED_PARAMS`` /
``PARAM_MAPPING`` for OpenAI-style param translation.
"""

import time
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

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.llms.fal_ai.error_utils import classify_fal_ai_error
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
    extract_original_video_id,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as _LiteLLMLoggingObj,
    )

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


def _strip_provider_prefix(model: str) -> str:
    """Drop the ``fal_ai/`` provider prefix LiteLLM adds to the model name."""
    return model[len("fal_ai/"):] if model.startswith("fal_ai/") else model


class FalAIBaseVideoConfig(BaseVideoConfig):
    """
    Subclass overrides
    ------------------
    VIDEO_ENDPOINT : Fal endpoint slug (e.g. ``bytedance/seedance-2.0/image-to-video``).
        Optional — when empty, derived from the model string at request time.
    SUPPORTED_PARAMS : OpenAI video-create params the model accepts. Anything
        outside this list is dropped silently in ``map_openai_params``.
    PARAM_MAPPING : OpenAI param name → Fal request body key
        (e.g. ``{"size": "resolution"}``).
    """

    DEFAULT_QUEUE_URL: ClassVar[str] = "https://queue.fal.run"

    VIDEO_ENDPOINT: ClassVar[str] = ""
    SUPPORTED_PARAMS: ClassVar[List[str]] = [
        "model",
        "prompt",
        "input_reference",
        "seconds",
        "size",
        "user",
        "extra_headers",
        "extra_body",
    ]
    PARAM_MAPPING: ClassVar[Dict[str, str]] = {
        "input_reference": "image_url",
    }

    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------ env

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[GenericLiteLLMParams] = None,
    ) -> dict:
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key
        final_api_key: Optional[str] = api_key or get_secret_str("FAL_AI_API_KEY")
        if not final_api_key:
            raise ValueError("FAL_AI_API_KEY is not set")
        headers["Authorization"] = f"Key {final_api_key}"
        headers["Content-Type"] = "application/json"
        return headers

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],
    ) -> BaseLLMException:
        return classify_fal_ai_error(error_message, status_code, headers)

    # ---------------------------------------------------------- url building

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        base = (
            api_base or get_secret_str("FAL_AI_API_BASE") or self.DEFAULT_QUEUE_URL
        ).rstrip("/")
        endpoint = self.VIDEO_ENDPOINT or _strip_provider_prefix(model)
        return f"{base}/{endpoint}"

    # --------------------------------------------------------------- params

    def get_supported_openai_params(self, model: str) -> list:
        return list(self.SUPPORTED_PARAMS)

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        supported = self.get_supported_openai_params(model)
        mapped: Dict[str, Any] = {}
        for key, value in video_create_optional_params.items():
            if key not in supported:
                continue
            if value is None:
                continue
            target = self.PARAM_MAPPING.get(key, key)
            mapped[target] = value

        # extra_body is a Fal-specific passthrough: clients drop raw Fal
        # params there (resolution, aspect_ratio, generate_audio, seed,
        # end_image_url) without going through the OpenAI mapping.
        extra = video_create_optional_params.get("extra_body")
        if extra:
            mapped.update(extra)
            mapped.pop("extra_body", None)

        return mapped

    # -------------------------------------------------------- create request

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles, str]:
        request_body: Dict[str, Any] = {"prompt": prompt}
        request_body.update(video_create_optional_request_params or {})

        if "image_url" not in request_body:
            raise ValueError(
                "Fal AI image-to-video requires 'image_url' (pass via "
                "input_reference or extra_body.image_url)."
            )

        empty_files = cast(RequestFiles, [])
        return request_body, empty_files, api_base

    # ------------------------------------------------------ create response

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
        request_data: Optional[Dict] = None,
    ) -> VideoObject:
        data = raw_response.json()

        request_id = data.get("request_id") or data.get("id") or ""
        video_data: Dict[str, Any] = {
            "id": request_id,
            "object": "video",
            "status": self._map_fal_status(data.get("status")),
            "created_at": int(time.time()),
            "model": model,
        }

        if request_data:
            if "duration" in request_data:
                video_data["seconds"] = str(request_data["duration"])
            if "resolution" in request_data:
                video_data["size"] = str(request_data["resolution"])

        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]
        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(
                video_obj.id, custom_llm_provider, model
            )
        self._capture_billable_units(video_obj, raw_response)
        return video_obj

    # ------------------------------------------------------ status retrieve

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        original_id = extract_original_video_id(video_id)
        endpoint = self._endpoint_slug_for(video_id)
        base = self._queue_base(api_base)
        url = f"{base}/{endpoint}/requests/{original_id}/status"
        return url, {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        data = raw_response.json()
        request_id = data.get("request_id") or ""
        video_data: Dict[str, Any] = {
            "id": request_id,
            "object": "video",
            "status": self._map_fal_status(data.get("status")),
            "created_at": int(time.time()),
        }

        queue_position = data.get("queue_position")
        if queue_position is not None:
            try:
                video_data["progress"] = max(0, 100 - int(queue_position))
            except (TypeError, ValueError):
                pass

        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]
        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(
                video_obj.id, custom_llm_provider, None
            )
        self._capture_billable_units(video_obj, raw_response)
        return video_obj

    # ------------------------------------------------------- content fetch

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        variant: Optional[str] = None,
    ) -> Tuple[str, Dict]:
        original_id = extract_original_video_id(video_id)
        endpoint = self._endpoint_slug_for(video_id)
        base = self._queue_base(api_base)
        url = f"{base}/{endpoint}/requests/{original_id}"
        return url, {}

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        data = raw_response.json()
        video_url = self._extract_video_url(data)
        httpx_client: HTTPHandler = _get_httpx_client()
        video_response = httpx_client.get(video_url)
        video_response.raise_for_status()
        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        data = raw_response.json()
        video_url = self._extract_video_url(data)
        async_httpx_client: AsyncHTTPHandler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.FAL_AI,
        )
        video_response = await async_httpx_client.get(video_url)
        video_response.raise_for_status()
        return video_response.content

    # ------------------------------------------------------------- helpers

    def _map_fal_status(self, fal_status: Optional[str]) -> str:
        """
        Fal queue statuses → OpenAI statuses.

        Fal: ``IN_QUEUE``, ``IN_PROGRESS``, ``COMPLETED``, plus error states
        we map to ``failed`` defensively.
        """
        if not fal_status:
            return "queued"
        mapping = {
            "IN_QUEUE": "queued",
            "IN_PROGRESS": "in_progress",
            "COMPLETED": "completed",
            "FAILED": "failed",
            "ERROR": "failed",
            "CANCELLED": "failed",
        }
        return mapping.get(fal_status.upper(), "queued")

    def _capture_billable_units(
        self, video_obj: VideoObject, raw_response: httpx.Response
    ) -> None:
        """
        Capture Fal's ``x-fal-billable-units`` header onto ``_hidden_params``
        so the existing cost calculator (path 1) can multiply by the static
        unit price configured in ``FAL_UNIT_PRICES``.
        """
        units_header = raw_response.headers.get("x-fal-billable-units")
        if units_header is None:
            return
        try:
            hidden = video_obj._hidden_params or {}
            hidden["fal_billable_units"] = float(units_header)
            video_obj._hidden_params = hidden
        except (TypeError, ValueError):
            pass

    def _extract_video_url(self, response_data: Dict[str, Any]) -> str:
        video = response_data.get("video") or {}
        url = video.get("url") if isinstance(video, dict) else None
        if not url:
            raise ValueError(
                "Video URL not found in Fal response. Video may not be ready yet."
            )
        return url

    def _endpoint_slug_for(self, video_id: str) -> str:
        """
        Recover the endpoint slug for a poll/fetch request.

        ``encode_video_id_with_provider`` baked the model name into the
        ``video_id`` at create-time, so we round-trip it here. Falls back to
        the class-declared ``VIDEO_ENDPOINT`` so concrete subclasses keep
        working even if the encoded model is missing.
        """
        decoded = decode_video_id_with_provider(video_id)
        model = decoded.get("model_id") or ""
        if model:
            return _strip_provider_prefix(model)
        return self.VIDEO_ENDPOINT

    def _queue_base(self, api_base: Optional[str]) -> str:
        if api_base:
            return api_base.rstrip("/")
        return self.DEFAULT_QUEUE_URL

    # -------------------------------------------------- unsupported methods

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("Video remix is not supported for Fal AI")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported for Fal AI")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        extra_query: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("Video listing is not supported for Fal AI")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError("Video listing is not supported for Fal AI")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        original_id = extract_original_video_id(video_id)
        endpoint = self._endpoint_slug_for(video_id)
        base = self._queue_base(api_base)
        url = f"{base}/{endpoint}/requests/{original_id}/cancel"
        return url, {}

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        data = raw_response.json() if raw_response.content else {}
        return VideoObject(
            id=data.get("request_id", ""),
            object="video",
            status="cancelled",
            created_at=int(time.time()),
        )  # type: ignore[arg-type]
