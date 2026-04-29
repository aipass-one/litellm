"""
Fal AI image edit handler — async queue mode.

Fal exposes the edit endpoint via two transports:
  * sync ``fal.run/<path>``     — blocking, ~60s wall clock, no cost echo
  * async ``queue.fal.run/<path>`` — submit → poll status → fetch result

We use the queue path because:
  * higher-quality / larger edits exceed the 60s sync cap
  * ``queue.result`` includes per-request metrics that may include cost
  * it aligns with Fal's recommended production pattern

The flow is the same shape as Black Forest Labs' image-edit handler at
``litellm/llms/black_forest_labs/image_edit/handler.py`` — submit, poll,
fetch — adapted to Fal's separate ``status_url`` / ``response_url`` model.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageResponse

from .transformation import FalAIGptImage2EditConfig

DEFAULT_QUEUE_BASE_URL = "https://queue.fal.run"
DEFAULT_POLLING_INTERVAL = 1.5
DEFAULT_MAX_POLLING_TIME = 240.0
TERMINAL_FAILURE_STATUSES = {"FAILED", "ERROR", "CANCELLED"}


class FalAIImageEditError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(message)


class FalAIImageEdit:
    """Submit → poll → fetch handler for Fal AI image-edit endpoints.

    Reuses ``FalAIGptImage2EditConfig`` for request body construction and
    response parsing; this class only orchestrates the HTTP dance.
    """

    def __init__(self):
        self.config = FalAIGptImage2EditConfig()

    def image_edit(
        self,
        model: str,
        image: Union[FileTypes, List[FileTypes]],
        prompt: Optional[str],
        image_edit_optional_request_params: Dict,
        litellm_params: Union[GenericLiteLLMParams, Dict],
        logging_obj: LiteLLMLoggingObj,
        timeout: Optional[Union[float, httpx.Timeout]],
        extra_headers: Optional[Dict[str, Any]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        aimage_edit: bool = False,
    ) -> Union[ImageResponse, Any]:
        if isinstance(litellm_params, dict):
            api_key = litellm_params.get("api_key")
            api_base = litellm_params.get("api_base")
            litellm_params_dict = litellm_params
        else:
            api_key = litellm_params.api_key
            api_base = litellm_params.api_base
            litellm_params_dict = dict(litellm_params)

        if aimage_edit:
            return self._async_image_edit(
                model=model,
                image=image,
                prompt=prompt,
                image_edit_optional_request_params=image_edit_optional_request_params,
                api_key=api_key,
                api_base=api_base,
                litellm_params_dict=litellm_params_dict,
                logging_obj=logging_obj,
                timeout=timeout,
                extra_headers=extra_headers,
                client=client if isinstance(client, AsyncHTTPHandler) else None,
            )

        sync_client: HTTPHandler = (
            client if isinstance(client, HTTPHandler) else _get_httpx_client()
        )

        headers, submit_url, request_body = self._prepare_call(
            model=model,
            image=image,
            prompt=prompt,
            image_edit_optional_request_params=image_edit_optional_request_params,
            api_key=api_key,
            api_base=api_base,
            litellm_params_dict=litellm_params_dict,
            extra_headers=extra_headers,
        )

        logging_obj.pre_call(
            input=prompt,
            api_key="",
            additional_args={
                "complete_input_dict": {
                    **request_body,
                    "image_urls": [
                        f"<{len(u)} bytes>" if u.startswith("data:") else u
                        for u in request_body.get("image_urls", [])
                    ],
                },
                "api_base": submit_url,
                "headers": headers,
            },
        )

        submit_response = self._post_sync(sync_client, submit_url, headers, request_body, timeout)
        status_url, response_url = self._parse_submit(submit_response)
        self._poll_sync(sync_client, status_url, headers, timeout)
        result_response = self._get_sync(sync_client, response_url, headers, timeout)

        return self._finalize_response(model, result_response, request_body, logging_obj)

    async def _async_image_edit(
        self,
        model: str,
        image: Union[FileTypes, List[FileTypes]],
        prompt: Optional[str],
        image_edit_optional_request_params: Dict,
        api_key: Optional[str],
        api_base: Optional[str],
        litellm_params_dict: Dict,
        logging_obj: LiteLLMLoggingObj,
        timeout: Optional[Union[float, httpx.Timeout]],
        extra_headers: Optional[Dict[str, Any]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> ImageResponse:
        async_client = client or get_async_httpx_client(
            llm_provider=litellm.LlmProviders.FAL_AI,
        )

        headers, submit_url, request_body = self._prepare_call(
            model=model,
            image=image,
            prompt=prompt,
            image_edit_optional_request_params=image_edit_optional_request_params,
            api_key=api_key,
            api_base=api_base,
            litellm_params_dict=litellm_params_dict,
            extra_headers=extra_headers,
        )

        logging_obj.pre_call(
            input=prompt,
            api_key="",
            additional_args={
                "complete_input_dict": {
                    **request_body,
                    "image_urls": [
                        f"<{len(u)} bytes>" if u.startswith("data:") else u
                        for u in request_body.get("image_urls", [])
                    ],
                },
                "api_base": submit_url,
                "headers": headers,
            },
        )

        submit_response = await self._post_async(async_client, submit_url, headers, request_body, timeout)
        status_url, response_url = self._parse_submit(submit_response)
        await self._poll_async(async_client, status_url, headers, timeout)
        result_response = await self._get_async(async_client, response_url, headers, timeout)

        return self._finalize_response(model, result_response, request_body, logging_obj)

    def _prepare_call(
        self,
        *,
        model: str,
        image: Union[FileTypes, List[FileTypes]],
        prompt: Optional[str],
        image_edit_optional_request_params: Dict,
        api_key: Optional[str],
        api_base: Optional[str],
        litellm_params_dict: Dict,
        extra_headers: Optional[Dict[str, Any]],
    ):
        headers = self.config.validate_environment(
            api_key=api_key,
            headers=image_edit_optional_request_params.get("extra_headers", {}) or {},
            model=model,
        )
        if extra_headers:
            headers.update(extra_headers)

        queue_base = (api_base or DEFAULT_QUEUE_BASE_URL).rstrip("/")
        if "queue.fal.run" not in queue_base:
            queue_base = DEFAULT_QUEUE_BASE_URL
        submit_url = f"{queue_base}/{self.config.EDIT_ENDPOINT}"

        request_body, _ = self.config.transform_image_edit_request(
            model=model,
            prompt=prompt or "",
            image=image,
            image_edit_optional_request_params=image_edit_optional_request_params,
            litellm_params=litellm_params_dict,
            headers=headers,
        )
        return headers, submit_url, request_body

    @staticmethod
    def _parse_submit(submit_response: httpx.Response):
        if submit_response.status_code >= 400:
            raise FalAIImageEditError(
                status_code=submit_response.status_code,
                message=f"Fal queue submit failed: {submit_response.text[:500]}",
            )
        try:
            payload = submit_response.json()
        except Exception as e:
            raise FalAIImageEditError(
                status_code=submit_response.status_code,
                message=f"Error parsing submit response: {e}",
            )
        status_url = payload.get("status_url")
        response_url = payload.get("response_url")
        if not status_url or not response_url:
            raise FalAIImageEditError(
                status_code=500,
                message=f"Fal queue submit missing status_url/response_url: {payload}",
            )
        return status_url, response_url

    def _poll_sync(
        self,
        sync_client: HTTPHandler,
        status_url: str,
        headers: dict,
        timeout: Optional[Union[float, httpx.Timeout]],
    ) -> None:
        start = time.time()
        while time.time() - start < DEFAULT_MAX_POLLING_TIME:
            response = sync_client.get(url=status_url, headers=headers)
            self._raise_if_terminal_error(response)
            data = response.json()
            status = (data.get("status") or "").upper()
            verbose_logger.debug(f"fal queue poll status={status}")
            if status == "COMPLETED":
                return
            time.sleep(DEFAULT_POLLING_INTERVAL)
        raise FalAIImageEditError(
            status_code=408,
            message=f"Fal queue polling timed out after {DEFAULT_MAX_POLLING_TIME}s",
        )

    async def _poll_async(
        self,
        async_client: AsyncHTTPHandler,
        status_url: str,
        headers: dict,
        timeout: Optional[Union[float, httpx.Timeout]],
    ) -> None:
        start = time.time()
        while time.time() - start < DEFAULT_MAX_POLLING_TIME:
            response = await async_client.get(url=status_url, headers=headers)
            self._raise_if_terminal_error(response)
            data = response.json()
            status = (data.get("status") or "").upper()
            verbose_logger.debug(f"fal queue poll status={status}")
            if status == "COMPLETED":
                return
            await asyncio.sleep(DEFAULT_POLLING_INTERVAL)
        raise FalAIImageEditError(
            status_code=408,
            message=f"Fal queue polling timed out after {DEFAULT_MAX_POLLING_TIME}s",
        )

    @staticmethod
    def _raise_if_terminal_error(response: httpx.Response) -> None:
        if response.status_code >= 400:
            raise FalAIImageEditError(
                status_code=response.status_code,
                message=f"Fal queue poll failed: {response.text[:500]}",
            )
        try:
            data = response.json()
        except Exception:
            return
        status = (data.get("status") or "").upper()
        if status in TERMINAL_FAILURE_STATUSES:
            raise FalAIImageEditError(
                status_code=400,
                message=f"Fal queue request {status}: {data}",
            )

    @staticmethod
    def _post_sync(client, url, headers, json_body, timeout):
        try:
            return client.post(url=url, headers=headers, json=json_body, timeout=timeout)
        except Exception as e:
            raise FalAIImageEditError(status_code=500, message=f"Submit POST failed: {e}")

    @staticmethod
    async def _post_async(client, url, headers, json_body, timeout):
        try:
            return await client.post(
                url=url, headers=headers, json=json_body, timeout=timeout
            )
        except Exception as e:
            raise FalAIImageEditError(status_code=500, message=f"Submit POST failed: {e}")

    @staticmethod
    def _get_sync(client, url, headers, timeout):
        try:
            return client.get(url=url, headers=headers)
        except Exception as e:
            raise FalAIImageEditError(status_code=500, message=f"Result GET failed: {e}")

    @staticmethod
    async def _get_async(client, url, headers, timeout):
        try:
            return await client.get(url=url, headers=headers)
        except Exception as e:
            raise FalAIImageEditError(status_code=500, message=f"Result GET failed: {e}")

    def _finalize_response(
        self,
        model: str,
        result_response: httpx.Response,
        request_body: Dict[str, Any],
        logging_obj: LiteLLMLoggingObj,
    ) -> ImageResponse:
        if result_response.status_code >= 400:
            raise FalAIImageEditError(
                status_code=result_response.status_code,
                message=f"Fal queue result fetch failed: {result_response.text[:500]}",
            )

        image_response = self.config.transform_image_edit_response(
            model=model, raw_response=result_response, logging_obj=logging_obj
        )

        # Stamp quality + size from the request body we built; Fal's queue
        # result GET strips the original request, so the transformation's
        # request-payload parser can't find them.
        image_response.quality = request_body.get("quality") or "high"

        explicit_size = request_body.get("image_size")
        if isinstance(explicit_size, dict):
            w, h = explicit_size.get("width"), explicit_size.get("height")
            if w and h:
                image_response.size = f"{w}-x-{h}"
        elif (
            isinstance(explicit_size, str)
            and explicit_size in FalAIGptImage2EditConfig._PRESET_TO_SIZE
        ):
            w, h = FalAIGptImage2EditConfig._PRESET_TO_SIZE[explicit_size]
            image_response.size = f"{w}-x-{h}"
        # Else leave whatever the transformation set from the response itself.

        # Best-effort: log Fal's per-request metrics if present so we can
        # iterate towards exact-cost extraction later.
        try:
            payload = result_response.json()
            if isinstance(payload, dict):
                metrics = payload.get("metrics") or payload.get("billing")
                if metrics:
                    verbose_logger.debug(f"fal queue result metrics: {metrics}")
        except Exception:
            pass

        return image_response


fal_ai_image_edit = FalAIImageEdit()
