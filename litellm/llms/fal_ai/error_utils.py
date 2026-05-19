"""
Shared error-classification helpers for Fal AI providers.

Both the image-generation (`FalAIBaseConfig`) and image-edit
(`FalAIImageEditConfig`) base classes need to override `get_error_class` so Fal
errors map to the right litellm exception types — otherwise they collapse to a
generic `BaseLLMException`, get wrapped as `APIConnectionError` (500), and slip
past Router.retry_policy's BadRequestError / ContentPolicyViolationError /
AuthenticationError matching. Router then falls back to num_retries=2 and burns
extra upstream calls on errors that will never succeed on retry.
"""

from typing import Union

import httpx

from litellm.exceptions import (
    AuthenticationError,
    BadRequestError,
    ContentPolicyViolationError,
    RateLimitError,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException


def classify_fal_ai_error(
    error_message: str,
    status_code: int,
    headers: Union[dict, httpx.Headers],
) -> BaseLLMException:
    """Return the right litellm exception type for a Fal AI error."""
    msg = error_message or ""
    msg_lower = msg.lower()
    if "content_policy_violation" in msg_lower or "content checker" in msg_lower:
        return ContentPolicyViolationError(
            message=msg, model="fal_ai", llm_provider="fal_ai"
        )
    if status_code == 401:
        return AuthenticationError(
            message=msg, model="fal_ai", llm_provider="fal_ai"
        )
    if status_code == 429:
        return RateLimitError(
            message=msg, model="fal_ai", llm_provider="fal_ai"
        )
    if 400 <= status_code < 500:
        return BadRequestError(
            message=msg, model="fal_ai", llm_provider="fal_ai"
        )
    return BaseLLMException(
        status_code=status_code, message=msg, headers=headers
    )
