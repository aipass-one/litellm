"""
Tests for Fal AI error classification.

Without these get_error_class overrides, the base BaseImageGenerationConfig /
BaseImageEditConfig implementations return a generic BaseLLMException — which
the central exception_mapper then wraps as APIConnectionError (status 500).
That means deterministic upstream failures (content_policy_violation, malformed
prompt, bad credentials) get presented to the caller as 500s, and they slip
past Router retry_policy rules that key off ContentPolicyViolationError /
BadRequestError / AuthenticationError — so Router falls back to its default
num_retries=2 and we burn ~6-11s and 2 extra Fal calls on a known-deterministic
failure.

These tests pin the behaviour we want: Fal errors are classified into the right
litellm exception types so retry_policy + downstream HTTP status mapping work
correctly. Parametrized across both base classes (image generation + image
edit) so a regression in either branch fails loudly.
"""

import pytest

from litellm.exceptions import (
    AuthenticationError,
    BadRequestError,
    ContentPolicyViolationError,
    RateLimitError,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.fal_ai.image_edit.aura_sr_transformation import FalAIAuraSREditConfig
from litellm.llms.fal_ai.image_generation.flux_pro_v11_transformation import (
    FalAIFluxProV11Config,
)


@pytest.fixture(
    params=[FalAIFluxProV11Config, FalAIAuraSREditConfig],
    ids=["image_generation", "image_edit"],
)
def fal_config(request):
    return request.param()


def test_content_policy_violation_phrase_returns_content_policy_error(fal_config):
    """
    Fal returns the literal string `content_policy_violation` in the JSON body
    when its safety filter blocks a prompt. Surface as ContentPolicyViolationError
    so HTTP response is 400 (not 500) and Router skips retries.
    """
    err = fal_config.get_error_class(
        error_message='{"detail":"content_policy_violation: NSFW detected"}',
        status_code=422,
        headers={},
    )
    assert isinstance(err, ContentPolicyViolationError)
    assert err.status_code == 400


def test_safety_checker_phrase_returns_content_policy_error(fal_config):
    """
    Fal's older models phrase the same rejection as `... content checker ...`
    rather than the structured `content_policy_violation` code. Catch both.
    """
    err = fal_config.get_error_class(
        error_message="The safety content checker has triggered, blocking the request.",
        status_code=422,
        headers={},
    )
    assert isinstance(err, ContentPolicyViolationError)


def test_authentication_status_returns_auth_error(fal_config):
    err = fal_config.get_error_class(
        error_message="Invalid API key",
        status_code=401,
        headers={},
    )
    assert isinstance(err, AuthenticationError)
    assert err.status_code == 401


def test_rate_limit_status_returns_rate_limit_error(fal_config):
    err = fal_config.get_error_class(
        error_message="rate_limit_exceeded",
        status_code=429,
        headers={},
    )
    assert isinstance(err, RateLimitError)
    assert err.status_code == 429


def test_generic_4xx_returns_bad_request_error(fal_config):
    """
    Catch-all for client errors that aren't policy / auth / rate-limit (e.g.
    malformed prompt, invalid model id) so they don't get retried by Router and
    don't surface as 500s.
    """
    err = fal_config.get_error_class(
        error_message='{"detail":[{"type":"string_too_short","loc":["body","prompt"]}]}',
        status_code=422,
        headers={},
    )
    assert isinstance(err, BadRequestError)
    # Not a ContentPolicyViolationError — purely a validation failure.
    assert not isinstance(err, ContentPolicyViolationError)


def test_5xx_returns_base_llm_exception_with_original_status(fal_config):
    """
    Server-side / transient errors stay as BaseLLMException so the central
    exception_mapper can decide what to do (typically wrap as
    InternalServerError, which Router DOES retry — and should).
    """
    err = fal_config.get_error_class(
        error_message="upstream timeout",
        status_code=502,
        headers={},
    )
    assert isinstance(err, BaseLLMException)
    assert not isinstance(err, BadRequestError)
    assert err.status_code == 502


def test_returns_exception_does_not_raise(fal_config):
    """
    Caller does `raise self.get_error_class(...)`. The method must RETURN the
    exception, not raise it directly — the base impl was buggy on this point.
    """
    result = fal_config.get_error_class(
        error_message="anything",
        status_code=400,
        headers={},
    )
    assert isinstance(result, BaseException)
