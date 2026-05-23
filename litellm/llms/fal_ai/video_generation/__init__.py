from litellm.llms.base_llm.videos.transformation import BaseVideoConfig

from .base import FalAIBaseVideoConfig
from .seedance_v2_transformation import SeedanceV2Config
from .seedance_v2_fast_transformation import SeedanceV2FastConfig

__all__ = [
    "FalAIBaseVideoConfig",
    "SeedanceV2Config",
    "SeedanceV2FastConfig",
    "get_fal_ai_video_generation_config",
]


def get_fal_ai_video_generation_config(model: str) -> BaseVideoConfig:
    """
    Pick the right Fal AI video config for a model string.

    LiteLLM passes the model as ``fal_ai/{endpoint_slug}``. We match on the
    slug (case-insensitive). Most-specific patterns are checked first so that
    ``seedance-2.0/fast`` is not eaten by the broader ``seedance-2.0`` match.
    """
    model_lower = model.lower()

    if "seedance-2.0/fast" in model_lower:
        return SeedanceV2FastConfig()
    if "seedance-2.0" in model_lower:
        return SeedanceV2Config()

    # Default: the base config can serve any Fal video endpoint that follows
    # the standard queue API contract (prompt + image_url body, COMPLETED
    # status, video.url result). The endpoint slug is derived from the model
    # name in get_complete_url, so this still hits the right URL.
    config = FalAIBaseVideoConfig()
    return config
