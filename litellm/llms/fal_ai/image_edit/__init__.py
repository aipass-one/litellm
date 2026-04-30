from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig

from .birefnet_transformation import FalAIBirefnetEditConfig
from .clarity_upscaler_transformation import FalAIClarityUpscalerEditConfig
from .transformation import FalAIGptImage2EditConfig

__all__ = [
    "FalAIGptImage2EditConfig",
    "FalAIClarityUpscalerEditConfig",
    "FalAIBirefnetEditConfig",
    "get_fal_ai_image_edit_config",
]


def get_fal_ai_image_edit_config(model: str) -> BaseImageEditConfig:
    """
    Resolve the right Fal AI image-edit config for the given model.

    Currently wired up:
    - ``openai/gpt-image-2[/edit]`` — token-based partner endpoint
    - ``fal-ai/clarity-upscaler`` — pixel-based upscaler
    - ``fal-ai/birefnet`` — flat-priced background removal
    """
    model_lower = model.lower()
    if "gpt-image-2" in model_lower:
        return FalAIGptImage2EditConfig()
    if "clarity-upscaler" in model_lower:
        return FalAIClarityUpscalerEditConfig()
    if "birefnet" in model_lower:
        return FalAIBirefnetEditConfig()
    raise NotImplementedError(
        f"No fal_ai image-edit config for model={model!r}. "
        "Supported: gpt-image-2, clarity-upscaler, birefnet."
    )
