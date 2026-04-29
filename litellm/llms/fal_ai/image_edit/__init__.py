from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig

from .handler import FalAIImageEdit, fal_ai_image_edit
from .transformation import FalAIGptImage2EditConfig

__all__ = [
    "FalAIGptImage2EditConfig",
    "FalAIImageEdit",
    "fal_ai_image_edit",
    "get_fal_ai_image_edit_config",
]


def get_fal_ai_image_edit_config(model: str) -> BaseImageEditConfig:
    """
    Resolve the right Fal AI image-edit config for the given model.

    Today only ``openai/gpt-image-2`` (and its ``/edit`` variant) is wired up.
    Future Fal edit models (Flux Kontext, Seedream v4 Edit, Nano Banana Edit
    via Fal, etc.) plug in here.
    """
    model_lower = model.lower()
    if "gpt-image-2" in model_lower:
        return FalAIGptImage2EditConfig()
    return FalAIGptImage2EditConfig()
