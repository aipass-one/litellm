from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig

from .aura_sr_transformation import FalAIAuraSREditConfig
from .base import FalAIImageEditConfig
from .ben_v2_transformation import FalAIBenV2EditConfig
from .birefnet_transformation import FalAIBirefnetEditConfig
from .birefnet_v2_transformation import FalAIBirefnetV2EditConfig
from .clarity_upscaler_transformation import FalAIClarityUpscalerEditConfig
from .esrgan_transformation import FalAIEsrganEditConfig
from .nano_banana_2_edit_transformation import FalAINanoBanana2EditConfig
from .nano_banana_pro_edit_transformation import FalAINanoBananaProEditConfig
from .recraft_upscale_creative_transformation import (
    FalAIRecraftUpscaleCreativeEditConfig,
)
from .recraft_upscale_crisp_transformation import (
    FalAIRecraftUpscaleCrispEditConfig,
)
from .topaz_upscale_transformation import FalAITopazUpscaleEditConfig
from .transformation import FalAIGptImage2EditConfig

__all__ = [
    "FalAIImageEditConfig",
    "FalAIGptImage2EditConfig",
    "FalAIClarityUpscalerEditConfig",
    "FalAIBirefnetEditConfig",
    "FalAIBirefnetV2EditConfig",
    "FalAIBenV2EditConfig",
    "FalAIAuraSREditConfig",
    "FalAIEsrganEditConfig",
    "FalAITopazUpscaleEditConfig",
    "FalAIRecraftUpscaleCrispEditConfig",
    "FalAIRecraftUpscaleCreativeEditConfig",
    "FalAINanoBananaProEditConfig",
    "FalAINanoBanana2EditConfig",
    "get_fal_ai_image_edit_config",
]


def get_fal_ai_image_edit_config(model: str) -> BaseImageEditConfig:
    """
    Resolve the right Fal AI image-edit config for the given model.

    Branches are evaluated in *most-specific-first* order so that, e.g.,
    ``birefnet/v2`` doesn't get swallowed by the ``birefnet`` branch.
    """
    m = model.lower()
    # Nano Banana family (multi-image gen+edit) — both contain unique substrings
    if "nano-banana-pro" in m:
        return FalAINanoBananaProEditConfig()
    if "nano-banana-2" in m:
        return FalAINanoBanana2EditConfig()
    # gpt-image-2 (composite-key tiered pricing)
    if "gpt-image-2" in m:
        return FalAIGptImage2EditConfig()
    # Recraft pair — exact suffix differentiates
    if "recraft/upscale/crisp" in m:
        return FalAIRecraftUpscaleCrispEditConfig()
    if "recraft/upscale/creative" in m:
        return FalAIRecraftUpscaleCreativeEditConfig()
    # Topaz
    if "topaz/upscale" in m:
        return FalAITopazUpscaleEditConfig()
    # BiRefNet — v2 must come before v1
    if "birefnet/v2" in m:
        return FalAIBirefnetV2EditConfig()
    if "birefnet" in m:
        return FalAIBirefnetEditConfig()
    # BEN
    if "ben/v2" in m:
        return FalAIBenV2EditConfig()
    # Single-name upscalers
    if "aura-sr" in m:
        return FalAIAuraSREditConfig()
    if "esrgan" in m:
        return FalAIEsrganEditConfig()
    if "clarity-upscaler" in m:
        return FalAIClarityUpscalerEditConfig()
    raise NotImplementedError(
        f"No fal_ai image-edit config for model={model!r}. "
        "Supported: gpt-image-2, clarity-upscaler, birefnet, birefnet/v2, "
        "ben/v2, aura-sr, esrgan, topaz/upscale, recraft/upscale/crisp, "
        "recraft/upscale/creative, nano-banana-pro, nano-banana-2."
    )
