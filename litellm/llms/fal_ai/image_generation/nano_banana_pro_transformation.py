from typing import Any, List

from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams

from .transformation import FalAIBaseConfig


class FalAINanoBananaProConfig(FalAIBaseConfig):
    """
    fal-ai/nano-banana-pro — Google Gemini 2.5 Flash Image (Pro tier).

    Endpoint: https://fal.run/fal-ai/nano-banana-pro
    Pricing entry: ``fal_ai/fal-ai/nano-banana-pro`` (output_cost_per_image).
    Docs: https://fal.ai/models/fal-ai/nano-banana-pro

    Pricing is flat per image — no quality tier. Fal-specific knobs
    (``aspect_ratio``, ``resolution``, ``output_format``,
    ``safety_tolerance``) flow through ``extra_body``.
    """

    IMAGE_GENERATION_ENDPOINT: str = "fal-ai/nano-banana-pro"

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        return ["n"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported = self.get_supported_openai_params(model)
        param_mapping = {"n": "num_images"}

        for k, v in non_default_params.items():
            if k in optional_params:
                continue
            if k not in supported:
                if drop_params:
                    continue
                raise ValueError(
                    f"Parameter {k} is not supported for model {model}. "
                    f"Supported parameters are {supported}. "
                    "Set drop_params=True to drop unsupported parameters."
                )
            optional_params[param_mapping.get(k, k)] = v

        return optional_params

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        return {"prompt": prompt, **optional_params}
