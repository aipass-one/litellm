"""
ByteDance Seedance 2.0 image-to-video on Fal AI.

Endpoint: ``bytedance/seedance-2.0/image-to-video``
Pricing : $0.3024 per output second (header-emitting via PR #22 path)

Fal-native request body
-----------------------
``prompt`` (str), ``image_url`` (str), optional: ``end_image_url`` (str),
``resolution`` ("480p" | "720p" | "1080p", default "720p"),
``duration`` ("auto" | "4"–"15", default "auto"),
``aspect_ratio`` ("auto" | "21:9" | "16:9" | "4:3" | "1:1" | "3:4" | "9:16"),
``generate_audio`` (bool, default True), ``seed`` (int).

We accept all of the above via ``extra_body``. We also translate two
OpenAI-style params for convenience:

* OpenAI ``size`` ("WIDTHxHEIGHT") → Fal ``resolution`` (nearest tier)
* OpenAI ``seconds`` (str / int) → Fal ``duration`` (str)
"""

from typing import Any, ClassVar, Dict, List, Optional

from litellm.types.videos.main import VideoCreateOptionalRequestParams

from .base import FalAIBaseVideoConfig


class SeedanceV2Config(FalAIBaseVideoConfig):
    VIDEO_ENDPOINT: ClassVar[str] = "bytedance/seedance-2.0/image-to-video"

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

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        mapped = super().map_openai_params(
            video_create_optional_params, model, drop_params
        )

        if "size" in mapped:
            resolution = self._size_to_resolution(mapped.pop("size"))
            if resolution is not None:
                mapped.setdefault("resolution", resolution)

        if "seconds" in mapped:
            mapped.setdefault("duration", str(mapped.pop("seconds")))

        return mapped

    @staticmethod
    def _size_to_resolution(size: Any) -> Optional[str]:
        """
        Translate OpenAI ``size`` ("WIDTHxHEIGHT") to Fal ``resolution``
        ("480p" / "720p" / "1080p"). Picks the nearest tier from the
        shorter side.
        """
        if not isinstance(size, str) or "x" not in size.lower():
            return None
        try:
            w_str, h_str = size.lower().replace(" ", "").split("x", 1)
            shorter = min(int(w_str), int(h_str))
        except (ValueError, AttributeError):
            return None
        if shorter <= 480:
            return "480p"
        if shorter <= 720:
            return "720p"
        return "1080p"
