"""
ByteDance Seedance 2.0 *Fast* image-to-video on Fal AI.

Endpoint: ``bytedance/seedance-2.0/fast/image-to-video``
Pricing : $0.2419 per output second (header-emitting via PR #22 path)

Identical request/response shape to the standard tier — caps at 720p instead
of 1080p, but the API doesn't reject 1080p requests, it just renders at 720p.
"""

from typing import ClassVar

from .seedance_v2_transformation import SeedanceV2Config


class SeedanceV2FastConfig(SeedanceV2Config):
    VIDEO_ENDPOINT: ClassVar[str] = "bytedance/seedance-2.0/fast/image-to-video"
