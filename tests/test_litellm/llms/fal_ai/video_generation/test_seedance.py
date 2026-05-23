"""
Seedance v2 (standard + fast) — dispatcher + param mapping + cost path.

The shared queue lifecycle is covered in ``test_base.py``. These tests
exercise the seedance-specific surface: endpoint slugs, OpenAI-style
param translation (size → resolution, seconds → duration), and cost
calculation through PR #22's header path.
"""
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.fal_ai.cost_calculator import cost_calculator
from litellm.llms.fal_ai.video_generation import (
    FalAIBaseVideoConfig,
    SeedanceV2Config,
    SeedanceV2FastConfig,
    get_fal_ai_video_generation_config,
)
from litellm.types.videos.main import VideoObject


# -------------------------------------------------------------- dispatcher


@pytest.mark.parametrize(
    "model,expected_class",
    [
        # standard tier
        ("fal_ai/bytedance/seedance-2.0/image-to-video", SeedanceV2Config),
        ("bytedance/seedance-2.0/image-to-video", SeedanceV2Config),
        # fast tier — must NOT match the bare "seedance-2.0" branch
        ("fal_ai/bytedance/seedance-2.0/fast/image-to-video", SeedanceV2FastConfig),
        ("bytedance/seedance-2.0/fast/image-to-video", SeedanceV2FastConfig),
    ],
)
def test_dispatcher(model, expected_class):
    config = get_fal_ai_video_generation_config(model)
    assert isinstance(config, expected_class)
    assert type(config) is expected_class


def test_dispatcher_unknown_returns_base():
    config = get_fal_ai_video_generation_config("fal_ai/unknown/video-model")
    assert isinstance(config, FalAIBaseVideoConfig)


# ------------------------------------------------------------- endpoints


def test_standard_endpoint_slug():
    assert (
        SeedanceV2Config().VIDEO_ENDPOINT
        == "bytedance/seedance-2.0/image-to-video"
    )


def test_fast_endpoint_slug():
    assert (
        SeedanceV2FastConfig().VIDEO_ENDPOINT
        == "bytedance/seedance-2.0/fast/image-to-video"
    )


def test_standard_get_complete_url():
    config = SeedanceV2Config()
    url = config.get_complete_url(
        model="fal_ai/bytedance/seedance-2.0/image-to-video",
        api_base=None,
        litellm_params={},
    )
    assert url == (
        "https://queue.fal.run/bytedance/seedance-2.0/image-to-video"
    )


def test_fast_get_complete_url():
    config = SeedanceV2FastConfig()
    url = config.get_complete_url(
        model="fal_ai/bytedance/seedance-2.0/fast/image-to-video",
        api_base=None,
        litellm_params={},
    )
    assert url == (
        "https://queue.fal.run/bytedance/seedance-2.0/fast/image-to-video"
    )


# ---------------------------------------------------- size → resolution


@pytest.mark.parametrize(
    "size,expected",
    [
        ("480x720", "480p"),
        ("720x720", "720p"),
        ("1280x720", "720p"),
        ("1920x1080", "1080p"),
        ("4096x4096", "1080p"),     # >1080 clamps to the top tier
        ("256x256", "480p"),        # below the smallest tier
        ("invalid", None),
        ("", None),
    ],
)
def test_size_to_resolution(size, expected):
    assert SeedanceV2Config._size_to_resolution(size) == expected


def test_size_to_resolution_non_string():
    assert SeedanceV2Config._size_to_resolution(None) is None  # type: ignore[arg-type]
    assert SeedanceV2Config._size_to_resolution(1024) is None  # type: ignore[arg-type]


# --------------------------------------------------------- param mapping


class TestMapOpenAIParams:
    def setup_method(self):
        self.config = SeedanceV2Config()

    def test_input_reference_maps_to_image_url(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "input_reference": "https://x/cat.jpg",
            },
            model="fal_ai/bytedance/seedance-2.0/image-to-video",
            drop_params=False,
        )
        assert mapped["image_url"] == "https://x/cat.jpg"
        assert "input_reference" not in mapped

    def test_size_translates_to_resolution(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "input_reference": "https://x/y.jpg",
                "size": "1920x1080",
            },
            model="fal_ai/bytedance/seedance-2.0/image-to-video",
            drop_params=False,
        )
        assert mapped["resolution"] == "1080p"
        assert "size" not in mapped

    def test_seconds_translates_to_duration(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "input_reference": "https://x/y.jpg",
                "seconds": 8,
            },
            model="fal_ai/bytedance/seedance-2.0/image-to-video",
            drop_params=False,
        )
        assert mapped["duration"] == "8"
        assert "seconds" not in mapped

    def test_extra_body_passthrough(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "input_reference": "https://x/y.jpg",
                "extra_body": {
                    "resolution": "720p",
                    "aspect_ratio": "16:9",
                    "generate_audio": False,
                    "end_image_url": "https://x/end.jpg",
                    "seed": 42,
                },
            },
            model="fal_ai/bytedance/seedance-2.0/image-to-video",
            drop_params=False,
        )
        assert mapped["resolution"] == "720p"
        assert mapped["aspect_ratio"] == "16:9"
        assert mapped["generate_audio"] is False
        assert mapped["end_image_url"] == "https://x/end.jpg"
        assert mapped["seed"] == 42
        assert "extra_body" not in mapped

    def test_extra_body_does_not_overwrite_explicit_size_translation(self):
        # If user passes both size and extra_body.resolution, size wins
        # because setdefault preserves the already-translated value.
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "input_reference": "https://x/y.jpg",
                "size": "1920x1080",
                "extra_body": {"resolution": "480p"},
            },
            model="fal_ai/bytedance/seedance-2.0/image-to-video",
            drop_params=False,
        )
        # extra_body update happens before size translation in
        # super().map_openai_params, so resolution from extra_body lands
        # first; size translation then uses setdefault and skips it.
        assert mapped["resolution"] == "480p"


# ----------------------------------------------------------- cost path


def _video_with_units(units):
    obj = VideoObject(
        id="v_1",
        object="video",
        status="completed",
        created_at=0,
    )
    obj._hidden_params = {"fal_billable_units": units}
    return obj


def test_cost_standard_header_path():
    obj = _video_with_units(5.0)
    cost = cost_calculator(
        model="fal_ai/bytedance/seedance-2.0/image-to-video",
        image_response=obj,
    )
    # 5 output seconds × $0.3024 = $1.512
    assert cost == pytest.approx(1.512, abs=1e-6)


def test_cost_fast_header_path():
    obj = _video_with_units(5.0)
    cost = cost_calculator(
        model="fal_ai/bytedance/seedance-2.0/fast/image-to-video",
        image_response=obj,
    )
    # 5 output seconds × $0.2419 = $1.2095
    assert cost == pytest.approx(1.2095, abs=1e-6)


def test_cost_no_header_falls_back_and_marks_reconcile():
    obj = VideoObject(
        id="v_2",
        object="video",
        status="completed",
        created_at=0,
    )
    obj._hidden_params = {}
    cost = cost_calculator(
        model="fal_ai/bytedance/seedance-2.0/image-to-video",
        image_response=obj,
    )
    # No header, no per-image, falls through to unit_price * 5
    assert cost == pytest.approx(0.3024 * 5, abs=1e-6)
    assert obj._hidden_params.get("needs_reconcile") is True


def test_cost_unknown_model_returns_zero():
    obj = _video_with_units(5.0)
    cost = cost_calculator(
        model="fal_ai/unknown/something",
        image_response=obj,
    )
    assert cost == 0.0


# ---------------------------------------------------------- e2e shape


def test_create_request_body_contains_image_url():
    config = SeedanceV2Config()
    body, _, _ = config.transform_video_create_request(
        model="fal_ai/bytedance/seedance-2.0/image-to-video",
        prompt="cat blinks",
        api_base="https://queue.fal.run/bytedance/seedance-2.0/image-to-video",
        video_create_optional_request_params={
            "image_url": "https://x/cat.jpg",
            "resolution": "720p",
            "duration": "5",
        },
        litellm_params=MagicMock(),
        headers={},
    )
    assert body == {
        "prompt": "cat blinks",
        "image_url": "https://x/cat.jpg",
        "resolution": "720p",
        "duration": "5",
    }
