"""
Tests for the Fal AI cost calculator's three paths.

After PR ``feat/fal-billable-units-header`` the calculator no longer relies
on a quality×size matrix in ``model_prices_and_context_window.json``.
Instead it picks one of three paths in priority order:

1. **Header present** — ``image_response._hidden_params["fal_billable_units"]``
   was stamped by the transformation from Fal's ``x-fal-billable-units``
   response header. Multiplied by the static unit_price for the model.
2. **Fixed per-image** — ``unit_kind == "images"`` and no header. Cost is
   ``num_images_returned × unit_price``.
3. **Compute-second flat** — ``unit_kind == "compute_seconds"`` and no
   header. Returns the per-model entry in ``COMPUTE_SECOND_FALLBACK`` and
   marks ``_hidden_params["needs_reconcile"] = True``.
"""
import pytest

from litellm.llms.fal_ai.cost_calculator import (
    COMPUTE_SECOND_FALLBACK,
    FAL_UNIT_PRICES,
    cost_calculator,
)
from litellm.types.utils import ImageObject, ImageResponse


def _response_with_header(billable_units: str) -> ImageResponse:
    """
    Build an ImageResponse with one image and the Fal header value already
    stamped on _hidden_params (mimics what the transformation does).
    """
    response = ImageResponse(
        data=[ImageObject(url="https://example.com/x.png", b64_json=None)]
    )
    response._hidden_params = {"fal_billable_units": float(billable_units)}
    return response


def _response_without_header(num_images: int = 1) -> ImageResponse:
    response = ImageResponse(
        data=[
            ImageObject(url=f"https://example.com/{i}.png", b64_json=None)
            for i in range(num_images)
        ]
    )
    response._hidden_params = {}
    return response


# ----------------------------------------------------------------- header path

@pytest.mark.parametrize(
    "model,header_value,expected_cost",
    [
        # gpt-image-2: $1/unit. header value IS the dollar cost.
        ("fal_ai/openai/gpt-image-2",      "0.156",   0.156),
        ("fal_ai/openai/gpt-image-2",      "0.006",   0.006),
        ("fal_ai/openai/gpt-image-2/edit", "0.221",   0.221),
        # clarity-upscaler: $0.03/megapixel
        ("fal_ai/fal-ai/clarity-upscaler", "2.54",    2.54 * 0.03),
        # topaz: $0.01/megapixel
        ("fal_ai/fal-ai/topaz/upscale/image", "8.0",   8.0 * 0.01),
        # ben/v2: $0.025/megapixel
        ("fal_ai/fal-ai/ben/v2/image",     "0.635",   0.635 * 0.025),
        # nano-banana-2 with header (Fal emits one when billed): $0.08/image
        ("fal_ai/fal-ai/nano-banana-2",    "1.0",     1.0 * 0.08),
    ],
)
def test_header_drives_cost_when_present(model, header_value, expected_cost):
    response = _response_with_header(header_value)
    cost = cost_calculator(model=model, image_response=response)
    assert cost == pytest.approx(expected_cost, rel=1e-9)


def test_header_takes_precedence_over_image_count():
    """Even for an "images"-billed model, header wins over n × unit_price."""
    # Use a value that's distinctly different from n × unit_price so we know
    # which path was taken.
    response = ImageResponse(
        data=[
            ImageObject(url="https://example.com/a.png", b64_json=None),
            ImageObject(url="https://example.com/b.png", b64_json=None),
        ]
    )
    response._hidden_params = {"fal_billable_units": 5.0}  # "5 units"
    # nano-banana-2 unit_price = $0.08
    cost = cost_calculator("fal_ai/fal-ai/nano-banana-2", response)
    assert cost == pytest.approx(5.0 * 0.08)  # header * price, not 2 * price


def test_malformed_header_falls_through():
    """Garbage header should not crash; falls through to the per-kind fallback."""
    response = ImageResponse(
        data=[ImageObject(url="https://example.com/x.png", b64_json=None)]
    )
    response._hidden_params = {"fal_billable_units": "not-a-number"}
    cost = cost_calculator("fal_ai/fal-ai/recraft/upscale/crisp", response)
    # Should silently fall through and use per-image: 1 * $0.004
    assert cost == pytest.approx(0.004)


# ------------------------------------------------------- fixed-per-image path

@pytest.mark.parametrize(
    "model,n_images,expected_cost",
    [
        ("fal_ai/fal-ai/recraft/upscale/crisp",    1, 0.004),
        ("fal_ai/fal-ai/recraft/upscale/crisp",    3, 0.012),
        ("fal_ai/fal-ai/recraft/upscale/creative", 1, 0.25),
        ("fal_ai/fal-ai/nano-banana-pro",          1, 0.15),
        ("fal_ai/fal-ai/nano-banana-2",            1, 0.08),
        ("fal_ai/fal-ai/nano-banana-2",            2, 0.16),
    ],
)
def test_fixed_per_image_when_header_missing(model, n_images, expected_cost):
    response = _response_without_header(num_images=n_images)
    cost = cost_calculator(model=model, image_response=response)
    assert cost == pytest.approx(expected_cost, rel=1e-9)


# ------------------------------------------------------ compute-second path

@pytest.mark.parametrize(
    "model,expected_flat",
    [
        ("fal_ai/fal-ai/esrgan",      0.008),
        ("fal_ai/fal-ai/aura-sr",     0.015),
        ("fal_ai/fal-ai/birefnet/v2", 0.008),
    ],
)
def test_compute_second_flat_when_header_missing(model, expected_flat):
    response = _response_without_header(num_images=1)
    cost = cost_calculator(model=model, image_response=response)
    assert cost == pytest.approx(expected_flat)


def test_compute_second_marks_needs_reconcile():
    """Compute-second fallback must flag the row so the audit skill / future
    reconciler can find it."""
    response = _response_without_header(num_images=1)
    cost_calculator("fal_ai/fal-ai/esrgan", response)
    assert response._hidden_params.get("needs_reconcile") is True


def test_header_path_does_not_mark_needs_reconcile():
    """When the header drives cost, no reconciliation is needed."""
    response = _response_with_header("0.123")
    cost_calculator("fal_ai/openai/gpt-image-2", response)
    assert "needs_reconcile" not in (response._hidden_params or {})


# ------------------------------------------------------ unknown model path

def test_unknown_fal_model_returns_zero():
    """Calculator should not crash on a Fal endpoint we haven't priced yet."""
    response = _response_without_header(num_images=1)
    cost = cost_calculator(
        model="fal_ai/fal-ai/some-future-model-not-yet-priced",
        image_response=response,
    )
    assert cost == 0.0


# -------------------------------------------------- invariants on constants

def test_every_priced_endpoint_has_a_unit_kind():
    """Sanity: every entry in FAL_UNIT_PRICES has a recognised unit_kind."""
    valid_kinds = {"units", "images", "megapixels", "compute_seconds"}
    for endpoint, (price, kind) in FAL_UNIT_PRICES.items():
        assert price > 0, f"{endpoint}: unit_price must be positive"
        assert kind in valid_kinds, f"{endpoint}: unknown unit_kind {kind!r}"


def test_compute_second_endpoints_have_fallback():
    """Every compute-seconds endpoint in FAL_UNIT_PRICES must also have a
    fallback flat estimate, otherwise no-header calls will undercharge by 0."""
    for endpoint, (_price, kind) in FAL_UNIT_PRICES.items():
        if kind == "compute_seconds":
            assert endpoint in COMPUTE_SECOND_FALLBACK, (
                f"{endpoint}: missing entry in COMPUTE_SECOND_FALLBACK"
            )
