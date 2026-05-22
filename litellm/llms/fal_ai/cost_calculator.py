"""
fal.ai cost calculator.

Source of truth: the ``x-fal-billable-units`` HTTP response header that Fal
emits on every modern endpoint. The transformation layer captures it onto
``image_response._hidden_params["fal_billable_units"]`` (see
``FalAIBaseConfig.transform_image_generation_response``,
``FalAIImageEditConfig.transform_image_edit_response``, and
``FalAIBaseVideoConfig._capture_billable_units``); this calculator reads it
back and multiplies by the static unit price for the endpoint.

Three paths, in priority order:

1. **Header present** — exact cost. Used by the 8+ modern Fal endpoints we
   care about (gpt-image-2, gpt-image-2/edit, clarity-upscaler, topaz,
   ben/v2, nano-banana family, seedance, …). Multiplies
   ``fal_billable_units`` by the model's static ``unit_price``.

2. **Fixed per-image** — the four ``"images"``-billed endpoints
   (recraft/upscale/{crisp,creative}, nano-banana families when the header
   is absent). Returns ``num_images_returned × unit_price``.

3. **Compute-seconds flat estimate** — the three legacy compute-priced
   endpoints (esrgan, aura-sr, birefnet/v2) that don't emit the header.
   Returns a per-model flat estimate calibrated against historical Fal
   billing. Real drift surfaces via the ``fal-cost-audit`` operator skill.

The previous matrix-based implementation (composite keys like
``high/fal_ai/1024-x-1024/openai/gpt-image-2``) is gone — the matrix was
brittle, never matched Fal's actual billing for variable-quality models,
and is unnecessary now that we read Fal's authoritative header.
"""

from typing import Any

from litellm.types.utils import ImageResponse
from litellm.types.videos.main import VideoObject


# ---------------------------------------------------------------------------
# Pricing snapshot — refresh when Fal changes published rates. Source:
# GET https://api.fal.ai/v1/models/pricing?endpoint_id=...
# ---------------------------------------------------------------------------
# (unit_price_usd, unit_kind)
FAL_UNIT_PRICES = {
    # variable per-token / per-unit (header-emitting)
    "openai/gpt-image-2":              (1.00,    "units"),
    "openai/gpt-image-2/edit":         (1.00,    "units"),
    # variable per-output-megapixel (header-emitting)
    "fal-ai/clarity-upscaler":         (0.03,    "megapixels"),
    "fal-ai/topaz/upscale/image":      (0.01,    "megapixels"),
    "fal-ai/ben/v2/image":             (0.025,   "megapixels"),
    # fixed per-output-image (header-emitting on newer endpoints,
    # falls back to per-image multiply when header absent)
    "fal-ai/nano-banana-pro":          (0.15,    "images"),
    "fal-ai/nano-banana-2":            (0.08,    "images"),
    "fal-ai/recraft/upscale/crisp":    (0.004,   "images"),
    "fal-ai/recraft/upscale/creative": (0.25,    "images"),
    # compute-second billed (no header on legacy endpoints)
    "fal-ai/esrgan":                   (0.00111, "compute_seconds"),
    "fal-ai/aura-sr":                  (0.00125, "compute_seconds"),
    "fal-ai/birefnet/v2":              (0.00111, "compute_seconds"),
    # video — billed per output second (header-emitting via PR #22 pattern)
    "bytedance/seedance-2.0/image-to-video":      (0.3024, "seconds"),
    "bytedance/seedance-2.0/fast/image-to-video": (0.2419, "seconds"),
}

# Per-model flat estimates for compute-second endpoints that don't emit the
# header. Calibrated against historical Fal billing — re-tune via
# `/fal-cost-audit` and update here if drift exceeds a few cents/day.
COMPUTE_SECOND_FALLBACK = {
    "fal-ai/esrgan":      0.008,   # ~7s × $0.00111
    "fal-ai/aura-sr":     0.015,   # ~12s × $0.00125
    "fal-ai/birefnet/v2": 0.008,   # ~7s × $0.00111
}

# Endpoints where we know the header isn't emitted. Used by
# cost_reconciler.py (when present) to decide which calls to reconcile.
NO_HEADER_COMPUTE_MODELS = frozenset(COMPUTE_SECOND_FALLBACK)


def _endpoint_id(model: str) -> str:
    """Strip the ``fal_ai/`` provider prefix LiteLLM adds to model names."""
    return model[len("fal_ai/"):] if model.startswith("fal_ai/") else model


def cost_calculator(model: str, image_response: Any) -> float:
    endpoint = _endpoint_id(model)
    unit_price_kind = FAL_UNIT_PRICES.get(endpoint)
    if unit_price_kind is None:
        # Unknown Fal endpoint. Fall back to LiteLLM's generic cost lookup
        # which will return 0.0 if no price is configured. Prefer this over
        # a hard error so a new Fal model can be added to model_list before
        # FAL_UNIT_PRICES is updated.
        return 0.0
    unit_price, unit_kind = unit_price_kind

    # Both ImageResponse (image gen / edit) and VideoObject (video) carry the
    # ``_hidden_params["fal_billable_units"]`` value captured from the
    # response header. Anything else is unexpected — return 0 defensively
    # rather than crashing the request.
    if not isinstance(image_response, (ImageResponse, VideoObject)):
        return 0.0

    hidden = getattr(image_response, "_hidden_params", None) or {}
    units = hidden.get("fal_billable_units")

    # Path 1 — exact via response header (the 8+ header-emitting models,
    # including seedance video models which bill per output second).
    if units is not None:
        try:
            return float(units) * unit_price
        except (TypeError, ValueError):
            pass  # malformed header, fall through to estimate

    # Path 2 — fixed per-image: just multiply by image count.
    if unit_kind == "images":
        data = getattr(image_response, "data", None)
        n = len(data) if data else 1
        return n * unit_price

    # Path 3 — endpoints without a usable header. Mark for reconciliation
    # (the audit skill or a future async reconciler can correct after the
    # fact). For per-second video the default fallback assumes ~5 output
    # seconds; the audit catches drift the next day.
    hidden["needs_reconcile"] = True
    image_response._hidden_params = hidden
    return COMPUTE_SECOND_FALLBACK.get(endpoint, unit_price * 5)
