from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Iterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Final

import pytest
import respx
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.routing import Match

import litellm
from litellm.proxy._lazy_features import LAZY_FEATURES, LazyFeature, LazyFeatureMiddleware, _force_load
from litellm.proxy.decisions_endpoints.endpoints import decisions
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import SafeRouteAdder
from litellm.proxy.proxy_server import (
    app,
    cleanup_router_config_variables,
    initialize,
)

_INPUT_TOKENS: Final[int] = 367
_OUTPUT_TOKENS: Final[int] = 3
_RESPONSE: Final[Mapping[str, object]] = {
    "model": "pplx-decider-v1-27b",
    "answers": {
        "is_defect": {"type": "noul", "noul": 0.9},
    },
    "usage": {"input_tokens": _INPUT_TOKENS, "output_tokens": _OUTPUT_TOKENS},
}
_STRANDS_RESPONSE: Final[Mapping[str, object]] = {
    "model": "strands-decider-2B-hobson-v19",
    "answers": {
        "is_defect": {"type": "noul", "noul": 0.9},
    },
    "usage": {"input_tokens": 216, "output_tokens": 3},
    "latency_ms": 3722.17,
}
_REQUEST: Final[Mapping[str, object]] = {
    "model": "decider",
    "state": {"source": "proxy-test"},
    "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    monkeypatch.setenv("OPENAI_API_BASE", "https://fake-openai.example")
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    cleanup_router_config_variables()
    config_path: Final = Path(__file__).parents[3] / "test_litellm" / "proxy" / "test_configs" / "test_config_no_auth.yaml"
    asyncio.run(initialize(config=str(config_path), debug=True))
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "decider",
                    "litellm_params": {
                        "model": "perplexity/pplx-decider-v1-27b",
                        "api_key": "test-key",
                    },
                }
            ]
        ),
    )
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield TestClient(app)
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.mark.parametrize("endpoint", ("/v1/decisions", "/decisions"))
def test_proxy_decisions_route_returns_answers_and_cost(
    client: TestClient,
    respx_mock: respx.MockRouter,
    endpoint: str,
) -> None:
    upstream: Final = respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(json=_RESPONSE)

    response: Final = client.post(endpoint, json=_REQUEST)

    assert response.status_code == 200, response.text
    assert response.json()["answers"] == _RESPONSE["answers"]
    assert "_hidden_params" not in response.json()
    perplexity_cost: Final = litellm.model_cost["perplexity/pplx-decider-v1-27b"]
    expected_cost: Final = _INPUT_TOKENS * float(perplexity_cost["input_cost_per_token"]) + _OUTPUT_TOKENS * float(
        perplexity_cost["output_cost_per_token"]
    )

    assert expected_cost > 0
    assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(expected_cost)
    assert upstream.called
    assert json.loads(upstream.calls[0].request.content) == {
        "model": "pplx-decider-v1-27b",
        "state": {"source": "proxy-test"},
        "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    }
    assert upstream.calls[0].request.headers["authorization"] == "Bearer test-key"


def test_proxy_decisions_dispatches_typesafe_deployment(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "jev",
                "litellm_params": {
                    "model": "typesafe/jev-latest",
                    "api_key": "k",
                },
            }
        ]
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)
    upstream: Final = respx_mock.post("https://api.typesafe.ai/v1/systemone").respond(json=_RESPONSE)

    response: Final = client.post(
        "/v1/decisions",
        json={
            "model": "jev",
            "state": {"source": "proxy-test"},
            "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["answers"] == _RESPONSE["answers"]
    assert upstream.called
    assert json.loads(upstream.calls[0].request.content) == {
        "model": "jev-latest",
        "state": {"source": "proxy-test"},
        "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    }


def test_proxy_decisions_sends_the_env_key_to_the_deployment_api_base(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setenv("PERPLEXITYAI_API_KEY", "server-key")
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "decider",
                "litellm_params": {
                    "model": "perplexity/pplx-decider-v1-27b",
                    "api_base": "https://egress.example/perplexity",
                },
            }
        ]
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)
    upstream: Final = respx_mock.post("https://egress.example/perplexity/v1/decisions").respond(json=_RESPONSE)

    response: Final = client.post("/v1/decisions", json=_REQUEST)

    assert response.status_code == 200, response.text
    assert upstream.call_count == 1
    assert upstream.calls[0].request.headers["authorization"] == "Bearer server-key"


def test_proxy_decisions_unknown_model_is_a_client_error(
    client: TestClient,
    respx_mock: respx.MockRouter,
) -> None:
    response: Final = client.post(
        "/v1/decisions",
        json={
            "model": "missing-model",
            "state": "review",
            "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
        },
    )

    assert 400 <= response.status_code < 500, response.text
    assert len(respx_mock.calls) == 0


@pytest.mark.parametrize(
    "request_body",
    (
        {
            "model": "decider",
            "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
        },
        {
            "model": "decider",
            "state": {"source": "proxy-test"},
        },
    ),
    ids=("missing_state", "missing_questions"),
)
def test_proxy_decisions_missing_required_field_is_a_client_error(
    client: TestClient,
    respx_mock: respx.MockRouter,
    request_body: Mapping[str, object],
) -> None:
    upstream: Final = respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(json=_RESPONSE)

    response: Final = client.post("/v1/decisions", json=request_body)

    assert response.status_code == 400, response.text
    assert not upstream.called


def test_proxy_decisions_dispatches_strands_decider(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.delenv("STRANDS_DECIDER_API_BASE", raising=False)
    monkeypatch.delenv("STRANDS_DECIDER_API_KEY", raising=False)
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "strands",
                "litellm_params": {
                    "model": "strands_decider/strands-decider-2B-hobson-v19",
                    "api_base": "https://strands.example",
                },
            }
        ]
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)
    upstream: Final = respx_mock.post("https://strands.example/v1/systemone").respond(json=_STRANDS_RESPONSE)

    response: Final = client.post(
        "/v1/decisions",
        json={
            "model": "strands",
            "state": {"source": "proxy-test"},
            "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["answers"] == _STRANDS_RESPONSE["answers"]
    assert upstream.called
    assert json.loads(upstream.calls[0].request.content) == {
        "model": "strands-decider-2B-hobson-v19",
        "state": {"source": "proxy-test"},
        "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    }
    assert "authorization" not in upstream.calls[0].request.headers


def test_proxy_decisions_without_model_uses_the_proxy_default_model(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setattr(litellm.proxy.proxy_server, "user_model", "decider")
    upstream: Final = respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(json=_RESPONSE)

    response: Final = client.post(
        "/v1/decisions", json={key: value for key, value in _REQUEST.items() if key != "model"}
    )

    assert response.status_code == 200, response.text
    assert response.json()["answers"] == _RESPONSE["answers"]
    assert upstream.called
    assert json.loads(upstream.calls[0].request.content)["model"] == "pplx-decider-v1-27b"


def _decisions_feature() -> LazyFeature:
    return next(feature for feature in LAZY_FEATURES if feature.name == "decisions")


def _serving_endpoint(bare: FastAPI, path: str) -> object:
    scope: Final = {"type": "http", "method": "POST", "path": path, "root_path": "", "query_string": b"", "headers": ()}
    return next(
        route.endpoint for route in bare.routes if isinstance(route, APIRoute) and route.matches(scope)[0] is Match.FULL
    )


def test_a_config_pass_through_at_v1_decisions_keeps_its_route_and_the_native_api_serves_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LITELLM_DISABLE_LAZY_ROUTES", raising=False)

    async def pass_through() -> dict[str, str]:
        return {"served_by": "pass-through"}

    bare: Final = FastAPI()
    bare.add_middleware(LazyFeatureMiddleware, fastapi_app=bare, features=(_decisions_feature(),))
    SafeRouteAdder.add_api_route_if_not_exists(bare, "/v1/decisions", pass_through, ["POST"])
    with TestClient(bare) as client:
        assert client.post("/v1/decisions", json={"model": "gpt-6-luna"}).json() == {"served_by": "pass-through"}
    assert _serving_endpoint(bare, "/v1/decisions") is pass_through
    assert _serving_endpoint(bare, "/decisions") is decisions


def test_an_eagerly_registered_decisions_route_still_yields_to_a_config_pass_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    async def pass_through() -> dict[str, str]:
        return {"served_by": "pass-through"}

    @asynccontextmanager
    async def loads_the_config(app_: FastAPI) -> AsyncGenerator[None]:
        assert SafeRouteAdder.add_api_route_if_not_exists(app_, "/v1/decisions", pass_through, ["POST"]), (
            "the native route registered at startup must not block the config pass-through"
        )
        yield

    bare: Final = FastAPI(lifespan=loads_the_config)
    asyncio.run(_force_load(bare, _decisions_feature(), (_decisions_feature(),)))
    bare.add_middleware(LazyFeatureMiddleware, fastapi_app=bare, features=(_decisions_feature(),))
    with TestClient(bare) as client:
        assert client.post("/v1/decisions", json={"model": "gpt-6-luna"}).json() == {"served_by": "pass-through"}
    assert _serving_endpoint(bare, "/v1/decisions") is pass_through
    assert _serving_endpoint(bare, "/decisions") is decisions


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ("/v1/decisions", "/decisions"))
@pytest.mark.parametrize(
    ("provider_model", "upstream_url", "wire_model"),
    (
        ("typesafe/jev-1.13", "https://api.typesafe.ai/v1/systemone", "jev-1.13"),
        ("openrouter/typesafe/jev-1.13", "https://openrouter.ai/api/alpha/decisions", "typesafe/jev-1.13"),
    ),
)
async def test_jev_virtual_key_cost_header_spendlogs_and_budget(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    endpoint: str,
    provider_model: str,
    upstream_url: str,
    wire_model: str,
) -> None:
    from unittest.mock import AsyncMock, MagicMock

    import httpx

    from litellm.caching.dual_cache import DualCache
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger
    from litellm.proxy.utils import ProxyLogging, ProxyUpdateSpend, hash_token, jsonify_object

    proxy: Final = litellm.proxy.proxy_server
    key: Final = "sk-decisions-billing-test"
    key_hash: Final = hash_token(key)
    key_cache: Final = UserApiKeyCache()
    key_cache.set_cache(
        key=key_hash,
        value=UserAPIKeyAuth(
            token=key_hash,
            api_key=key_hash,
            models=["jev"],
            allowed_routes=["llm_api_routes"],
            user_role=LitellmUserRoles.INTERNAL_USER,
            max_budget=0.000001,
            spend=0,
        ),
    )
    counter_cache: Final = DualCache()
    counter_cache.set_cache(key=f"spend:key:{key_hash}", value=0.0)
    logging: Final = ProxyLogging(user_api_key_cache=key_cache)
    prisma: Final = MagicMock()
    prisma.spend_log_transactions = []
    prisma._spend_log_transactions_lock = asyncio.Lock()
    prisma.jsonify_object = jsonify_object
    prisma.db.litellm_spendlogs.create_many = AsyncMock(return_value=1)
    prisma.db.litellm_verificationtoken.find_unique = AsyncMock(return_value=None)
    prisma.db.litellm_modelaccessgroupbudgettable.find_many = AsyncMock(return_value=[])
    monkeypatch.setattr(proxy, "master_key", "sk-test-master")
    monkeypatch.setattr(proxy, "user_api_key_cache", key_cache)
    monkeypatch.setattr(proxy, "spend_counter_cache", counter_cache)
    monkeypatch.setattr(proxy, "proxy_logging_obj", logging)
    monkeypatch.setattr(proxy, "prisma_client", prisma)
    monkeypatch.setattr(proxy, "disable_spend_logs", False)
    monkeypatch.setattr(proxy, "general_settings", {})
    monkeypatch.setattr(litellm, "callbacks", [_ProxyDBLogger()])
    monkeypatch.setattr(
        proxy,
        "llm_router",
        litellm.Router(
            model_list=[
                {"model_name": "jev", "litellm_params": {"model": provider_model, "api_key": "provider-test-key"}}
            ]
        ),
    )
    upstream: Final = respx_mock.post(upstream_url).respond(json={**_RESPONSE, "model": wire_model})
    body: Final = {**_REQUEST, "model": "jev"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http_client:
        unauthenticated: Final = await http_client.post(endpoint, json=body)
        assert unauthenticated.status_code == 401
        assert not upstream.called
        response: Final = await http_client.post(endpoint, json=body, headers={"Authorization": f"Bearer {key}"})
        assert response.status_code == 200, response.text
        await asyncio.sleep(0)
        GLOBAL_LOGGING_WORKER.start()
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10)
        expected: Final = _INPUT_TOKENS * 4.2e-08
        assert litellm.model_cost[provider_model]["input_cost_per_token"] == 4.2e-08
        assert litellm.model_cost[provider_model]["output_cost_per_token"] == 0
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(expected)
        assert json.loads(upstream.calls[0].request.content)["model"] == wire_model
        successes: Final = tuple(row for row in prisma.spend_log_transactions if row["status"] == "success")
        assert len(successes) == 1
        row: Final = successes[0]
        assert row["spend"] == pytest.approx(expected)
        assert row["spend"] > 0
        assert row["prompt_tokens"] == _INPUT_TOKENS
        assert row["completion_tokens"] == _OUTPUT_TOKENS
        assert row["api_key"] == key_hash
        assert row["call_type"] == "adecisions"
        assert row["status"] == "success"
        assert await counter_cache.async_get_cache(f"spend:key:{key_hash}") == pytest.approx(expected)
        await ProxyUpdateSpend.update_spend_logs(
            n_retry_times=0, prisma_client=prisma, db_writer_client=None, proxy_logging_obj=logging
        )
        prisma.db.litellm_spendlogs.create_many.assert_awaited_once()
        inserted: Final = tuple(
            row for row in prisma.db.litellm_spendlogs.create_many.call_args.kwargs["data"] if row["status"] == "success"
        )
        assert len(inserted) == 1
        assert inserted[0]["spend"] == pytest.approx(expected)
        assert prisma.spend_log_transactions == []
        over_budget: Final = await http_client.post(endpoint, json=body, headers={"Authorization": f"Bearer {key}"})
        assert over_budget.status_code == 429, over_budget.text
        assert over_budget.json()["error"]["type"] == "budget_exceeded"
        assert "budget" in over_budget.text.lower()
        assert upstream.call_count == 1
