from io import BytesIO
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.fal_ai.image_edit.handler import (
    FalAIImageEdit,
    FalAIImageEditError,
)


def _mock_response(status_code: int, json_payload):
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = json_payload
    response.text = ""
    return response


def _sync_client():
    """A MagicMock that ``isinstance(_, HTTPHandler)`` returns True for, so the
    handler uses it instead of falling through to the real httpx client."""
    return MagicMock(spec=HTTPHandler)


def _patched_handler():
    handler = FalAIImageEdit()
    handler.config = MagicMock()
    handler.config.EDIT_ENDPOINT = "openai/gpt-image-2/edit"
    handler.config._PRESET_TO_SIZE = {"square_hd": (1024, 1024)}
    handler.config.validate_environment.return_value = {"Authorization": "Key test"}
    handler.config.transform_image_edit_request.return_value = (
        {
            "image_urls": ["data:image/png;base64,Zm9v"],
            "prompt": "edit me",
            "quality": "low",
            "image_size": "square_hd",
            "num_images": 1,
        },
        [],
    )
    image_response = MagicMock()
    image_response.size = None
    image_response.quality = None
    handler.config.transform_image_edit_response.return_value = image_response
    return handler


def test_image_edit_submits_polls_and_fetches_result(monkeypatch):
    handler = _patched_handler()

    submit_resp = _mock_response(
        200,
        {
            "request_id": "abc-123",
            "status_url": "https://queue.fal.run/openai/gpt-image-2/edit/requests/abc-123/status",
            "response_url": "https://queue.fal.run/openai/gpt-image-2/edit/requests/abc-123",
        },
    )
    in_progress = _mock_response(200, {"status": "IN_PROGRESS"})
    completed = _mock_response(200, {"status": "COMPLETED"})
    result_resp = _mock_response(
        200,
        {
            "images": [{"url": "https://v3b.fal.media/x.png", "width": 1024, "height": 1024}],
            "metrics": {"inference_time": 12.5},
        },
    )

    sync_client = _sync_client()
    sync_client.post.return_value = submit_resp
    sync_client.get.side_effect = [in_progress, completed, result_resp]

    monkeypatch.setattr("time.sleep", lambda *_args, **_kw: None)

    response = handler.image_edit(
        model="fal_ai/openai/gpt-image-2/edit",
        image=BytesIO(b"\x89PNG\r\n\x1a\n"),
        prompt="edit me",
        image_edit_optional_request_params={"quality": "low"},
        litellm_params={},
        logging_obj=MagicMock(),
        timeout=30,
        client=sync_client,
        aimage_edit=False,
    )

    # Submit POST went to the queue base, not fal.run
    submit_call = sync_client.post.call_args
    assert "queue.fal.run" in submit_call.kwargs["url"]
    assert submit_call.kwargs["url"].endswith("openai/gpt-image-2/edit")

    # Polled status, then fetched result url — last GET is response_url
    assert sync_client.get.call_count == 3
    assert sync_client.get.call_args_list[-1].kwargs["url"] == (
        "https://queue.fal.run/openai/gpt-image-2/edit/requests/abc-123"
    )

    # Stamped quality + size from request body, not response
    assert response.quality == "low"
    assert response.size == "1024-x-1024"


def test_image_edit_raises_on_failed_status(monkeypatch):
    handler = _patched_handler()

    submit_resp = _mock_response(
        200,
        {
            "request_id": "abc-123",
            "status_url": "https://queue.fal.run/x/status",
            "response_url": "https://queue.fal.run/x",
        },
    )
    failed = _mock_response(200, {"status": "FAILED", "logs": ["broken"]})

    sync_client = _sync_client()
    sync_client.post.return_value = submit_resp
    sync_client.get.return_value = failed

    monkeypatch.setattr("time.sleep", lambda *_a, **_kw: None)

    with pytest.raises(FalAIImageEditError) as exc_info:
        handler.image_edit(
            model="fal_ai/openai/gpt-image-2/edit",
            image=BytesIO(b"x"),
            prompt="edit me",
            image_edit_optional_request_params={},
            litellm_params={},
            logging_obj=MagicMock(),
            timeout=30,
            client=sync_client,
            aimage_edit=False,
        )

    assert "FAILED" in exc_info.value.message


def test_image_edit_rejects_submit_error(monkeypatch):
    handler = _patched_handler()

    submit_resp = _mock_response(403, {"detail": "auth fail"})
    sync_client = _sync_client()
    sync_client.post.return_value = submit_resp

    with pytest.raises(FalAIImageEditError) as exc_info:
        handler.image_edit(
            model="fal_ai/openai/gpt-image-2/edit",
            image=BytesIO(b"x"),
            prompt="edit me",
            image_edit_optional_request_params={},
            litellm_params={},
            logging_obj=MagicMock(),
            timeout=30,
            client=sync_client,
            aimage_edit=False,
        )

    assert exc_info.value.status_code == 403
