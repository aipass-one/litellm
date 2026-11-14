import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import assert_never

from litellm.types.decisions import (
    ChoiceQuestion,
    DecisionQuestion,
    DecisionsJSON,
    DecisionsRequest,
    NoulQuestion,
    ScoreQuestion,
)


def _text(value: DecisionsJSON) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _question(question: DecisionQuestion) -> Mapping[str, object]:
    match question:
        case NoulQuestion():
            return {
                "type": "noul",
                "instructions": _text(question.instructions)
                if question.instructions is not None
                else "Does the input match the true rather than false criteria?",
                **(
                    {
                        "criteria": {
                            f"{key}_description": _text(value) if value is not None else ""
                            for key, value in question.criteria.items()
                        }
                    }
                    if question.criteria is not None
                    else {}
                ),
            }
        case ChoiceQuestion():
            return {
                "type": "choice",
                "instructions": _text(question.instructions)
                if question.instructions is not None
                else "Choose the option that best matches the input.",
                "criteria": {
                    key: _text(value) if value is not None else "" for key, value in question.criteria.items()
                },
            }
        case ScoreQuestion():
            return {
                "type": "score",
                "instructions": _text(question.instructions)
                if question.instructions is not None
                else "Rate the input against the ordered criteria.",
                "criteria": [_text(value) for value in question.criteria],
            }
        case _:
            assert_never(question)


class _GeminiUsage(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(default=0, ge=0, le=0)


class _GeminiResponse(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
    answers: Mapping[str, object]
    usage: _GeminiUsage


@dataclass(frozen=True, slots=True)
class GeminiDecisionsEndpoint:
    api_key_env: tuple[str, ...] = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    api_base_env: str = "GEMINI_API_BASE"
    api_key_required: bool = True

    def default_api_base(self) -> str:
        return "https://generativelanguage.googleapis.com"

    def missing_api_base_message(self, provider: str) -> str:
        return f"api_base is required for Decisions provider '{provider}'"

    def canonical_model(self, model: str) -> str:
        return model

    def request_model(self, model: str) -> str:
        return model

    def endpoint_url(self, api_base: str, model: str) -> str:
        return f"{api_base.rstrip('/').removesuffix('/v1beta')}/v1beta/decisions"

    def prepare_request(
        self,
        model: str,
        request: DecisionsRequest,
        headers: Mapping[str, str],
        api_key: str | None,
    ) -> tuple[Mapping[str, str], Mapping[str, object]]:
        return (
            {
                **{
                    key: value
                    for key, value in headers.items()
                    if key.lower() not in {"authorization", "x-goog-api-key"}
                },
                **({"x-goog-api-key": api_key} if api_key is not None else {}),
            },
            {
                "model": model,
                "input": _text(request.state),
                "questions": {name: _question(question) for name, question in request.questions.items()},
            },
        )

    def unwrap_response(self, payload: object) -> object:
        response: Final = _GeminiResponse.model_validate(payload)
        return {
            **response.model_dump(exclude={"usage"}),
            "usage": {
                **response.usage.model_dump(),
                "input_tokens": response.usage.total_input_tokens,
                "output_tokens": response.usage.total_output_tokens,
            },
        }


GEMINI_DECISIONS_ENDPOINT: Final[GeminiDecisionsEndpoint] = GeminiDecisionsEndpoint()
