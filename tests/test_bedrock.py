import asyncio

import pytest

from backend.app.llm.base import ChatTurn, ProviderError
from backend.app.llm.bedrock import BedrockLLMProvider


class FakeBedrockRuntimeClient:
    def __init__(self, stream_events: list[dict[str, object]] | None = None) -> None:
        self.request: dict[str, object] | None = None
        self.stream_events = stream_events

    def converse(self, **kwargs: object) -> dict[str, object]:
        self.request = kwargs
        return {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "A model-independent response"}],
                }
            }
        }

    def converse_stream(self, **kwargs: object) -> dict[str, object]:
        self.request = kwargs
        events = (
            self.stream_events
            if self.stream_events is not None
            else [
                {"messageStart": {"role": "assistant"}},
                {"contentBlockDelta": {"delta": {"text": "Streamed "}}},
                {"contentBlockDelta": {"delta": {"text": "answer"}}},
                {"messageStop": {"stopReason": "end_turn"}},
            ]
        )
        return {
            "stream": iter(events)
        }


def test_bedrock_provider_uses_model_agnostic_converse(monkeypatch):
    fake_client = FakeBedrockRuntimeClient()
    monkeypatch.setattr(
        "backend.app.llm.bedrock.boto3.client",
        lambda *args, **kwargs: fake_client,
    )
    provider = BedrockLLMProvider(
        model_id="eu.amazon.nova-pro-v1:0",
        region_name="eu-north-1",
        temperature=0.2,
        max_tokens=1000,
    )

    answer = asyncio.run(
        provider.generate(
            "Medical system prompt",
            [
                ChatTurn(role="user", content="First question"),
                ChatTurn(role="assistant", content="First answer"),
                ChatTurn(role="user", content="Follow-up"),
            ],
        )
    )

    assert answer == "A model-independent response"
    assert fake_client.request == {
        "modelId": "eu.amazon.nova-pro-v1:0",
        "system": [{"text": "Medical system prompt"}],
        "messages": [
            {"role": "user", "content": [{"text": "First question"}]},
            {"role": "assistant", "content": [{"text": "First answer"}]},
            {"role": "user", "content": [{"text": "Follow-up"}]},
        ],
        "inferenceConfig": {"temperature": 0.2, "maxTokens": 1000},
    }


def test_bedrock_provider_bridges_converse_stream(monkeypatch):
    fake_client = FakeBedrockRuntimeClient()
    monkeypatch.setattr(
        "backend.app.llm.bedrock.boto3.client",
        lambda *args, **kwargs: fake_client,
    )
    provider = BedrockLLMProvider(
        model_id="eu.amazon.nova-pro-v1:0",
        region_name="eu-north-1",
        temperature=0.2,
        max_tokens=1000,
    )

    async def collect_chunks() -> list[str]:
        return [
            chunk
            async for chunk in provider.stream(
                "Medical system prompt",
                [ChatTurn(role="user", content="Question")],
            )
        ]

    assert asyncio.run(collect_chunks()) == ["Streamed ", "answer"]
    assert fake_client.request == {
        "modelId": "eu.amazon.nova-pro-v1:0",
        "system": [{"text": "Medical system prompt"}],
        "messages": [{"role": "user", "content": [{"text": "Question"}]}],
        "inferenceConfig": {"temperature": 0.2, "maxTokens": 1000},
    }


@pytest.mark.parametrize(
    ("event_name", "retryable"),
    [
        ("internalServerException", True),
        ("modelStreamErrorException", True),
        ("validationException", False),
        ("throttlingException", True),
        ("serviceUnavailableException", True),
    ],
)
def test_bedrock_provider_raises_for_stream_error_events(
    monkeypatch, event_name: str, retryable: bool
):
    fake_client = FakeBedrockRuntimeClient(
        [
            {"contentBlockDelta": {"delta": {"text": "Partial answer"}}},
            {event_name: {"message": "Simulated stream failure"}},
        ]
    )
    monkeypatch.setattr(
        "backend.app.llm.bedrock.boto3.client",
        lambda *args, **kwargs: fake_client,
    )
    provider = BedrockLLMProvider(
        model_id="eu.amazon.nova-pro-v1:0",
        region_name="eu-north-1",
        temperature=0.2,
        max_tokens=1000,
    )
    received: list[str] = []

    async def collect_chunks() -> None:
        async for chunk in provider.stream(
            "Medical system prompt",
            [ChatTurn(role="user", content="Question")],
        ):
            received.append(chunk)

    with pytest.raises(ProviderError, match=event_name) as error:
        asyncio.run(collect_chunks())

    assert received == ["Partial answer"]
    assert error.value.retryable is retryable
