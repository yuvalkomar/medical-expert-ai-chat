import asyncio

from backend.app.llm.base import ChatTurn
from backend.app.llm.bedrock import BedrockLLMProvider


class FakeBedrockRuntimeClient:
    def __init__(self) -> None:
        self.request: dict[str, object] | None = None

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
