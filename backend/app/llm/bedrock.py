import asyncio
import json
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from backend.app.llm.base import ChatTurn, LLMProvider, ProviderError


class BedrockLLMProvider(LLMProvider):
    """Claude Messages API implementation through AWS Bedrock Runtime."""

    _NON_RETRYABLE_CODES = {
        "AccessDeniedException",
        "ResourceNotFoundException",
        "ValidationException",
    }

    def __init__(
        self,
        *,
        model_id: str,
        region_name: str,
        temperature: float,
        max_tokens: int,
        endpoint_url: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        client_options: dict[str, Any] = {"region_name": region_name}
        if endpoint_url:
            client_options["endpoint_url"] = endpoint_url
        self._client = boto3.client("bedrock-runtime", **client_options)

    async def generate(self, system_prompt: str, messages: list[ChatTurn]) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": system_prompt,
            "messages": [
                {"role": item.role, "content": [{"type": "text", "text": item.content}]}
                for item in messages
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        return await asyncio.to_thread(self._invoke, body)

    def _invoke(self, body: dict[str, Any]) -> str:
        try:
            response = self._client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            payload = json.loads(response["body"].read())
            text_parts = [
                item.get("text", "")
                for item in payload.get("content", [])
                if item.get("type") == "text"
            ]
            answer = "".join(text_parts).strip()
            if not answer:
                raise ProviderError("Bedrock returned an empty response", retryable=True)
            return answer
        except ClientError as exc:
            error = exc.response.get("Error", {})
            code = str(error.get("Code", "ClientError"))
            message = str(error.get("Message", "AWS Bedrock request failed"))
            raise ProviderError(
                f"Bedrock request failed ({code}): {message}",
                retryable=code not in self._NON_RETRYABLE_CODES,
            ) from exc
        except BotoCoreError as exc:
            raise ProviderError(f"Bedrock connection failed: {exc}", retryable=True) from exc
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProviderError("Bedrock returned an invalid response", retryable=True) from exc

