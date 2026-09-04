import asyncio
from collections.abc import AsyncIterator
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from backend.app.llm.base import ChatTurn, LLMProvider, ProviderError


class BedrockLLMProvider(LLMProvider):
    """Model-agnostic implementation through AWS Bedrock Runtime Converse."""

    _NON_RETRYABLE_CODES = {
        "AccessDeniedException",
        "ResourceNotFoundException",
        "ValidationException",
    }
    _STREAM_ERROR_EVENTS = {
        "internalServerException": True,
        "modelStreamErrorException": True,
        "validationException": False,
        "throttlingException": True,
        "serviceUnavailableException": True,
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
        return await asyncio.to_thread(self._converse, system_prompt, messages)

    async def stream(
        self, system_prompt: str, messages: list[ChatTurn]
    ) -> AsyncIterator[str]:
        """Bridge Bedrock's blocking event stream into the application's async loop."""
        loop = asyncio.get_running_loop()
        chunks: asyncio.Queue[str | Exception | None] = asyncio.Queue()

        def produce() -> None:
            try:
                response = self._client.converse_stream(
                    **self._request(system_prompt, messages)
                )
                for event in response["stream"]:
                    stream_error = self._stream_event_error(event)
                    if stream_error is not None:
                        loop.call_soon_threadsafe(chunks.put_nowait, stream_error)
                        return
                    delta = event.get("contentBlockDelta", {}).get("delta", {})
                    text = delta.get("text")
                    if text:
                        loop.call_soon_threadsafe(chunks.put_nowait, text)
            except Exception as exc:
                loop.call_soon_threadsafe(chunks.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(chunks.put_nowait, None)

        producer = asyncio.create_task(asyncio.to_thread(produce))
        try:
            while True:
                item = await chunks.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise self._provider_error(item)
                yield item
            await producer
        finally:
            if not producer.done():
                producer.cancel()

    def _converse(self, system_prompt: str, messages: list[ChatTurn]) -> str:
        try:
            response = self._client.converse(**self._request(system_prompt, messages))
            text_parts = [
                item.get("text", "")
                for item in response["output"]["message"].get("content", [])
                if "text" in item
            ]
            answer = "".join(text_parts).strip()
            if not answer:
                raise ProviderError("Bedrock returned an empty response", retryable=True)
            return answer
        except Exception as exc:
            raise self._provider_error(exc) from exc

    def _stream_event_error(self, event: dict[str, Any]) -> ProviderError | None:
        for event_name, retryable in self._STREAM_ERROR_EVENTS.items():
            details = event.get(event_name)
            if details is None:
                continue
            message = details.get("message") or details.get("originalMessage")
            return ProviderError(
                f"Bedrock stream failed ({event_name}): "
                f"{message or 'Unknown stream error'}",
                retryable=retryable,
            )
        return None

    def _request(
        self, system_prompt: str, messages: list[ChatTurn]
    ) -> dict[str, Any]:
        return {
            "modelId": self.model_id,
            "system": [{"text": system_prompt}],
            "messages": [
                {"role": item.role, "content": [{"text": item.content}]}
                for item in messages
            ],
            "inferenceConfig": {
                "temperature": self.temperature,
                "maxTokens": self.max_tokens,
            },
        }

    def _provider_error(self, exc: Exception) -> ProviderError:
        if isinstance(exc, ProviderError):
            return exc
        if isinstance(exc, ClientError):
            error = exc.response.get("Error", {})
            code = str(error.get("Code", "ClientError"))
            message = str(error.get("Message", "AWS Bedrock request failed"))
            return ProviderError(
                f"Bedrock request failed ({code}): {message}",
                retryable=code not in self._NON_RETRYABLE_CODES,
            )
        if isinstance(exc, BotoCoreError):
            return ProviderError(f"Bedrock connection failed: {exc}", retryable=True)
        return ProviderError("Bedrock returned an invalid response", retryable=True)
