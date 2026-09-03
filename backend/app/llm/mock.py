import asyncio
from collections import defaultdict
from collections.abc import Callable

from backend.app.llm.base import ChatTurn, LLMProvider, ProviderError


class MockLLMProvider(LLMProvider):
    """Deterministic provider for local development and credential-free tests."""

    def __init__(
        self,
        *,
        response_delay: float = 0.1,
        failures_before_success: int = 0,
        always_fail_questions: set[str] | None = None,
        response_factory: Callable[[list[ChatTurn]], str] | None = None,
    ) -> None:
        self.response_delay = response_delay
        self.failures_before_success = failures_before_success
        self.always_fail_questions = always_fail_questions or set()
        self.response_factory = response_factory
        self.attempts: dict[str, int] = defaultdict(int)
        self.calls: list[list[ChatTurn]] = []
        self.active_calls = 0
        self.max_active_calls = 0
        self._state_lock = asyncio.Lock()

    async def generate(self, system_prompt: str, messages: list[ChatTurn]) -> str:
        del system_prompt  # The mock records messages but does not interpret instructions.
        question = messages[-1].content
        async with self._state_lock:
            self.attempts[question] += 1
            attempt = self.attempts[question]
            self.calls.append(list(messages))
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)

        try:
            if self.response_delay:
                await asyncio.sleep(self.response_delay)
            if question in self.always_fail_questions or attempt <= self.failures_before_success:
                raise ProviderError("Simulated transient LLM failure", retryable=True)
            if self.response_factory is not None:
                return self.response_factory(messages)
            return (
                "Mock medical information response: "
                f"{question} Please consult a licensed healthcare professional for personal advice."
            )
        finally:
            async with self._state_lock:
                self.active_calls -= 1

