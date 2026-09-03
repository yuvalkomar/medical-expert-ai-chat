from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class ChatTurn:
    role: Literal["user", "assistant"]
    content: str


class ProviderError(Exception):
    """An LLM failure with an explicit indication of whether retrying can help."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, system_prompt: str, messages: list[ChatTurn]) -> str:
        """Generate one assistant response for the supplied chronological messages."""

