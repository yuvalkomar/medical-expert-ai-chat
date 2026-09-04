import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class StreamEvent:
    event: str
    data: dict[str, Any]


@dataclass(slots=True)
class _StreamState:
    chunks: list[str] = field(default_factory=list)
    subscribers: set[asyncio.Queue[StreamEvent]] = field(default_factory=set)
    terminal: StreamEvent | None = None


class StreamRegistry:
    """In-memory fan-out for live chunks; final truth remains persisted in SQLite."""

    def __init__(self, *, max_retained: int = 1000) -> None:
        self._states: dict[str, _StreamState] = {}
        self.max_retained = max_retained

    def create(self, message_id: str) -> None:
        self._evict_old_terminal_states()
        self._states.setdefault(message_id, _StreamState())

    def publish(self, message_id: str, text: str) -> None:
        if not text:
            return
        state = self._states.setdefault(message_id, _StreamState())
        state.chunks.append(text)
        self._broadcast(state, StreamEvent("chunk", {"text": text}))

    def reset(self, message_id: str) -> None:
        state = self._states.setdefault(message_id, _StreamState())
        state.chunks.clear()
        self._broadcast(state, StreamEvent("reset", {"reason": "retry"}))

    def complete(self, message_id: str, answer: str) -> None:
        state = self._states.setdefault(message_id, _StreamState())
        event = StreamEvent("completed", {"status": "completed", "answer": answer})
        state.terminal = event
        self._broadcast(state, event)

    def fail(self, message_id: str, error: str) -> None:
        state = self._states.setdefault(message_id, _StreamState())
        event = StreamEvent("failed", {"status": "failed", "error": error})
        state.terminal = event
        self._broadcast(state, event)

    async def subscribe(self, message_id: str) -> AsyncIterator[StreamEvent]:
        state = self._states.setdefault(message_id, _StreamState())
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        state.subscribers.add(queue)
        buffered_chunks = list(state.chunks)
        terminal = state.terminal
        try:
            for chunk in buffered_chunks:
                yield StreamEvent("chunk", {"text": chunk})
            if terminal is not None:
                yield terminal
                return

            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield StreamEvent("ping", {})
                    continue
                yield event
                if event.event in {"completed", "failed"}:
                    return
        finally:
            state.subscribers.discard(queue)

    @staticmethod
    def _broadcast(state: _StreamState, event: StreamEvent) -> None:
        for subscriber in tuple(state.subscribers):
            subscriber.put_nowait(event)

    def _evict_old_terminal_states(self) -> None:
        excess = len(self._states) - self.max_retained + 1
        if excess <= 0:
            return
        terminal_ids = [
            message_id
            for message_id, state in self._states.items()
            if state.terminal is not None and not state.subscribers
        ]
        for message_id in terminal_ids[:excess]:
            self._states.pop(message_id, None)
