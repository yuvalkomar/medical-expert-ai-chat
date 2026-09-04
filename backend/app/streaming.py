"""In-process Server-Sent Events fan-out for responses produced by background workers.

Workers publish text chunks under a message ID while HTTP clients subscribe to the same ID. Chunks
are retained in bounded memory so a client that connects after processing starts can catch up.
Completed answers and failures are also persisted by the worker; this registry exists only for
live delivery and is not a second database.
"""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """One named SSE event and its JSON-serializable payload."""

    event: str
    data: dict[str, Any]


@dataclass(slots=True)
class _StreamState:
    """Buffered output and active subscribers belonging to one message."""

    chunks: list[str] = field(default_factory=list)
    subscribers: set[asyncio.Queue[StreamEvent]] = field(default_factory=set)
    terminal: StreamEvent | None = None


class StreamRegistry:
    """Replay and fan out live events to any number of subscribers.

    Mutations occur on the application's asyncio event loop, so the registry does not need locks.
    Subscriber queues decouple the worker from network speed: publishing a chunk never waits for a
    client to send it. Terminal states are retained up to ``max_retained`` so late subscribers can
    still receive the completed or failed event.
    """

    def __init__(self, *, max_retained: int = 1000) -> None:
        self._states: dict[str, _StreamState] = {}
        self.max_retained = max_retained

    def create(self, message_id: str) -> None:
        """Create state for a newly queued message without replacing existing buffered events."""
        self._evict_old_terminal_states()
        self._states.setdefault(message_id, _StreamState())

    def publish(self, message_id: str, text: str) -> None:
        """Buffer a non-empty text chunk and deliver it to current subscribers."""
        if not text:
            return
        state = self._states.setdefault(message_id, _StreamState())
        state.chunks.append(text)
        self._broadcast(state, StreamEvent("chunk", {"text": text}))

    def reset(self, message_id: str) -> None:
        """Discard partial chunks and tell clients that a provider retry is beginning."""
        state = self._states.setdefault(message_id, _StreamState())
        state.chunks.clear()
        self._broadcast(state, StreamEvent("reset", {"reason": "retry"}))

    def complete(self, message_id: str, answer: str) -> None:
        """Publish and retain successful termination with the full persisted answer."""
        state = self._states.setdefault(message_id, _StreamState())
        event = StreamEvent("completed", {"status": "completed", "answer": answer})
        state.terminal = event
        self._broadcast(state, event)

    def fail(self, message_id: str, error: str) -> None:
        """Publish and retain terminal failure information."""
        state = self._states.setdefault(message_id, _StreamState())
        event = StreamEvent("failed", {"status": "failed", "error": error})
        state.terminal = event
        self._broadcast(state, event)

    async def subscribe(self, message_id: str) -> AsyncIterator[StreamEvent]:
        """Yield replayed and live events until completion, failure, or cancellation.

        A heartbeat is emitted during quiet periods to keep proxies and clients from treating the
        connection as idle. The subscriber queue is always detached in ``finally``, including when
        an HTTP client disconnects.
        """
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
        """Place an event on every subscriber queue without blocking the producing worker."""
        for subscriber in tuple(state.subscribers):
            subscriber.put_nowait(event)

    def _evict_old_terminal_states(self) -> None:
        """Bound memory use by evicting the oldest inactive terminal states."""
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
