import asyncio

from backend.app.streaming import StreamRegistry


def test_stream_registry_replays_chunks_and_resets_failed_attempt():
    async def scenario():
        registry = StreamRegistry()
        registry.create("message-id")
        registry.publish("message-id", "partial")

        async def consume():
            return [event async for event in registry.subscribe("message-id")]

        consumer = asyncio.create_task(consume())
        await asyncio.sleep(0)
        registry.reset("message-id")
        registry.publish("message-id", "final answer")
        registry.complete("message-id", "final answer")
        return await consumer

    events = asyncio.run(scenario())
    assert [(event.event, event.data) for event in events] == [
        ("chunk", {"text": "partial"}),
        ("reset", {"reason": "retry"}),
        ("chunk", {"text": "final answer"}),
        ("completed", {"status": "completed", "answer": "final answer"}),
    ]
