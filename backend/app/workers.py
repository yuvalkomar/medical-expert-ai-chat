import asyncio
import logging
from datetime import datetime, timezone

from sqlmodel import select

from backend.app.config import Settings
from backend.app.database import Database
from backend.app.llm.base import ChatTurn, LLMProvider, ProviderError
from backend.app.models import ChatMessage, MessageStatus, utc_now
from backend.app.prompts import MEDICAL_SYSTEM_PROMPT


def _elapsed_ms(started_at: datetime, completed_at: datetime) -> float:
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    if completed_at.tzinfo is None:
        completed_at = completed_at.replace(tzinfo=timezone.utc)
    return max(0.0, (completed_at - started_at).total_seconds() * 1000)


class WorkerPool:
    """In-process queue with a fixed number of independently sessioned workers."""

    def __init__(
        self,
        *,
        database: Database,
        provider: LLMProvider,
        settings: Settings,
        logger: logging.Logger,
    ) -> None:
        self.database = database
        self.provider = provider
        self.settings = settings
        self.logger = logger
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.tasks: list[asyncio.Task[None]] = []
        self._accepting = False

    async def start(self) -> None:
        self._accepting = True
        self.tasks = [
            asyncio.create_task(self._worker(index), name=f"chat-worker-{index}")
            for index in range(self.settings.max_concurrency)
        ]
        with self.database.session() as session:
            unfinished = session.exec(
                select(ChatMessage.id)
                .where(ChatMessage.status == MessageStatus.PROCESSING.value)
                .order_by(ChatMessage.created_at)
            ).all()
        for message_id in unfinished:
            await self.queue.put(message_id)
        self.logger.info("worker_pool_started", extra={"worker": len(self.tasks)})

    async def enqueue(self, message_id: str) -> None:
        if not self._accepting:
            raise RuntimeError("Worker pool is not accepting messages")
        await self.queue.put(message_id)

    async def stop(self) -> None:
        self._accepting = False
        try:
            await asyncio.wait_for(
                self.queue.join(), timeout=self.settings.shutdown_grace_period
            )
        except asyncio.TimeoutError:
            self.logger.warning("worker_pool_shutdown_timeout")
            for task in self.tasks:
                task.cancel()
            await asyncio.gather(*self.tasks, return_exceptions=True)
        else:
            for _ in self.tasks:
                await self.queue.put(None)
            await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()
        self.logger.info("worker_pool_stopped")

    async def _worker(self, index: int) -> None:
        while True:
            message_id = await self.queue.get()
            try:
                if message_id is None:
                    return
                await self._process_message(message_id, index)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.exception(
                    "unhandled_worker_error",
                    extra={"message_id": message_id, "worker": index},
                )
            finally:
                self.queue.task_done()

    def _load_request(self, message_id: str) -> tuple[ChatMessage, list[ChatTurn]] | None:
        with self.database.session() as session:
            message = session.get(ChatMessage, message_id)
            if message is None or message.status != MessageStatus.PROCESSING.value:
                return None
            if message.started_at is None:
                message.started_at = utc_now()
                session.add(message)
                session.commit()

            previous = session.exec(
                select(ChatMessage)
                .where(
                    ChatMessage.conversation_id == message.conversation_id,
                    ChatMessage.status == MessageStatus.COMPLETED.value,
                    ChatMessage.id != message.id,
                    ChatMessage.created_at <= message.created_at,
                )
                .order_by(ChatMessage.created_at, ChatMessage.id)
            ).all()
            turns: list[ChatTurn] = []
            for prior in previous:
                turns.append(ChatTurn(role="user", content=prior.question))
                if prior.answer:
                    turns.append(ChatTurn(role="assistant", content=prior.answer))
            turns.append(ChatTurn(role="user", content=message.question))
            return message, turns

    async def _process_message(self, message_id: str, worker_index: int) -> None:
        loaded = self._load_request(message_id)
        if loaded is None:
            return
        message, turns = loaded
        final_error: str | None = None

        for retry_number in range(self.settings.max_retries + 1):
            try:
                answer = await self.provider.generate(MEDICAL_SYSTEM_PROMPT, turns)
                self._save_success(message_id, answer)
                self.logger.info(
                    "message_completed",
                    extra={
                        "message_id": message_id,
                        "question": message.question,
                        "response": answer,
                        "retry_count": retry_number,
                        "worker": worker_index,
                    },
                )
                return
            except asyncio.CancelledError:
                raise
            except ProviderError as exc:
                final_error = str(exc)
                can_retry = exc.retryable and retry_number < self.settings.max_retries
            except Exception as exc:
                final_error = f"Unexpected provider error: {exc}"
                can_retry = retry_number < self.settings.max_retries

            if not can_retry:
                break
            self._record_retry(message_id)
            self.logger.warning(
                "message_retry_scheduled",
                extra={
                    "message_id": message_id,
                    "question": message.question,
                    "error": final_error,
                    "retry_count": retry_number + 1,
                    "worker": worker_index,
                },
            )
            if self.settings.retry_delay:
                await asyncio.sleep(self.settings.retry_delay)

        error_message = (
            f"LLM request failed after {self._get_retry_count(message_id)} retries: "
            f"{final_error or 'unknown provider error'}"
        )
        self._save_failure(message_id, error_message)
        self.logger.error(
            "message_failed",
            extra={
                "message_id": message_id,
                "question": message.question,
                "error": error_message,
                "retry_count": self._get_retry_count(message_id),
                "worker": worker_index,
            },
        )

    def _record_retry(self, message_id: str) -> None:
        with self.database.session() as session:
            message = session.get(ChatMessage, message_id)
            if message is not None:
                message.retry_count += 1
                session.add(message)
                session.commit()

    def _get_retry_count(self, message_id: str) -> int:
        with self.database.session() as session:
            message = session.get(ChatMessage, message_id)
            return message.retry_count if message else 0

    def _save_success(self, message_id: str, answer: str) -> None:
        completed_at = utc_now()
        with self.database.session() as session:
            message = session.get(ChatMessage, message_id)
            if message is None:
                return
            message.answer = answer
            message.error = None
            message.status = MessageStatus.COMPLETED.value
            message.completed_at = completed_at
            message.processing_time_ms = _elapsed_ms(
                message.started_at or completed_at, completed_at
            )
            session.add(message)
            session.commit()

    def _save_failure(self, message_id: str, error: str) -> None:
        completed_at = utc_now()
        with self.database.session() as session:
            message = session.get(ChatMessage, message_id)
            if message is None:
                return
            message.error = error
            message.status = MessageStatus.FAILED.value
            message.completed_at = completed_at
            message.processing_time_ms = _elapsed_ms(
                message.started_at or completed_at, completed_at
            )
            session.add(message)
            session.commit()

