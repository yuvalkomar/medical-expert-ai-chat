from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4

from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def uuid_string() -> str:
    return str(uuid4())


class MessageStatus(StrEnum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Conversation(SQLModel, table=True):
    __tablename__ = "conversations"

    id: str = Field(default_factory=uuid_string, primary_key=True)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False, index=True)


class ChatMessage(SQLModel, table=True):
    __tablename__ = "chat_messages"

    id: str = Field(default_factory=uuid_string, primary_key=True)
    conversation_id: str = Field(foreign_key="conversations.id", index=True)
    question: str
    answer: str | None = None
    status: str = Field(default=MessageStatus.PROCESSING.value, index=True)
    error: str | None = None
    retry_count: int = Field(default=0, nullable=False)
    created_at: datetime = Field(default_factory=utc_now, nullable=False, index=True)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    processing_time_ms: float | None = None

