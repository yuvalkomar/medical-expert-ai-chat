from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class APIModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class ChatRequest(APIModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, alias="conversationId", max_length=64)

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be blank")
        return cleaned


class ChatCreatedResponse(APIModel):
    message_id: str = Field(alias="messageId")
    conversation_id: str = Field(alias="conversationId")


class ChatStatusResponse(APIModel):
    status: str
    answer: str | None = None
    error: str | None = None


class StatisticsResponse(APIModel):
    messages_processed: int = Field(alias="messagesProcessed")
    messages_succeeded: int = Field(alias="messagesSucceeded")
    messages_failed: int = Field(alias="messagesFailed")
    total_retries: int = Field(alias="totalRetries")
    average_processing_time_ms: float = Field(alias="averageProcessingTimeMs")


class ConversationMessageResponse(APIModel):
    message_id: str = Field(alias="messageId")
    question: str
    answer: str | None = None
    status: str
    error: str | None = None
    created_at: datetime = Field(alias="createdAt")


class ConversationResponse(APIModel):
    conversation_id: str = Field(alias="conversationId")
    messages: list[ConversationMessageResponse]

