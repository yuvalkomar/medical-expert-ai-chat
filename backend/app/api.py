from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy import func
from sqlmodel import select

from backend.app.database import Database
from backend.app.models import ChatMessage, Conversation, MessageStatus, utc_now
from backend.app.schemas import (
    ChatCreatedResponse,
    ChatRequest,
    ChatStatusResponse,
    ConversationMessageResponse,
    ConversationResponse,
    StatisticsResponse,
)
from backend.app.workers import WorkerPool

router = APIRouter()


def _resources(request: Request) -> tuple[Database, WorkerPool]:
    return request.app.state.database, request.app.state.worker_pool


@router.post(
    "/chat",
    response_model=ChatCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_chat(payload: ChatRequest, request: Request) -> ChatCreatedResponse:
    database, workers = _resources(request)
    with database.session() as session:
        if payload.conversation_id:
            conversation = session.get(Conversation, payload.conversation_id)
            if conversation is None:
                raise HTTPException(status_code=404, detail="Conversation not found")
            conversation.updated_at = utc_now()
        else:
            conversation = Conversation()
        session.add(conversation)
        # There is deliberately no ORM relationship object to keep the models simple;
        # flush the parent explicitly so SQLite can enforce the foreign key safely.
        session.flush()
        message = ChatMessage(
            conversation_id=conversation.id,
            question=payload.question,
        )
        session.add(message)
        session.commit()

    try:
        await workers.enqueue(message.id)
    except RuntimeError as exc:
        with database.session() as session:
            persisted = session.get(ChatMessage, message.id)
            if persisted is not None:
                persisted.status = MessageStatus.FAILED.value
                persisted.error = "The processing service is shutting down"
                persisted.completed_at = utc_now()
                persisted.processing_time_ms = 0
                session.add(persisted)
                session.commit()
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ChatCreatedResponse(messageId=message.id, conversationId=conversation.id)


@router.get(
    "/chat/{message_id}",
    response_model=ChatStatusResponse,
    response_model_exclude_none=True,
)
async def get_chat(message_id: str, request: Request) -> ChatStatusResponse:
    database, _ = _resources(request)
    with database.session() as session:
        message = session.get(ChatMessage, message_id)
        if message is None:
            raise HTTPException(status_code=404, detail="Message not found")
        return ChatStatusResponse(
            status=message.status,
            answer=message.answer if message.status == MessageStatus.COMPLETED.value else None,
            error=message.error if message.status == MessageStatus.FAILED.value else None,
        )


@router.get("/statistics", response_model=StatisticsResponse)
async def get_statistics(request: Request) -> StatisticsResponse:
    database, _ = _resources(request)
    terminal_statuses = [MessageStatus.COMPLETED.value, MessageStatus.FAILED.value]
    with database.session() as session:
        processed = session.exec(
            select(func.count()).select_from(ChatMessage).where(
                ChatMessage.status.in_(terminal_statuses)
            )
        ).one()
        succeeded = session.exec(
            select(func.count()).select_from(ChatMessage).where(
                ChatMessage.status == MessageStatus.COMPLETED.value
            )
        ).one()
        failed = session.exec(
            select(func.count()).select_from(ChatMessage).where(
                ChatMessage.status == MessageStatus.FAILED.value
            )
        ).one()
        total_retries = session.exec(
            select(func.coalesce(func.sum(ChatMessage.retry_count), 0))
        ).one()
        average_ms = session.exec(
            select(func.coalesce(func.avg(ChatMessage.processing_time_ms), 0)).where(
                ChatMessage.status.in_(terminal_statuses)
            )
        ).one()
    return StatisticsResponse(
        messagesProcessed=int(processed),
        messagesSucceeded=int(succeeded),
        messagesFailed=int(failed),
        totalRetries=int(total_retries),
        averageProcessingTimeMs=round(float(average_ms), 2),
    )


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse,
    response_model_exclude_none=True,
)
async def get_conversation(conversation_id: str, request: Request) -> ConversationResponse:
    database, _ = _resources(request)
    with database.session() as session:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        messages = session.exec(
            select(ChatMessage)
            .where(ChatMessage.conversation_id == conversation_id)
            .order_by(ChatMessage.created_at, ChatMessage.id)
        ).all()
        return ConversationResponse(
            conversationId=conversation_id,
            messages=[
                ConversationMessageResponse(
                    messageId=item.id,
                    question=item.question,
                    answer=item.answer,
                    status=item.status,
                    error=item.error,
                    createdAt=item.created_at,
                )
                for item in messages
            ],
        )


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
