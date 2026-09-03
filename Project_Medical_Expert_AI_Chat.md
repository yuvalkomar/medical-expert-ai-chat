# PROJECT: Medical Expert AI Chat

## High-Level Goal

Design and implement a self-contained service that exposes an HTTP API and a simple frontend for interacting with a Medical Expert AI Chat Agent.

The system should allow users to submit medical questions, process them asynchronously via an LLM, store and log interactions, and expose real-time usage and performance statistics.

Implementation details (language, frameworks, storage, UI technology, and LLM provider) are left to your discretion.

## Context

Your task is to implement the core business logic of a new service that consists of:

- A backend service with APIs for:
  - Submitting chat questions
  - Retrieving chat responses
  - Exposing system statistics
- A simple frontend UI that allows a user to:
  - Ask medical questions
  - View AI responses
- An asynchronous component that processes incoming questions using an LLM

The system should demonstrate clean backend development, asynchronous processing, basic error handling, logging, concurrency, and metrics.

The solution does not need to be production-grade. Focus on correctness, clarity, and reasonable engineering decisions.

## Core Concepts

### Chat Message

A chat message represents a single user question sent to the AI:

```json
{
  "question": "string"
}
```

Each message should:

- Be assigned a unique ID
- Be processed asynchronously
- Result in an AI-generated response (or a failure)

## Backend Responsibilities

### 1. Question Processing Pipeline

When a new question is submitted:

1. Store the incoming question.
2. Schedule it for asynchronous processing.
3. An asynchronous component:
   - Sends the question to an LLM using medical-expert-style system instructions.
   - Logs the interaction.
   - Handles failures and retries (configurable).

### 2. Processing Behavior

The processing component should:

- Call an LLM with:
  - A system prompt that instructs the model to behave as a medical expert.
  - Proper temperature / max tokens configuration.
- Log the interaction.
- Retry failed requests, with a configurable max retry count and a delay between attempts.

## Asynchronous Processing

- Questions must be processed asynchronously so that submitting a question does not wait for the LLM response.
- You may implement asynchronous processing using any reasonable approach, for example:
  - Background tasks
  - A worker pool
  - An async queue
  - Another mechanism
- If you implement a worker pool, you may optionally limit the number of concurrent workers/tasks and implement worker lifecycle management.

## Configuration (Environment Variables)

The server should be configured via environment variables.

For example, the following configuration may be supported:

| Variable | Description |
|---|---|
| `SERVER_PORT` | HTTP server port |
| `LLM_PROVIDER` | OpenAI / Anthropic / mock |
| `LLM_MODEL` | Model name |
| `LLM_TEMPERATURE` | Temperature setting |
| `LLM_MAX_TOKENS` | Max tokens per request |
| `RETRY_DELAY` | Delay between retries |
| `MAX_RETRIES` | Maximum retry attempts |
| `MAX_CONCURRENCY` | Max messages processed concurrently |

## Frontend Requirements

Build a very simple UI (no design polish required) that allows users to:

- Enter and submit medical questions
- See a loading state while a response is being processed
- View the AI response (via polling or fetching)
- Request and display backend statistics

## Logging Requirements

Processed messages should be logged to a log file or other reasonable storage.

Each entry should include:

- Timestamp
- Message ID
- Question
- Response or error

If your implementation performs concurrent writes to the same log, ensure that writes are handled safely.

## HTTP API

### `POST /chat`

Submit a new medical question.

#### Request

```json
{
  "question": "What are the symptoms of iron deficiency?"
}
```

#### Response

```json
{
  "messageId": "uuid-or-id"
}
```

### `GET /chat/{messageId}`

Retrieve the status and response of a previously submitted question.

#### Response - Processing

```json
{
  "status": "processing"
}
```

#### Response - Completed

```json
{
  "status": "completed",
  "answer": "Iron deficiency commonly causes fatigue, pale skin..."
}
```

#### Response - Failed

```json
{
  "status": "failed",
  "error": "LLM request failed after retries"
}
```

### `GET /statistics`

Returns current system metrics in JSON format:

```json
{
  "messagesProcessed": 120,
  "messagesSucceeded": 110,
  "messagesFailed": 10,
  "totalRetries": 27,
  "averageProcessingTimeMs": 850
}
```

## Bonus (Optional)

- Continuous chat ("agent-mode"): chat considers previous questions and answers
- Persistence across service restarts
- Streaming responses

These features are not required and should not be implemented at the expense of the core requirements.

## Non-Goals (Explicitly Out of Scope)

- Authentication / authorization
- Real medical advice validation
- Production-grade UI or infrastructure
- Distributed deployment and orchestration or high availability

**Good Luck!**
