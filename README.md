# Medical Expert AI Chat

A self-contained assignment implementation for submitting medical-information questions,
processing them asynchronously through an LLM, and retrieving answers and persisted statistics.
It includes a FastAPI backend, a Streamlit HTTP client, SQLite persistence, an `asyncio.Queue`
worker pool, AWS Bedrock and mock LLM providers, structured logging, retries, tests, and Docker
support. Bedrock is accessed through its model-independent Converse API, so the configured model
can be Anthropic Claude, Amazon Nova, or another Converse-compatible model.

This application provides general medical information. It is not a diagnostic tool and does not
replace a licensed healthcare professional.

## Architecture

```text
Streamlit client
      |
      | HTTP: submit, poll, statistics
      v
FastAPI  --->  SQLite (conversations, messages, status, metrics)
      |
      | message ID
      v
asyncio.Queue  --->  fixed async worker pool
                           |
                           v
                    LLMProvider interface
                       /          \
             Mock provider      AWS Bedrock Runtime
                                      |
                                      v
                          Configured Bedrock model
                            (Claude, Nova, etc.)
```

`POST /chat` validates and commits a message with `processing` status, enqueues only its ID, and
returns HTTP 202 without waiting for the provider. A worker opens a session to load the message
and completed conversation history, closes it, waits for the provider, and then opens a fresh
session to save the result. No SQLModel session is shared between requests or workers, and no
session remains open during an external LLM call.

### Technology choices

- **FastAPI** provides typed validation, an OpenAPI schema, and a clean lifespan hook for worker
  startup and shutdown.
- **Streamlit** keeps the assignment UI small while remaining a real HTTP client of the backend.
- **SQLite + SQLModel** provides durable, inspectable storage with minimal operational overhead.
  WAL mode, foreign keys, a connection timeout, and short transactions make it appropriate for
  this bounded, single-service assignment.
- **`asyncio.Queue` + async workers** gives immediate submissions, natural backpressure at the
  worker boundary, simple lifecycle control, and no extra infrastructure. `MAX_CONCURRENCY`
  fixes the maximum number of concurrent provider calls.
- **A provider interface** isolates application logic from AWS. The mock implementation makes
  development and all tests deterministic and credential-free; the Bedrock implementation uses
  Bedrock Runtime's model-agnostic Converse API, supporting Claude, Amazon Nova, and other
  conversational models.

This is intentionally a self-contained assignment solution. In production, SQLite and the
in-process queue could be replaced with a transactional database and durable broker/worker
system, without changing the public API or the provider boundary.

## Quick start with Docker

Docker is the simplest way to run the complete application. The host does not need to run
`python -m backend.run`; that command is already the backend container's startup command.

### Mock mode (no AWS account or token)

```powershell
Copy-Item .env.example .env
docker compose up -d --build
```

Open:

- Streamlit UI: `http://localhost:8501`
- FastAPI documentation: `http://localhost:8000/docs`
- Backend health check: `http://localhost:8000/health`

The example configuration defaults to `LLM_PROVIDER=mock`, so it makes no external request and
incurs no AWS cost.

### AWS Bedrock mode

Create the private `.env` file if it does not exist, then set a Bedrock API key and a model/profile
that is valid for the selected source Region. For the currently tested Amazon Nova configuration:

```env
LLM_PROVIDER=bedrock
LLM_MODEL=eu.amazon.nova-pro-v1:0
AWS_REGION=eu-north-1
AWS_BEARER_TOKEN_BEDROCK=replace-with-your-real-bedrock-api-key
```

For Claude Haiku 4.5 in the same EU Region, the model can instead be:

```env
LLM_MODEL=eu.anthropic.claude-haiku-4-5-20251001-v1:0
```

Never put a real key in `.env.example`, source code, tests, screenshots, or Git. The private
`.env` file is excluded by `.gitignore` and passed into the containers by Docker Compose.

After changing application code, rebuild and recreate the services:

```powershell
docker compose up -d --build
```

After changing only `.env`, recreating the backend is sufficient:

```powershell
docker compose up -d --force-recreate backend
```

Inspect startup and processing logs with:

```powershell
docker compose logs -f backend
```

Verify the active non-secret configuration without displaying the key:

```powershell
docker compose exec backend python -c "import os; print('model:', os.getenv('LLM_MODEL')); print('region:', os.getenv('AWS_REGION')); print('token loaded:', bool(os.getenv('AWS_BEARER_TOKEN_BEDROCK')))"
```

## Processing, retries, and recovery

Each worker loads only previous **completed** question/answer pairs from the same conversation,
in chronological order. Those pairs and the current question are passed as chat messages; the
medical system prompt remains a separate provider argument.

An initial provider attempt may be followed by up to `MAX_RETRIES` additional attempts. A failed
retryable call increments the message's persisted `retry_count`, logs the event, waits
`RETRY_DELAY` seconds, and tries again. Non-retryable provider errors (for example a Bedrock
validation or access-denied error) fail immediately. Terminal results persist timestamps and
`processing_time_ms`; `/statistics` calculates its values from the database rather than volatile
counters.

During normal shutdown the backend lets queued work finish for up to
`SHUTDOWN_GRACE_PERIOD`. Any message still marked `processing` after interruption is discovered
and re-enqueued on the next startup. This closes the common commit-before-enqueue crash gap,
although an in-flight LLM request may be repeated after an abrupt process failure.

## Configuration

Copy the example and edit it as needed:

```bash
cp .env.example .env
```

On PowerShell, use `Copy-Item .env.example .env`.

| Variable | Default | Purpose |
|---|---:|---|
| `SERVER_HOST` | `0.0.0.0` | Backend bind address |
| `SERVER_PORT` | `8000` | Backend port |
| `DATABASE_URL` | `sqlite:///./data/medical_chat.db` | SQLAlchemy database URL |
| `LLM_PROVIDER` | `mock` | `mock` or `bedrock` |
| `LLM_MODEL` | Claude model ID | Bedrock model or inference-profile ID |
| `LLM_TEMPERATURE` | `0.2` | Bedrock generation temperature |
| `LLM_MAX_TOKENS` | `1000` | Bedrock response-token limit |
| `MAX_RETRIES` | `3` | Additional attempts after the initial call |
| `RETRY_DELAY` | `2` | Seconds between retries |
| `MAX_CONCURRENCY` | `5` | Number of worker tasks/provider calls |
| `SHUTDOWN_GRACE_PERIOD` | `15` | Seconds to drain work before cancellation |
| `AWS_REGION` | `us-east-1` | Bedrock Runtime region |
| `AWS_BEARER_TOKEN_BEDROCK` | unset | Bedrock bearer API key; secret, only set in private `.env` or runtime environment |
| `AWS_BEDROCK_ENDPOINT_URL` | unset | Optional custom Bedrock endpoint |
| `LOG_LEVEL` | `INFO` | Python log level |
| `LOG_FILE` | `./logs/medical_chat.log` | Rotating JSON-lines log path |
| `MOCK_RESPONSE_DELAY` | `0.1` | Simulated latency in seconds |
| `MOCK_FAILURES_BEFORE_SUCCESS` | `0` | Simulated failures per distinct question |
| `BACKEND_URL` | `http://localhost:8000` | Backend URL used by Streamlit |
| `POLL_INTERVAL` | `1` | Streamlit polling interval in seconds |

Configuration is validated at startup. Real secrets do not belong in `.env.example` or source
control.

### AWS Bedrock

Set `LLM_PROVIDER=bedrock`, choose a Converse-compatible model or inference profile available from
`AWS_REGION`, and make sure model access is enabled for the account. Geographic inference-profile
prefixes must agree with the source Region: for example, use an `eu.` profile from `eu-north-1`
and a `us.` profile from a supported US Region.

The easiest Docker authentication method is a Bedrock API key in the private `.env` file:

```env
AWS_BEARER_TOKEN_BEDROCK=replace-with-your-real-bedrock-api-key
```

Boto3 also supports its standard credential chain: `AWS_ACCESS_KEY_ID` plus
`AWS_SECRET_ACCESS_KEY` (and `AWS_SESSION_TOKEN` for temporary credentials), an AWS credentials
file/profile, workload identity, or an attached IAM role. The identity needs
`bedrock:InvokeModel` permission for the configured inference profile and its destination models.
See the AWS documentation for [Bedrock API keys](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys-use.html)
and [Converse](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html).
Credentials and authorization data are never deliberately logged.

## Run locally

Python 3.11 or newer is required.

```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
python -m backend.run
```

In another terminal:

```powershell
./.venv/Scripts/Activate.ps1
streamlit run frontend/app.py
```

Open `http://localhost:8501`. FastAPI documentation is available at
`http://localhost:8000/docs`. The default mock provider needs no AWS account or network call.

## API

### Submit a question

```http
POST /chat
Content-Type: application/json

{"question":"What are common symptoms of iron deficiency?"}
```

The HTTP 202 response is immediate:

```json
{
  "messageId": "a-message-uuid",
  "conversationId": "a-conversation-uuid"
}
```

To continue a conversation, include the returned ID:

```json
{
  "question": "Which tests are commonly used?",
  "conversationId": "a-conversation-uuid"
}
```

### Poll a message

`GET /chat/{messageId}` returns one of:

```json
{"status":"processing"}
```

```json
{"status":"completed","answer":"..."}
```

```json
{"status":"failed","error":"..."}
```

Unknown IDs return 404. Empty, missing, or overlong questions return 422.

### Statistics and history

- `GET /statistics` returns `messagesProcessed`, `messagesSucceeded`, `messagesFailed`,
  `totalRetries`, and `averageProcessingTimeMs`. "Processed" means terminal (completed or failed),
  so currently queued work is not counted yet.
- `GET /conversations/{conversationId}` returns persisted messages in chronological order.
- `GET /health` is a lightweight container health endpoint.

## Docker lifecycle

The complete Docker setup is documented in [Quick start with Docker](#quick-start-with-docker).
Named volumes retain the SQLite database and logs across container recreation. Stop the services
without deleting their data using:

```powershell
docker compose down
```

## Tests

```bash
python -m pytest
```

The test suite uses temporary SQLite files and `MockLLMProvider`; it never needs AWS credentials.
It verifies immediate submission, processing/completion states, request and ID errors, retry
success, retry exhaustion, persisted statistics, actual worker concurrency, chronological
conversation context, database survival across app restarts, and the model-independent Bedrock
Converse request shape.

## Logging

The backend uses Python's thread-safe logging handlers and writes one JSON object per line to a
rotating file and stderr. Completed or failed interaction entries include timestamp, message ID,
question, response or error, retry count, and worker index. Retry events are logged separately.
Credentials and configuration secrets are not logged.

Questions and responses can themselves contain sensitive health information. A real deployment
should define retention/redaction policy, lock down log access, encrypt storage, and avoid logging
full content unless there is a justified operational need.

## Assignment coverage and bonus features

Implemented core requirements:

- Required `POST /chat`, `GET /chat/{messageId}`, and `GET /statistics` behavior
- Non-blocking submission through `asyncio.Queue`
- Configurable worker pool with graceful lifecycle management
- SQLite/SQLModel persistence with independent, short-lived sessions
- Dedicated medical safety system prompt
- Configurable retry delay/count and persisted retry metrics
- Structured, concurrency-safe rotating logs
- Bedrock Converse and mock providers behind one interface
- Streamlit submission, processing state, polling, failure handling, history, and statistics
- Environment validation, `.env.example`, Dockerfiles, Compose, and automated tests

Implemented bonuses:

- **Continuous chat:** completed conversation history is supplied to the provider.
- **Persistence across restarts:** schema/data are retained and unfinished records are recovered.

Not implemented:

- **Token streaming:** polling remains deliberately simple and reliable. Adding `ConverseStream`
  plus a persisted/event fanout layer would be a separate enhancement.

## Known limitations and production improvements

- The queue is process-local. Database recovery prevents messages from being forgotten, but only
  one backend process should consume this SQLite database at a time.
- SQLite schema creation is automatic but there is no migration framework; use Alembic for
  evolving a production schema.
- Conversation context is not token-budgeted or summarized, so very long conversations can exceed
  model limits.
- Retries use a fixed delay; production code would normally add exponential backoff, jitter,
  provider-specific throttling behavior, timeouts, and a circuit breaker.
- The assignment intentionally omits authentication, authorization, real medical validation,
  RAG, distributed infrastructure, and high availability.
- Production observability would use centralized logs, traces, health/readiness separation, and
  exported metrics. A durable broker and scalable database would allow multiple API and worker
  processes.
