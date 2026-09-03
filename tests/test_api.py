import time

from backend.app.llm.mock import MockLLMProvider
from tests.conftest import wait_for_terminal


def test_post_returns_immediately_and_message_completes(client_factory):
    provider = MockLLMProvider(response_delay=0.3)
    with client_factory(provider) as client:
        started = time.monotonic()
        response = client.post("/chat", json={"question": "What can cause a headache?"})
        elapsed = time.monotonic() - started

        assert response.status_code == 202
        assert set(response.json()) == {"messageId", "conversationId"}
        assert elapsed < 0.2
        message_id = response.json()["messageId"]
        assert client.get(f"/chat/{message_id}").json() == {"status": "processing"}

        completed = wait_for_terminal(client, message_id)
        assert completed["status"] == "completed"
        assert "headache" in completed["answer"]


def test_validation_unknown_ids_and_unknown_conversations(client_factory):
    with client_factory(MockLLMProvider(response_delay=0)) as client:
        assert client.post("/chat", json={"question": "   "}).status_code == 422
        assert client.post("/chat", json={}).status_code == 422
        assert client.get("/chat/not-a-real-id").status_code == 404
        response = client.post(
            "/chat",
            json={"question": "Question", "conversationId": "not-a-conversation"},
        )
        assert response.status_code == 404


def test_retries_then_succeeds_and_statistics_are_persisted(client_factory):
    provider = MockLLMProvider(response_delay=0, failures_before_success=2)
    with client_factory(provider, max_retries=3) as client:
        created = client.post("/chat", json={"question": "Why am I tired?"}).json()
        result = wait_for_terminal(client, created["messageId"])
        assert result["status"] == "completed"
        assert provider.attempts["Why am I tired?"] == 3

        statistics = client.get("/statistics").json()
        assert statistics["messagesProcessed"] == 1
        assert statistics["messagesSucceeded"] == 1
        assert statistics["messagesFailed"] == 0
        assert statistics["totalRetries"] == 2
        assert statistics["averageProcessingTimeMs"] >= 0


def test_message_fails_after_retry_exhaustion(client_factory):
    question = "Always fail"
    provider = MockLLMProvider(
        response_delay=0,
        always_fail_questions={question},
    )
    with client_factory(provider, max_retries=2) as client:
        created = client.post("/chat", json={"question": question}).json()
        result = wait_for_terminal(client, created["messageId"])

        assert result["status"] == "failed"
        assert "after 2 retries" in result["error"]
        assert provider.attempts[question] == 3
        assert client.get("/statistics").json() == {
            "messagesProcessed": 1,
            "messagesSucceeded": 0,
            "messagesFailed": 1,
            "totalRetries": 2,
            "averageProcessingTimeMs": client.get("/statistics").json()[
                "averageProcessingTimeMs"
            ],
        }


def test_multiple_messages_are_processed_concurrently(client_factory):
    provider = MockLLMProvider(response_delay=0.15)
    with client_factory(provider, max_concurrency=3) as client:
        ids = [
            client.post("/chat", json={"question": f"Question {index}"}).json()[
                "messageId"
            ]
            for index in range(6)
        ]
        for message_id in ids:
            assert wait_for_terminal(client, message_id)["status"] == "completed"
        assert provider.max_active_calls == 3


def test_conversation_history_is_passed_in_chronological_order(client_factory):
    provider = MockLLMProvider(
        response_delay=0,
        response_factory=lambda turns: f"Answer to: {turns[-1].content}",
    )
    with client_factory(provider) as client:
        first = client.post("/chat", json={"question": "First question"}).json()
        assert wait_for_terminal(client, first["messageId"])["status"] == "completed"

        second = client.post(
            "/chat",
            json={
                "question": "Follow-up question",
                "conversationId": first["conversationId"],
            },
        ).json()
        assert wait_for_terminal(client, second["messageId"])["status"] == "completed"

        assert [(turn.role, turn.content) for turn in provider.calls[-1]] == [
            ("user", "First question"),
            ("assistant", "Answer to: First question"),
            ("user", "Follow-up question"),
        ]
        history = client.get(f'/conversations/{first["conversationId"]}')
        assert history.status_code == 200
        assert [item["question"] for item in history.json()["messages"]] == [
            "First question",
            "Follow-up question",
        ]


def test_database_survives_application_restart(client_factory):
    first_provider = MockLLMProvider(response_delay=0)
    with client_factory(first_provider, database_name="persistent.db") as client:
        created = client.post("/chat", json={"question": "Persist me"}).json()
        assert wait_for_terminal(client, created["messageId"])["status"] == "completed"

    second_provider = MockLLMProvider(response_delay=0)
    with client_factory(second_provider, database_name="persistent.db") as client:
        persisted = client.get(f'/chat/{created["messageId"]}')
        assert persisted.status_code == 200
        assert persisted.json()["status"] == "completed"
        assert client.get("/statistics").json()["messagesProcessed"] == 1

