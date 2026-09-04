import json
import os
import time
from typing import Any

import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "1"))
HTTP_TIMEOUT = 10

st.set_page_config(page_title="Medical Expert AI Chat", page_icon="🩺")
st.title("Medical Expert AI Chat")
st.caption(
    "General medical information only — this service does not replace a licensed clinician."
)

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_message_id" not in st.session_state:
    st.session_state.pending_message_id = None


def api_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    response = requests.request(
        method,
        f"{BACKEND_URL}{path}",
        timeout=HTTP_TIMEOUT,
        **kwargs,
    )
    response.raise_for_status()
    return response.json()


def stream_message(message_id: str, placeholder: Any) -> tuple[str, str]:
    """Consume the backend SSE stream and update the current chat bubble."""
    assembled = ""
    event_name = ""
    data_lines: list[str] = []
    with requests.get(
        f"{BACKEND_URL}/chat/{message_id}/stream",
        stream=True,
        timeout=(HTTP_TIMEOUT, 300),
        headers={"Accept": "text/event-stream"},
    ) as response:
        response.raise_for_status()
        response.encoding = "utf-8"
        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())
            elif line == "":
                if not event_name:
                    continue
                payload = json.loads("\n".join(data_lines) or "{}")
                if event_name == "chunk":
                    assembled += payload.get("text", "")
                    placeholder.markdown(f"{assembled}▌")
                elif event_name == "reset":
                    assembled = ""
                    placeholder.info("The provider is retrying…")
                elif event_name == "completed":
                    answer = payload.get("answer", assembled)
                    placeholder.markdown(answer)
                    return "completed", answer
                elif event_name == "failed":
                    error = payload.get("error", "Unknown processing error")
                    placeholder.error(f"Processing failed: {error}")
                    return "failed", error
                event_name = ""
                data_lines = []
    raise requests.ConnectionError("The response stream ended before completion")


with st.sidebar:
    st.header("Conversation")
    if st.button("Start new conversation", use_container_width=True):
        st.session_state.conversation_id = None
        st.session_state.messages = []
        st.session_state.pending_message_id = None
        st.rerun()

    if st.button("Refresh statistics", use_container_width=True):
        try:
            st.session_state.statistics = api_request("GET", "/statistics")
        except requests.RequestException as exc:
            st.error(f"Could not load statistics: {exc}")

    statistics = st.session_state.get("statistics")
    if statistics:
        st.metric("Processed", statistics["messagesProcessed"])
        st.metric("Succeeded", statistics["messagesSucceeded"])
        st.metric("Failed", statistics["messagesFailed"])
        st.metric("Retries", statistics["totalRetries"])
        st.metric(
            "Average processing",
            f'{statistics["averageProcessingTimeMs"]:.0f} ms',
        )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("failed"):
            st.error(message["content"])
        else:
            st.markdown(message["content"])

pending_id = st.session_state.pending_message_id
should_poll_again = False
if pending_id:
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.info("Connecting to the response stream…")
        try:
            final_status, content = stream_message(pending_id, response_placeholder)
            if final_status == "completed":
                st.session_state.messages.append(
                    {"role": "assistant", "content": content}
                )
            else:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"Processing failed: {content}",
                        "failed": True,
                    }
                )
            st.session_state.pending_message_id = None
            st.rerun()
        except (requests.RequestException, ValueError) as exc:
            # Streaming is a bonus path; retain the required polling behavior as fallback.
            try:
                result = api_request("GET", f"/chat/{pending_id}")
                if result["status"] == "completed":
                    st.session_state.messages.append(
                        {"role": "assistant", "content": result["answer"]}
                    )
                    st.session_state.pending_message_id = None
                    st.rerun()
                elif result["status"] == "failed":
                    error = result.get("error", "Unknown error")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"Processing failed: {error}",
                            "failed": True,
                        }
                    )
                    st.session_state.pending_message_id = None
                    st.rerun()
                else:
                    response_placeholder.info(
                        f"Live stream unavailable ({exc}); continuing with polling…"
                    )
                    should_poll_again = True
            except requests.RequestException as polling_error:
                response_placeholder.warning(
                    f"The backend is temporarily unavailable: {polling_error}"
                )
                should_poll_again = True

question = st.chat_input(
    "Ask a medical question",
    disabled=bool(st.session_state.pending_message_id),
)
if question:
    request_body: dict[str, str] = {"question": question}
    if st.session_state.conversation_id:
        request_body["conversationId"] = st.session_state.conversation_id
    try:
        created = api_request("POST", "/chat", json=request_body)
        st.session_state.conversation_id = created["conversationId"]
        st.session_state.pending_message_id = created["messageId"]
        st.session_state.messages.append({"role": "user", "content": question})
        st.rerun()
    except requests.RequestException as exc:
        st.error(f"Could not submit the question: {exc}")

if should_poll_again:
    time.sleep(POLL_INTERVAL)
    st.rerun()

