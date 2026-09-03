import os
import time
from typing import Any

import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "1"))
HTTP_TIMEOUT = 10

st.set_page_config(
    page_title="Medical Expert AI Chat",
    page_icon="🩺",
    initial_sidebar_state="expanded",
)
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
if "statistics" not in st.session_state:
    st.session_state.statistics = None


def api_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    response = requests.request(
        method,
        f"{BACKEND_URL}{path}",
        timeout=HTTP_TIMEOUT,
        **kwargs,
    )
    response.raise_for_status()
    return response.json()


with st.sidebar:
    st.header("Conversation")
    if st.button("Start new conversation", use_container_width=True):
        st.session_state.conversation_id = None
        st.session_state.messages = []
        st.session_state.pending_message_id = None
        st.rerun()

    if st.session_state.conversation_id:
        st.caption(f"Conversation: {st.session_state.conversation_id}")


st.subheader("Backend statistics")
if st.button("Request / refresh statistics", use_container_width=False):
    try:
        st.session_state.statistics = api_request("GET", "/statistics")
    except requests.RequestException as exc:
        st.error(f"Could not load backend statistics: {exc}")

statistics = st.session_state.statistics
if statistics:
    metric_columns = st.columns(5)
    metric_columns[0].metric("Processed", statistics["messagesProcessed"])
    metric_columns[1].metric("Succeeded", statistics["messagesSucceeded"])
    metric_columns[2].metric("Failed", statistics["messagesFailed"])
    metric_columns[3].metric("Retries", statistics["totalRetries"])
    metric_columns[4].metric(
        "Average time",
        f'{statistics["averageProcessingTimeMs"]:.0f} ms',
    )
else:
    st.caption("Select the button to request current statistics from the FastAPI backend.")

st.divider()


pending_id = st.session_state.pending_message_id
should_poll_again = False
if pending_id:
    try:
        result = api_request("GET", f"/chat/{pending_id}")
        if result["status"] == "completed":
            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"]}
            )
            st.session_state.pending_message_id = None
        elif result["status"] == "failed":
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f'Processing failed: {result.get("error", "Unknown error")}',
                    "failed": True,
                }
            )
            st.session_state.pending_message_id = None
        else:
            should_poll_again = True
    except requests.RequestException as exc:
        st.warning(f"The backend is temporarily unavailable: {exc}")
        should_poll_again = True


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("failed"):
            st.error(message["content"])
        else:
            st.markdown(message["content"])

if st.session_state.pending_message_id:
    with st.chat_message("assistant"):
        st.info("Thinking… The backend is processing your question.")

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
