# streamlit_app/app.py
"""
Streamlit chat UI — thin client for the FastAPI backend (api/main.py).

Run:
    streamlit run streamlit_app/app.py

Requires the FastAPI backend to be running separately (default: localhost:8000):
    uvicorn api.main:app --reload

This app deliberately contains no business logic — it only calls the REST
API (see `_api_*` helper functions below) and renders the results. This
matches the Tech Stack spec's separation of "Backend API" (FastAPI +
Pydantic) from "Frontend" (Streamlit) as two independently deployable layers.
"""

import os

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 30

st.set_page_config(
    page_title="AI Customer Support Assistant",
    page_icon="🛠️",
    layout="centered",
)


# ---------------------------------------------------------------------------
# API client helpers
# ---------------------------------------------------------------------------
def _api_health() -> dict | None:
    try:
        resp = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def _api_send_message(session_id: str | None, user_id: str, text: str) -> dict:
    payload = {"user_id": user_id, "text": text}
    if session_id:
        payload["session_id"] = session_id
    resp = requests.post(f"{API_BASE_URL}/api/v1/chat", json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json()


def _api_clear_session(session_id: str) -> None:
    try:
        requests.delete(f"{API_BASE_URL}/api/v1/chat/{session_id}", timeout=10)
    except requests.RequestException:
        pass  # best-effort cleanup; not worth surfacing to the user


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str, "meta": dict|None}
if "user_id" not in st.session_state:
    st.session_state.user_id = "cust_" + os.urandom(4).hex()


# ---------------------------------------------------------------------------
# Sidebar — backend status + session controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🛠️ System Status")

    health = _api_health()
    if health is None:
        st.error(f"Backend unreachable at:\n`{API_BASE_URL}`\n\nStart it with:\n```\nuvicorn api.main:app --reload\n```")
    else:
        st.success("Backend connected")
        st.caption(f"**LLM Provider:** {health['llm_provider']}")
        st.caption(f"**LLM Available:** {'✅' if health['llm_available'] else '⚠️ Not configured (mock mode)'}")
        st.caption(f"**Embeddings:** {health['embedding_provider']}")
        st.caption(f"**Vector Store:** {health['vector_store']}")
        st.caption(f"**Knowledge Base:** {health['rag_collection_size']} chunks")
        with st.expander("Registered Agents"):
            for agent_name in health["registered_agents"]:
                st.write(f"• {agent_name}")

    st.divider()
    st.text_input("Customer ID", key="user_id", help="Simulates the logged-in customer.")

    if st.button("🔄 New Conversation", use_container_width=True):
        if st.session_state.session_id:
            _api_clear_session(st.session_state.session_id)
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()

    if st.session_state.session_id:
        st.caption(f"Session: `{st.session_state.session_id}`")


# ---------------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------------
st.title("🛠️ AI Customer Support Assistant")
st.caption("Multi-agent support: order tracking, product recommendations, returns, and general Q&A.")

# Render existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("meta"):
            meta = msg["meta"]
            st.caption(
                f"Agent: `{meta.get('agent_invoked', 'n/a')}` · "
                f"Confidence: {meta.get('confidence_score', 0):.2f}"
            )

# Chat input
if prompt := st.chat_input("How can we help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt, "meta": None})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = _api_send_message(
                    st.session_state.session_id, st.session_state.user_id, prompt
                )
                st.session_state.session_id = result["session_id"]
                st.write(result["response_text"])
                st.caption(
                    f"Agent: `{result.get('agent_invoked', 'n/a')}` · "
                    f"Confidence: {result.get('confidence_score', 0):.2f}"
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["response_text"],
                        "meta": {
                            "agent_invoked": result.get("agent_invoked"),
                            "confidence_score": result.get("confidence_score", 0),
                        },
                    }
                )
            except requests.RequestException as exc:
                error_text = f"Sorry, I couldn't reach the support backend: {exc}"
                st.error(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text, "meta": None})

# Sample prompts to help users get started on a fresh conversation
if not st.session_state.messages:
    st.divider()
    st.caption("Try asking:")
    cols = st.columns(3)
    sample_prompts = [
        "Where is my order 12345?",
        "Recommend a laptop for gaming",
        "What's your return policy?",
    ]
    for col, sample in zip(cols, sample_prompts):
        with col:
            if st.button(sample, use_container_width=True):
                st.session_state._pending_prompt = sample
                st.rerun()

# Handle a sample-prompt click (Streamlit reruns the script; chat_input can't
# be set programmatically, so we simulate the same send-flow directly here)
if "_pending_prompt" in st.session_state:
    pending = st.session_state.pop("_pending_prompt")
    st.session_state.messages.append({"role": "user", "content": pending, "meta": None})
    try:
        result = _api_send_message(st.session_state.session_id, st.session_state.user_id, pending)
        st.session_state.session_id = result["session_id"]
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["response_text"],
                "meta": {
                    "agent_invoked": result.get("agent_invoked"),
                    "confidence_score": result.get("confidence_score", 0),
                },
            }
        )
    except requests.RequestException as exc:
        st.session_state.messages.append(
            {"role": "assistant", "content": f"Sorry, I couldn't reach the support backend: {exc}", "meta": None}
        )
    st.rerun()
