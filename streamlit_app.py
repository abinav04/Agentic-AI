import streamlit as st
import requests
import uuid
import json
import os
import re
from app.config import settings
from app.graph.workflow import KestrelResearchAssistantPipeline

# Page Config
st.set_page_config(
    page_title="Kestrel Labs Research Assistant",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Glassmorphism Dark UI
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Glassmorphism Card Containers */
    .glass-card {
        background: rgba(22, 27, 34, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* Header Accent */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #58a6ff 0%, #a5d6ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    
    .sub-title {
        color: #8b949e;
        font-size: 1.0rem;
        margin-bottom: 24px;
    }

    /* Provider Badges */
    .badge-provider {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .badge-groq {
        background-color: rgba(240, 136, 62, 0.2);
        color: #f0883e;
        border: 1px solid rgba(240, 136, 62, 0.4);
    }
    .badge-gemini {
        background-color: rgba(56, 139, 253, 0.2);
        color: #58a6ff;
        border: 1px solid rgba(56, 139, 253, 0.4);
    }

    /* Agent Status Checkmarks */
    .agent-status-container {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .agent-step {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 0.8rem;
        color: #7ee787;
    }

    /* Dropdown Styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #58a6ff !important;
        background-color: #161b22 !important;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = f"conv_{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Controls
with st.sidebar:
    st.markdown("### 🦅 Kestrel Research Assistant")
    st.markdown("---")

    primary_provider = st.selectbox(
        "Primary LLM Provider",
        options=["groq", "gemini"],
        index=0 if settings.LLM_PRIMARY_PROVIDER.lower() == "groq" else 1,
        help="Select primary generation provider"
    )

    fallback_enabled = st.toggle(
        "Enable Provider Failover",
        value=settings.LLM_FALLBACK_ENABLED,
        help="Automatically switch provider if primary encounters rate limit (HTTP 429) or failure."
    )

    st.markdown("---")
    st.markdown("#### System Information")
    st.markdown(f"**Embedding Model:** `local sentence-transformers`")
    st.markdown(f"**Vector Database:** `Chroma DB (154 chunks)`")
    st.markdown(f"**LangSmith Observability:** `{'Active' if settings.LANGSMITH_API_KEY else 'Configured (Env)'}`")

    if st.button("Clear Conversation History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
        st.rerun()

# Header Area
st.markdown('<div class="main-title">Kestrel Labs Multi-Agent Research Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Corpus-Grounded Internal Technical Documentation Assistant with Dropdown Sources & Reasoning</div>', unsafe_allow_html=True)

def extract_clean_answer_and_sources(full_text: str):
    """Splits raw answer into clean text and sources block for dropdown rendering."""
    if not full_text:
        return "", ""
    match = re.search(r"\n\s*Sources:?", full_text, flags=re.IGNORECASE)
    if match:
        idx = match.start()
        clean_ans = full_text[:idx].strip()
        sources_block = full_text[idx:].strip()
        return clean_ans, sources_block
    return full_text.strip(), ""

def render_assistant_message(msg: dict):
    raw_content = msg.get("content", "")
    clean_answer, inline_sources = extract_clean_answer_and_sources(raw_content)

    # Main Answer Text (without inline sources)
    st.markdown(clean_answer)

    # Agent Execution Badges
    steps = msg.get("agent_steps", [])
    if steps:
        st.markdown('<div class="agent-status-container">', unsafe_allow_html=True)
        for step in steps:
            st.markdown(f'<span class="agent-step">{step}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Provider Badge
    provider_used = msg.get("llm_provider", "groq")
    fallback_used = msg.get("fallback_occurred", False)
    badge_class = "badge-groq" if provider_used == "groq" else "badge-gemini"
    fallback_text = f" (Fallback Active)" if fallback_used else ""
    st.markdown(
        f'<span class="badge-provider {badge_class}">Generated using {provider_used.upper()}{fallback_text}</span>',
        unsafe_allow_html=True
    )

    # Dropdown 1: Sources & Citations (Shown ONLY when clicked/selected)
    citations_text = msg.get("formatted_citations", "") or inline_sources
    citations_list = msg.get("citations", [])
    if citations_text or citations_list:
        with st.expander("📚 View Sources & Citations", expanded=False):
            if citations_text:
                st.markdown(citations_text)
            elif citations_list:
                st.markdown("**Sources:**")
                for c in citations_list:
                    st.markdown(f"- **{c.get('chunk_id')}** — {c.get('title')}")

    # Dropdown 2: Agent Reasoning & Plan (Shown ONLY when clicked/selected)
    planner_output = msg.get("planner_output", {})
    verifier_results = msg.get("verifier_results", {})

    with st.expander("🧠 View Agent Reasoning & Retrieval Plan", expanded=False):
        if planner_output:
            st.markdown(f"**Intent Classified:** `{planner_output.get('intent', 'single_hop')}`")
            st.markdown(f"**Planner Rationale:** {planner_output.get('reasoning', 'N/A')}")
            subq = planner_output.get("subqueries", [])
            if subq:
                st.markdown("**Search Subqueries Run:**")
                for sq in subq:
                    st.markdown(f"- `{sq}`")

        if verifier_results:
            st.markdown(f"**Verifier Verdict:** `{verifier_results.get('overall_verdict', 'supported')}`")
            conf_res = verifier_results.get("conflict_resolution")
            if conf_res:
                st.markdown(f"**Conflict Resolution Details:** {conf_res}")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_assistant_message(message)

# Chat Input Handler
if prompt := st.chat_input("Ask a question about Kestrel documentation, specs, or release notes..."):
    # Render user message immediately
    st.chat_message("user").markdown(prompt)

    # Format history for multi-turn resolution
    history_payload = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing documentation with 4-agent LangGraph workflow..."):
            api_url = f"{settings.FASTAPI_URL}/api/chat"
            response_data = None

            try:
                res = requests.post(
                    api_url,
                    json={
                        "question": prompt,
                        "conversation_id": st.session_state.conversation_id,
                        "messages": history_payload,
                        "primary_provider": primary_provider,
                        "fallback_enabled": fallback_enabled
                    },
                    timeout=60
                )
                if res.status_code == 200:
                    response_data = res.json()
            except Exception as e:
                pipeline = KestrelResearchAssistantPipeline(
                    primary_provider=primary_provider,
                    fallback_enabled=fallback_enabled
                )
                state = pipeline.run(
                    question=prompt,
                    conversation_id=st.session_state.conversation_id,
                    messages=history_payload
                )
                response_data = {
                    "final_answer": state["final_answer"],
                    "citations": state["citations"],
                    "formatted_citations": state["formatted_citations"],
                    "verifier_verdict": state["verifier_verdict"],
                    "llm_provider": state["llm_provider"],
                    "llm_model": state["llm_model"],
                    "fallback_occurred": state["fallback_occurred"],
                    "agent_steps": state["agent_steps"],
                    "planner_output": state.get("planner_output", {}),
                    "verifier_results": state.get("verifier_results", {})
                }

            if response_data:
                full_answer = response_data.get("final_answer", "")

                assistant_msg = {
                    "role": "assistant",
                    "content": full_answer,
                    "citations": response_data.get("citations", []),
                    "formatted_citations": response_data.get("formatted_citations", ""),
                    "verifier_verdict": response_data.get("verifier_verdict", "supported"),
                    "llm_provider": response_data.get("llm_provider", primary_provider),
                    "llm_model": response_data.get("llm_model", ""),
                    "fallback_occurred": response_data.get("fallback_occurred", False),
                    "agent_steps": response_data.get("agent_steps", []),
                    "planner_output": response_data.get("planner_output", {}),
                    "verifier_results": response_data.get("verifier_results", {})
                }

                render_assistant_message(assistant_msg)
                st.session_state.messages.append(assistant_msg)
