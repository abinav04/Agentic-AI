# Kestrel Labs Multi-Agent Research Assistant - System Architecture

This document details the production architecture, state transitions, retrieval design, model failover strategy, and observability integration for the Kestrel Multi-Agent Research Assistant.

---

## 1. System Architecture Diagram

```
User / Streamlit UI
       │
       ▼
   FastAPI REST API (/api/chat)
       │
       ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                        LangGraph StateGraph Workflow                   │
 │                                                                        │
 │   START ──► Router / Planner Agent                                     │
 │                 │ (Decomposes query & resolves context)                │
 │                 ▼                                                      │
 │             Retriever Agent ◄────────────┐ (Retry if insufficient)     │
 │                 │                        │                             │
 │                 ├──► search_corpus Tool  │                             │
 │                 │        │               │                             │
 │                 │        ▼               │                             │
 │                 │    Chroma Vector DB    │                             │
 │                 │    (154 chunks)        │                             │
 │                 ▼                        │                             │
 │             Verifier / Critic Agent ─────┘                             │
 │                 │ (Date-aware claim verification)                      │
 │                 ▼                                                      │
 │             Synthesizer Agent                                          │
 │                 │ (Grounded answer + Option A citations)               │
 │                 ▼                                                      │
 │                END                                                     │
 └────────────────────────────────────────────────────────────────────────┘
       │
       ▼
 Provider Engine (Fallback Wrapper)
  ┌─────────────────────────────────┐
  │ Primary LLM (e.g. Groq)         │
  │     │ success        │ failure  │
  │     ▼                ▼          │
  │   Result      Fallback (Gemini) │
  └─────────────────────────────────┘
```

---

## 2. 4-Agent Responsibilities

### 1. Router / Planner Agent (`app/agents/router.py`)
- Analyzes incoming question and multi-turn conversation history.
- Resolves follow-up pronouns (e.g., *"them"*, *"it"*, *"how many can I create?"*) into explicit search queries.
- Classifies query intent (`single_hop`, `multi_hop`, `conflicting_evidence`, `unsupported_evidence`, `follow_up`).
- Decomposes complex multi-hop questions into 1 to 3 subqueries.
- Does **not** attempt to answer the question directly.

### 2. Retriever Agent (`app/agents/retriever.py`)
- Receives retrieval plan from Router Agent.
- Invokes `@tool search_corpus(query, top_k=8)` for each subquery.
- Embeds queries locally via `sentence-transformers/all-MiniLM-L6-v2`.
- Deduplicates and ranks retrieved candidate chunks.

### 3. Verifier / Critic Agent (`app/agents/verifier.py`)
- Receives retrieved evidence chunks.
- Checks every material factual claim against retrieved evidence.
- Detects evidence conflicts (e.g., release notes vs updated specifications).
- Compares `published` date metadata (e.g., `20260203` vs `20250121`) to establish active current truth.
- Assigns verdicts: `supported`, `partially_supported`, `conflicting_evidence`, `insufficient_evidence`.

### 4. Synthesizer Agent (`app/agents/synthesizer.py`)
- Generates final user-facing answer using **only** verified evidence chunks.
- Strictly adheres to corpus-only grounding rules.
- Appends Option A format citations:
  ```
  Sources:
  - chunk_id — title
  ```
- Handles unsupported questions by stating: *"Insufficient evidence: the provided Kestrel documentation does not establish whether..."*.

---

## 3. Model Provider Architecture & Failover

The system abstracts LLM providers behind `BaseLLMProvider`:
- `GroqProvider`: Calls Groq API (`llama-3.3-70b-versatile`).
- `GeminiProvider`: Calls Google Gemini API (`gemini-2.5-flash`).
- `FallbackLLMProvider`: Manages primary provider selection and automatic failover.

### Failover Trigger Conditions:
- HTTP 429 Rate Limit
- API Timeout
- 5xx Server Error
- API Key Unconfigured / Network Failure

When primary fails, the provider engine logs a warning, sets `fallback_occurred = True`, automatically executes via the fallback provider, and updates Streamlit UI and LangSmith trace metadata.

---

## 4. Local Embeddings & Chroma Storage

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (running 100% locally).
- **Vector Database**: Persistent Chroma DB store located at `./chroma_db`.
- **Indexed Chunks**: All 154 chunks from `corpus.jsonl` with full metadata (`chunk_id`, `doc_id`, `title`, `category`, `owner`, `source_url`, `published`, `version`).

---

## 5. LangSmith Observability

LangSmith tracing is integrated across the pipeline via environment configuration:
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT=kestrel-research-assistant`
- `LANGSMITH_TRACING_V2=true`

Every node execution and `search_corpus` tool call is captured in LangSmith with metadata:
- `conversation_id`
- `llm_provider`
- `llm_model`
- `verifier_verdict`
- `retrieved_chunk_ids`
