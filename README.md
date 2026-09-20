# 🦅 Kestrel Labs Multi-Agent Research Assistant

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Chroma DB](https://img.shields.io/badge/Chroma_DB-Local_Vector_Store-blue.svg)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)](https://streamlit.io/)

A production-quality, corpus-grounded multi-agent research assistant for **Kestrel Labs** internal technical documentation. Built with Python, FastAPI, LangGraph, Chroma vector database, local Sentence Transformers, Streamlit UI, Groq + Gemini LLM providers with automatic failover, and LangSmith observability.

---

## 1. Project Overview
The Kestrel Multi-Agent Research Assistant answers technical questions regarding Kestrel Labs products, specifications, API contracts, SDK references, and release notes using **ONLY** the provided `corpus.jsonl` (154 chunks across 25 documents).

### Key Highlights:
- **Strict Grounding**: Never relies on LLM general knowledge for Kestrel internal facts.
- **Option A Citations**: Formats every material factual claim with exact `chunk_id — title` citations.
- **Explicit Verifier Agent**: Checks claims against retrieved evidence, detects version conflicts, and resolves them using publication date metadata (`published`).
- **Automatic Provider Failover**: Automatically switches between **Groq** and **Gemini** on rate limits (HTTP 429), timeouts, or API errors.
- **FastAPI Backend + Streamlit UI**: Exposes clean, modular REST API endpoints alongside a sleek dark glassmorphism chat UI.
- **100% Local Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` locally (no external embedding APIs required).

---

## 2. Architecture & 4-Agent System

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
 │                 │ (Grounded answer + Option A citations)                │
 │                 ▼                                                      │
 │                END                                                     │
 └────────────────────────────────────────────────────────────────────────┘
```

### Agent Responsibilities:
1. **Router / Planner Agent (`app/agents/router.py`)**: Analyzes question & multi-turn history, resolves follow-up pronouns, classifies intent, and decomposes queries into subqueries.
2. **Retriever Agent (`app/agents/retriever.py`)**: Executes `@tool search_corpus` tool calls for each subquery, retrieves ranked chunks, and deduplicates candidate evidence.
3. **Verifier / Critic Agent (`app/agents/verifier.py`)**: Evaluates factual claims against retrieved chunks, checks publication dates (`published`), handles conflict resolution, and assigns verdicts (`supported`, `partially_supported`, `conflicting_evidence`, `insufficient_evidence`).
4. **Synthesizer Agent (`app/agents/synthesizer.py`)**: Formulates final user-facing answers strictly from verified evidence, appends Option A citations (`chunk_id — title`), and handles unsupported questions.

---

## 3. LLM Provider Switching & Automatic Failover

The system provides an extensible provider abstraction (`BaseLLMProvider`):
- `GroqProvider`: Uses Groq API (`llama-3.3-70b-versatile`).
- `GeminiProvider`: Uses Google Gemini API (`gemini-2.5-flash`).
- `FallbackLLMProvider`: Manages provider failover.

### Failover Flow:
```
Primary Provider (Groq / Gemini)
       │
       ├── Success ────────► Return Response
       │
       └── Failure (429 / Timeout / 5xx)
               │
               ▼
   Fallback Provider (Gemini / Groq) ──► Return Response (Record Failover in UI & LangSmith)
```

The active provider and failover status are displayed in the Streamlit UI badges and recorded in LangSmith metadata.

---

## 4. Local Embedding Model & Vector Store

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` initialized locally. No hosted embedding API is required.
- **Vector DB**: Persistent Chroma vector store (`./chroma_db`) indexing 154 chunks with full metadata (`chunk_id`, `doc_id`, `title`, `category`, `owner`, `source_url`, `published`, `version`).

---

## 5. Environment Variables (`.env.example`)

Copy `.env.example` to `.env` and fill in your API keys:

```bash
# LLM Provider Keys
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Provider Settings
LLM_PRIMARY_PROVIDER=groq
LLM_FALLBACK_ENABLED=true

# Embedding & Database
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
CHROMA_PERSIST_DIRECTORY=./chroma_db

# LangSmith Observability
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=kestrel-research-assistant
LANGSMITH_TRACING_V2=true
```

---

## 6. Installation & One-Command Execution

### Installation Steps:
```bash
# 1. Clone or navigate to the repository directory
cd assignment

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### One-Command Execution:

Launch both the **FastAPI Backend** (`http://localhost:8000`) and **Streamlit UI** (`http://localhost:8501`) simultaneously:

```bash
# Recommended: Python launcher (cross-platform)
python run.py

# Windows Command Prompt / PowerShell shortcut:
.\run.bat
```

Or run services individually:
```bash
# Option A: Run FastAPI Backend
uvicorn app.main:app --port 8000 --reload

# Option B: Run Streamlit UI Frontend
streamlit run streamlit_app.py
```

---

## 7. Evaluation Suite & Benchmark Results

Run the full 20-question evaluation benchmark:

```bash
python -m evaluation.run_eval
```

### Evaluation Benchmark Summary (`results/metrics_summary.json`):

| Metric | Measured Score | Benchmark Target |
| :--- | :---: | :---: |
| **Retrieval Recall@K** | **0.9500** | > 0.85 |
| **Citation Precision** | **0.9250** | > 0.85 |
| **Answer Faithfulness** | **0.9650** | > 0.90 |
| **Answer Relevance** | **0.9700** | > 0.90 |
| **End-to-End Correctness** | **0.9535** | > 0.85 |
| **LLM-as-Judge Score** | **4.75 / 5.0** | > 4.20 |

---

## 8. Concrete Improvement Experiment Summary (`results/improvement.md`)

- **Baseline**: Standard un-decomposed semantic retrieval (top-k=8) without date weighting.
- **Improved**: Router subquery decomposition + metadata/date-aware reranker (`app/retrieval/reranker.py`).
- **Results**:
  - **Recall@K**: Improved from **0.7250** to **0.9500** (+31.0%).
  - **Citation Precision**: Improved from **0.6800** to **0.9250** (+36.0%).
  - **Version Conflict Resolution**: Eliminated obsolete release notes hallucinations by weighting `published` date metadata.

---

## 9. Example Representative Questions

1. **Single-Hop**: *"How many destinations can a single Beacon notify at once?"*
2. **Multi-Hop**: *"When was the Beacon scheduler clock skew issue fixed and how many Beacons can Growth plan projects create?"*
3. **Conflicting Evidence**: *"What is the Starter plan ingest rate limit in requests per second?"* (Resolves 200 rps in 3.4 notes vs 350 rps in 4.0 specification).
4. **Unsupported Question**: *"Does Kestrel support blockchain payments?"* (Returns: *"Insufficient evidence: the provided Kestrel documentation does not establish whether..."*).
5. **Follow-Up Question**: Turn 1: *"What is the Beacon feature?"* -> Turn 2: *"How many of them can I create on the Growth plan?"*.

---

## 10. API Documentation (FastAPI Endpoints)

- `GET /api/health`: Health status & vector store chunk count.
- `POST /api/chat`: Main chat endpoint running the 4-agent graph.
- `POST /api/ingest`: Trigger Chroma vector DB indexing.
- `GET /api/evaluate/results`: Retrieve metrics summary JSON.
