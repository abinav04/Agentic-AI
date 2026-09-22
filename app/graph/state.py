from typing import List, Dict, Any, Optional, TypedDict

# TypedDict class defining state shared across LangGraph agent nodes.
# Stores conversation context, router subqueries, retrieved chunks, verifier verdicts, and final answers.
class ResearchState(TypedDict):
    """LangGraph State definition for Kestrel Multi-Agent Research Assistant."""

    # Conversation Context
    conversation_id: str
    messages: List[Dict[str, str]]
    original_question: str

    # Router / Planner Agent Outputs
    intent: str  # single_hop, multi_hop, follow_up, conflicting_evidence, unsupported_evidence
    subqueries: List[str]
    planner_output: Dict[str, Any]

    # Retriever Agent Outputs
    retrieved_chunks: List[Dict[str, Any]]
    retrieved_chunk_ids: List[str]
    retrieval_queries: List[str]
    retrieval_iteration: int

    # Verifier / Critic Agent Outputs
    claims: List[Dict[str, Any]]
    verifier_results: Dict[str, Any]
    verifier_verdict: str  # supported, partially_supported, conflicting_evidence, insufficient_evidence

    # Synthesizer Agent Outputs
    final_answer: str
    citations: List[Dict[str, str]]
    formatted_citations: str

    # System & Observability Metadata
    llm_provider: str
    llm_model: str
    fallback_occurred: bool
    latency_seconds: float
    agent_steps: List[str]
    errors: List[str]
