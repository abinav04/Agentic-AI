from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.graph.workflow import KestrelResearchAssistantPipeline
from app.config import settings

router = APIRouter()

class ChatRequest(BaseModel):
    question: str = Field(description="User question regarding Kestrel internal documentation")
    conversation_id: str = Field(default="default_conv", description="Unique conversation identifier for multi-turn tracking")
    messages: List[Dict[str, str]] = Field(default=[], description="Prior turn history [{'role': 'user'|'assistant', 'content': '...'}]")
    primary_provider: Optional[str] = Field(default=None, description="Groq or Gemini override")
    fallback_enabled: Optional[bool] = Field(default=None, description="Enable or disable automatic provider failover")

class ChatResponse(BaseModel):
    conversation_id: str
    question: str
    final_answer: str
    citations: List[Dict[str, str]]
    formatted_citations: str
    verifier_verdict: str
    llm_provider: str
    llm_model: str
    fallback_occurred: bool
    latency_seconds: float
    agent_steps: List[str]
    retrieved_chunk_ids: List[str]
    planner_output: Dict[str, Any]
    verifier_results: Dict[str, Any]
    errors: List[str]

# FastAPI endpoint handler for processing interactive user chat queries.
# Triggers multi-agent pipeline execution and returns detailed answer payloads with citations and metadata.
@router.post("/chat", response_model=ChatResponse)
def execute_chat(request: ChatRequest):
    """Executes 4-agent research workflow on user question."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    provider_name = request.primary_provider or settings.LLM_PRIMARY_PROVIDER
    fallback_active = request.fallback_enabled if request.fallback_enabled is not None else settings.LLM_FALLBACK_ENABLED

    pipeline = KestrelResearchAssistantPipeline(
        primary_provider=provider_name,
        fallback_enabled=fallback_active
    )

    state = pipeline.run(
        question=request.question,
        conversation_id=request.conversation_id,
        messages=request.messages
    )

    return ChatResponse(
        conversation_id=state["conversation_id"],
        question=state["original_question"],
        final_answer=state["final_answer"],
        citations=state["citations"],
        formatted_citations=state["formatted_citations"],
        verifier_verdict=state["verifier_verdict"],
        llm_provider=state["llm_provider"],
        llm_model=state["llm_model"],
        fallback_occurred=state["fallback_occurred"],
        latency_seconds=state["latency_seconds"],
        agent_steps=state["agent_steps"],
        retrieved_chunk_ids=state["retrieved_chunk_ids"],
        planner_output=state.get("planner_output", {}),
        verifier_results=state.get("verifier_results", {}),
        errors=state["errors"]
    )
