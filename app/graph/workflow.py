import time
import logging
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from app.graph.state import ResearchState
from app.agents.router import run_router_agent
from app.agents.retriever import run_retriever_agent
from app.agents.verifier import run_verifier_agent
from app.agents.synthesizer import run_synthesizer_agent
from app.llm.fallback import FallbackLLMProvider

logger = logging.getLogger(__name__)

MAX_RETRIEVAL_ITERATIONS = 2

def create_research_graph(llm_provider: Optional[FallbackLLMProvider] = None):
    """Constructs explicit LangGraph StateGraph for multi-agent research workflow."""

    if llm_provider is None:
        llm_provider = FallbackLLMProvider()

    # Define Node functions
    def router_node(state: ResearchState) -> Dict[str, Any]:
        return run_router_agent(state, llm_provider)

    def retriever_node(state: ResearchState) -> Dict[str, Any]:
        return run_retriever_agent(state)

    def verifier_node(state: ResearchState) -> Dict[str, Any]:
        return run_verifier_agent(state, llm_provider)

    def synthesizer_node(state: ResearchState) -> Dict[str, Any]:
        return run_synthesizer_agent(state, llm_provider)

    # Conditional edge routing logic
    def should_retry_retrieval(state: ResearchState) -> str:
        verdict = state.get("verifier_verdict", "supported")
        iteration = state.get("retrieval_iteration", 1)
        intent = state.get("intent", "")

        # Only retry if verdict is insufficient, intent wasn't explicitly unsupported, and under iteration cap
        if verdict == "insufficient_evidence" and intent != "unsupported_evidence" and iteration < MAX_RETRIEVAL_ITERATIONS:
            logger.info("Verifier found insufficient evidence. Routing back to Retriever for expanded search.")
            return "retriever"
        return "synthesizer"

    # Build Graph
    builder = StateGraph(ResearchState)

    builder.add_node("router", router_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("verifier", verifier_node)
    builder.add_node("synthesizer", synthesizer_node)

    # Add Edges
    builder.add_edge(START, "router")
    builder.add_edge("router", "retriever")
    builder.add_edge("retriever", "verifier")

    builder.add_conditional_edges(
        "verifier",
        should_retry_retrieval,
        {
            "retriever": "retriever",
            "synthesizer": "synthesizer"
        }
    )

    builder.add_edge("synthesizer", END)

    return builder.compile()

class KestrelResearchAssistantPipeline:
    """Wrapper pipeline executing graph workflow and tracking timing, provider info, and state."""

    def __init__(self, primary_provider: str = "groq", fallback_enabled: bool = True):
        self.provider = FallbackLLMProvider(
            primary_name=primary_provider,
            fallback_enabled=fallback_enabled
        )
        self.graph = create_research_graph(self.provider)

    def run(
        self,
        question: str,
        conversation_id: str = "default_conv",
        messages: Optional[list] = None
    ) -> ResearchState:
        start_time = time.time()

        initial_state: ResearchState = {
            "conversation_id": conversation_id,
            "messages": messages or [],
            "original_question": question,
            "intent": "",
            "subqueries": [],
            "planner_output": {},
            "retrieved_chunks": [],
            "retrieved_chunk_ids": [],
            "retrieval_queries": [],
            "retrieval_iteration": 0,
            "claims": [],
            "verifier_results": {},
            "verifier_verdict": "supported",
            "final_answer": "",
            "citations": [],
            "formatted_citations": "",
            "llm_provider": self.provider.last_provider_used,
            "llm_model": self.provider.model_name,
            "fallback_occurred": False,
            "latency_seconds": 0.0,
            "agent_steps": [],
            "errors": []
        }

        try:
            final_state = self.graph.invoke(initial_state)
        except Exception as e:
            logger.error(f"Graph execution failed: {e}")
            final_state = initial_state
            final_state["final_answer"] = f"Error during multi-agent research execution: {str(e)}"
            final_state["errors"].append(str(e))

        elapsed = time.time() - start_time
        final_state["latency_seconds"] = round(elapsed, 3)
        final_state["llm_provider"] = self.provider.last_provider_used
        final_state["llm_model"] = self.provider.model_name
        final_state["fallback_occurred"] = self.provider.last_fallback_occurred

        return final_state
