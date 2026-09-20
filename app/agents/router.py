import logging
import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from app.graph.state import ResearchState
from app.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

class RouterOutputSchema(BaseModel):
    intent: str = Field(description="One of: single_hop, multi_hop, follow_up, conflicting_evidence, unsupported_evidence")
    subqueries: List[str] = Field(description="1 to 3 search subqueries derived from user question and conversation history")
    requires_multi_hop: bool = Field(description="True if question requires joining multiple distinct documentation areas")
    resolved_question: str = Field(description="Question with any follow-up pronouns ('them', 'it', 'this feature') explicitly resolved")
    reasoning: str = Field(description="Brief rationale for the plan")

ROUTER_SYSTEM_PROMPT = """You are the Router / Planner Agent for the Kestrel Labs Internal Technical Documentation System.
Your job is to analyze the user's question and produce a structured retrieval plan.

RESPONSIBILITIES:
1. Examine the user's question and any prior message history.
2. Resolve follow-up pronouns (e.g., 'them', 'it', 'that feature', 'how many can I create?') into fully explicit queries using prior conversation context.
3. Classify the query intent into one of:
   - 'single_hop': Simple factual lookup in one spec/document.
   - 'multi_hop': Question needing evidence from multiple documents or release notes.
   - 'follow_up': Question continuing a previous turn.
   - 'conflicting_evidence': Question about features or limits that changed across versions/dates.
   - 'unsupported_evidence': Question asking about external or third-party topics (e.g. blockchain, AWS services not in Kestrel).
4. Decompose the resolved question into 1 to 3 targeted search subqueries.
5. DO NOT attempt to answer the user's question yourself. Your job is ONLY to plan retrieval.
"""

def run_router_agent(state: ResearchState, llm_provider: BaseLLMProvider) -> Dict[str, Any]:
    question = state["original_question"]
    history = state.get("messages", [])

    history_str = ""
    if history:
        history_lines = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history[-6:]]
        history_str = "\n".join(history_lines)

    prompt = f"CONVERSATION HISTORY:\n{history_str if history_str else 'None'}\n\nUSER QUESTION: {question}"

    try:
        plan: RouterOutputSchema = llm_provider.generate_structured(
            prompt=prompt,
            schema_class=RouterOutputSchema,
            system_prompt=ROUTER_SYSTEM_PROMPT,
            temperature=0.0
        )
        planner_dict = plan.model_dump()
    except Exception as e:
        logger.error(f"Router Agent fallback heuristic triggered: {e}")
        planner_dict = {
            "intent": "single_hop",
            "subqueries": [question],
            "requires_multi_hop": False,
            "resolved_question": question,
            "reasoning": "Default single query fallback."
        }

    steps = state.get("agent_steps", [])
    steps.append("✓ Router Planned Retrieval")

    return {
        "intent": planner_dict.get("intent", "single_hop"),
        "subqueries": planner_dict.get("subqueries", [question]),
        "planner_output": planner_dict,
        "agent_steps": steps
    }
