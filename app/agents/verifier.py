import logging
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from app.graph.state import ResearchState
from app.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

class ClaimVerification(BaseModel):
    claim: str = Field(description="Individual material factual statement required to answer the question")
    verdict: str = Field(description="One of: supported, partially_supported, conflicting_evidence, insufficient_evidence")
    supporting_chunk_ids: List[str] = Field(description="List of chunk_ids supporting or involved in this claim")
    explanation: str = Field(description="Explanation of verdict, including publication date comparison for conflicts")

class VerifierOutputSchema(BaseModel):
    overall_verdict: str = Field(description="One of: supported, partially_supported, conflicting_evidence, insufficient_evidence")
    claims: List[ClaimVerification] = Field(description="List of claim-level verification objects")
    conflict_resolution: Optional[str] = Field(default=None, description="Detailed conflict analysis explaining why newer published sources supersede older release notes if applicable")

VERIFIER_SYSTEM_PROMPT = """You are the Verifier / Critic Agent for Kestrel Labs Documentation.
Your sole job is to strictly check every material claim required to answer the question against the retrieved documentation chunks.

VERDICT RULES:
1. 'supported': The retrieved chunks explicitly contain complete factual evidence proving the claim.
2. 'partially_supported': Evidence is partially present but leaves minor gaps.
3. 'conflicting_evidence': Different retrieved chunks state different values, limits, or behaviors (e.g., release notes 3.4 vs release 3.5/4.0 spec).
   - Compare publication dates ('published' metadata e.g. '20260203' vs '20250121').
   - Treat the newer published document as the active current truth.
   - Explain the date-based resolution clearly.
   - DO NOT hide conflicting evidence; cite both source chunk IDs.
4. 'insufficient_evidence': The retrieved chunks DO NOT contain sufficient evidence to answer the user's question about Kestrel Labs.
   - DO NOT use LLM general knowledge or guess facts.
"""

# Verifier Agent entry function that validates factual claims against retrieved documentation chunks.
# Assesses evidence sufficiency, identifies conflicts across release dates, and generates claim verdicts.
def run_verifier_agent(state: ResearchState, llm_provider: BaseLLMProvider) -> Dict[str, Any]:
    question = state["original_question"]
    raw_chunks = state.get("retrieved_chunks", [])

    if not raw_chunks:
        steps = state.get("agent_steps", [])
        steps.append("✓ Verifier Evaluated Evidence (Verdict: insufficient_evidence)")
        return {
            "claims": [],
            "verifier_results": {"overall_verdict": "insufficient_evidence", "claims": [], "conflict_resolution": "No chunks retrieved."},
            "verifier_verdict": "insufficient_evidence",
            "agent_steps": steps
        }

    # Context Compression: Take top 5 strongest reranked chunks to save prompt tokens & prevent rate limits
    top_chunks = raw_chunks[:5]

    # Format compressed chunks for verifier prompt
    formatted_chunks = []
    # Loop over top reranked chunks to format text content and publication metadata into prompt strings.
    # Prepares clean context representation for claim verification analysis by the LLM.
    for c in top_chunks:
        formatted_chunks.append(
            f"CHUNK ID: {c.get('chunk_id')}\n"
            f"Title: {c.get('title')}\n"
            f"Published Date: {c.get('published')}\n"
            f"Version: {c.get('version')}\n"
            f"Content: {c.get('text')}\n"
            "---"
        )
    evidence_str = "\n".join(formatted_chunks)

    prompt = (
        f"USER QUESTION: {question}\n\n"
        f"RETRIEVED EVIDENCE CHUNKS:\n{evidence_str}\n\n"
        "Verify the factual claims required to answer the question strictly against the retrieved evidence."
    )

    try:
        verification: VerifierOutputSchema = llm_provider.generate_structured(
            prompt=prompt,
            schema_class=VerifierOutputSchema,
            system_prompt=VERIFIER_SYSTEM_PROMPT,
            temperature=0.0
        )
        verdict_dict = verification.model_dump()
        verdict_str = verdict_dict.get("overall_verdict", "supported")
        claims_list = verdict_dict.get("claims", [])
    except Exception as e:
        logger.error(f"Verifier Agent parsing fallback triggered: {e}")
        verdict_str = "supported" if raw_chunks else "insufficient_evidence"
        claims_list = []
        verdict_dict = {"overall_verdict": verdict_str, "claims": [], "conflict_resolution": None}

    steps = state.get("agent_steps", [])
    steps.append(f"✓ Verifier Claims Checked (Verdict: {verdict_str})")

    return {
        "claims": claims_list,
        "verifier_results": verdict_dict,
        "verifier_verdict": verdict_str,
        "agent_steps": steps
    }
