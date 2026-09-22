import logging
from typing import Dict, Any, List
from app.graph.state import ResearchState
from app.llm.base import BaseLLMProvider
from app.utils.citations import format_option_a_citations, build_citation_sources_list

logger = logging.getLogger(__name__)

SYNTHESIZER_SYSTEM_PROMPT = """You are the Synthesizer Agent for Kestrel Labs Internal Technical Documentation.
Your responsibility is to synthesize a final, clear, helpful answer for the user based strictly on verified retrieved evidence.

STRICT GROUNDING RULES:
1. Rely ONLY on the provided verified evidence. NEVER use general knowledge for Kestrel internal facts.
2. If the verifier verdict is 'insufficient_evidence' or the retrieved chunks do not establish an answer, state:
   "Insufficient evidence: the provided Kestrel documentation does not establish whether [topic]."
3. If evidence conflicts, explicitly explain the conflict (e.g., "In release notes 3.4 published 2025-01-21..., but as of release 4.0 published 2025-10-07...").
4. Every factual claim must be grounded in the retrieved chunks.
5. Keep answers concise, direct, and professional.
6. Do NOT include internal chain-of-thought or raw JSON schemas in the response text.
7. Append Option A format citations at the very end of your response:
   Sources:
   - chunk_id — Title
"""

# Synthesizer Agent entry function that generates grounded final answers using verified retrieved context.
# Appends standardized citation blocks and handles insufficient evidence cases cleanly.
def run_synthesizer_agent(state: ResearchState, llm_provider: BaseLLMProvider) -> Dict[str, Any]:
    question = state["original_question"]
    verdict = state.get("verifier_verdict", "supported")
    raw_chunks = state.get("retrieved_chunks", [])
    verifier_results = state.get("verifier_results", {})

    # If insufficient evidence, handle directly without hallucinating
    if verdict == "insufficient_evidence" or not raw_chunks:
        final_ans = f"Insufficient evidence: the provided Kestrel documentation does not establish whether '{question}' is supported."
        steps = state.get("agent_steps", [])
        steps.append("✓ Synthesizer Answer Formulated (Insufficient Evidence)")
        return {
            "final_answer": final_ans,
            "citations": [],
            "formatted_citations": "",
            "agent_steps": steps
        }

    # Context Compression: Use top 5 reranked evidence chunks to minimize prompt token count
    top_chunks = raw_chunks[:5]

    # Format compressed evidence chunks for prompt
    formatted_chunks = []
    # Loop through top reranked chunks to construct evidence strings containing chunk IDs and titles.
    # Formats evidence context cleanly for the final answer synthesis prompt.
    for c in top_chunks:
        formatted_chunks.append(
            f"[{c.get('chunk_id')}] (Title: {c.get('title')}, Date: {c.get('published')}): {c.get('text')}"
        )
    evidence_str = "\n\n".join(formatted_chunks)

    conflict_res = verifier_results.get("conflict_resolution")
    conflict_prompt = f"\nVERIFIER CONFLICT RESOLUTION NOTE:\n{conflict_res}\n" if conflict_res else ""

    prompt = (
        f"USER QUESTION: {question}\n\n"
        f"VERIFIED EVIDENCE CHUNKS:\n{evidence_str}\n"
        f"{conflict_prompt}\n"
        "Synthesize the concise answer based strictly on the evidence above."
    )

    try:
        raw_answer = llm_provider.generate(
            prompt=prompt,
            system_prompt=SYNTHESIZER_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=1000
        )
    except Exception as e:
        logger.error(f"Synthesizer generation exception: {type(e).__name__}: {e}", exc_info=True)
        raw_answer = "I am currently unable to synthesize the answer due to temporary rate limits on both LLM providers. Please try asking again in a few seconds."

    # Build Option A citations block from top used chunks
    citations_list = build_citation_sources_list(top_chunks)
    formatted_cit = format_option_a_citations(top_chunks)

    # Append citations if not already appended by model
    final_answer = raw_answer.strip()
    if "Sources:" not in final_answer and formatted_cit:
        final_answer = f"{final_answer}\n\n{formatted_cit}"

    steps = state.get("agent_steps", [])
    steps.append("✓ Synthesizer Completed Final Answer")

    return {
        "final_answer": final_answer,
        "citations": citations_list,
        "formatted_citations": formatted_cit,
        "agent_steps": steps
    }
