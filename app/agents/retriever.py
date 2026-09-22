import logging
import json
from typing import Dict, Any, List
from app.graph.state import ResearchState
from app.tools.search_corpus import search_corpus_func

logger = logging.getLogger(__name__)

# Retriever Agent entry point that runs vector search queries using the corpus tool.
# Collects, deduplicates, and compiles relevant context chunks for verification and answer synthesis.
def run_retriever_agent(state: ResearchState) -> Dict[str, Any]:
    subqueries = state.get("subqueries", [])
    if not subqueries:
        subqueries = [state.get("original_question", "")]

    retrieved_chunks = []
    seen_ids = set()
    queries_run = []

    # Loop over planned search subqueries to execute corpus retrieval operations.
    # Runs hybrid vector search for each subquery and aggregates document matches.
    for subq in subqueries:
        if not subq.strip():
            continue
        queries_run.append(subq)
        results = search_corpus_func(query=subq, top_k=6)
        # Inner loop to deduplicate chunks across multiple subquery vector search results.
        # Guarantees each chunk ID appears at most once in the combined retrieved set.
        for chunk in results:
            cid = chunk.get("chunk_id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                retrieved_chunks.append(chunk)

    chunk_ids = [c["chunk_id"] for c in retrieved_chunks if "chunk_id" in c]

    iteration = state.get("retrieval_iteration", 0) + 1
    steps = state.get("agent_steps", [])
    steps.append(f"✓ Retriever Executed search_corpus Tool ({len(chunk_ids)} chunks)")

    return {
        "retrieved_chunks": retrieved_chunks,
        "retrieved_chunk_ids": chunk_ids,
        "retrieval_queries": queries_run,
        "retrieval_iteration": iteration,
        "agent_steps": steps
    }
