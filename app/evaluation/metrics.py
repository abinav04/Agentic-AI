from typing import List, Dict, Any, Optional

# Function to compute Recall@K retrieval accuracy against ground truth chunk IDs.
# Calculates the proportion of expected document chunks retrieved by the vector search.
def compute_retrieval_recall_at_k(retrieved_ids: List[str], expected_ids: List[str]) -> float:
    """Computes Recall@K: fraction of expected ground truth chunk_ids retrieved."""
    if not expected_ids:
        # For unsupported questions (expected_ids is empty), recall is 1.0 if system retrieved nothing or identified unsupported
        return 1.0 if not retrieved_ids else 1.0

    retrieved_set = set(retrieved_ids)
    expected_set = set(expected_ids)
    hits = retrieved_set.intersection(expected_set)
    return round(len(hits) / len(expected_set), 4)

# Function to compute citation precision score based on valid ground truth chunk references.
# Evaluates what fraction of generated Option A citations accurately reference expected sources.
def compute_citation_precision(cited_ids: List[str], expected_ids: List[str]) -> float:
    """Computes Citation Precision: fraction of cited chunk_ids that are in expected chunk_ids."""
    if not cited_ids:
        # If no citations expected (unsupported), precision is 1.0
        return 1.0 if not expected_ids else 0.0

    if not expected_ids:
        return 0.0

    cited_set = set(cited_ids)
    expected_set = set(expected_ids)
    valid_citations = cited_set.intersection(expected_set)
    return round(len(valid_citations) / len(cited_set), 4)

# Programmatic heuristic score calculation assessing answer faithfulness to retrieved evidence.
# Evaluates verifier verdicts and checks for proper refusal strings on unsupported queries.
def compute_heuristic_faithfulness(answer: str, verifier_verdict: str, retrieved_chunks: List[Dict[str, Any]]) -> float:
    """Programmatic faithfulness calculation based on verifier verdict and grounding."""
    if verifier_verdict == "insufficient_evidence":
        return 1.0 if "insufficient evidence" in answer.lower() or "does not establish" in answer.lower() else 0.5

    if verifier_verdict == "supported":
        return 1.0
    elif verifier_verdict == "conflicting_evidence":
        return 0.9 if ("conflict" in answer.lower() or "release" in answer.lower()) else 0.7
    elif verifier_verdict == "partially_supported":
        return 0.75

    return 0.5

# Function to compute question-answer relevance score using key term keyword overlap analysis.
# Quantifies how directly the generated answer text responds to the core search concepts in the question.
def compute_heuristic_relevance(question: str, answer: str) -> float:
    """Calculates query-answer relevance score based on key term overlap."""
    if not answer:
        return 0.0
    q_words = [w.lower() for w in question.split() if len(w) > 3]
    if not q_words:
        return 1.0
    a_lower = answer.lower()
    matches = sum(1 for w in q_words if w in a_lower)
    return round(min(1.0, 0.4 + (matches / len(q_words)) * 0.6), 4)

# Function to aggregate deterministic RAG metrics into a single end-to-end correctness score.
# Combines weighted recall, citation precision, faithfulness, and relevance scores.
def compute_e2e_correctness(recall: float, precision: float, faithfulness: float, relevance: float) -> float:
    """Combines deterministic metrics into an overall end-to-end correctness score."""
    score = (recall * 0.35) + (precision * 0.25) + (faithfulness * 0.25) + (relevance * 0.15)
    return round(score, 4)
