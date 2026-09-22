import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Helper function to extract and convert publication date strings into integer YYYYMMDD numbers.
# Provides a clean numeric score representation used for calculating doc recency bonuses.
def parse_published_date(date_str: str) -> int:
    """Converts published date string e.g. '20260203' to integer 20260203 for recency comparisons."""
    try:
        clean = re.sub(r"[^\d]", "", str(date_str))
        if len(clean) >= 8:
            return int(clean[:8])
        return 20200101
    except Exception:
        return 20200101

# Function to rerank vector search retrieved chunks based on vector similarity, exact keywords, and date recency.
# Filters duplicates and returns top-k highest scoring candidate context chunks.
def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 8,
    date_weight: float = 0.15
) -> List[Dict[str, Any]]:
    """Reranks retrieved chunks using semantic similarity, keyword matching, and publication date recency."""
    if not chunks:
        return []

    # Deduplicate by chunk_id
    seen_ids = set()
    unique_chunks = []
    # Loop over retrieved chunks to deduplicate items by unique chunk_id before scoring.
    # Prevents duplicate documents from occupying top retrieval slots in context windows.
    for c in chunks:
        cid = c.get("chunk_id")
        if cid and cid not in seen_ids:
            seen_ids.add(cid)
            unique_chunks.append(c)

    query_words = set(re.findall(r"\w+", query.lower()))

    scored_chunks = []
    # Loop over unique candidate chunks to calculate combined hybrid relevance and recency scores.
    # Combines vector cosine score with keyword matching bonuses and publication date weighting.
    for chunk in unique_chunks:
        base_score = chunk.get("similarity_score", 0.5)
        text_lower = chunk.get("text", "").lower()
        title_lower = chunk.get("title", "").lower()

        # Keyword match bonus
        matches = sum(1 for word in query_words if len(word) > 3 and (word in text_lower or word in title_lower))
        keyword_bonus = min(0.2, matches * 0.04)

        # Date recency bonus (normalized score based on YYYYMMDD)
        pub_int = parse_published_date(chunk.get("published", ""))
        # 20260203 vs 20250121 delta is ~10000. Give small recency advantage
        recency_bonus = (pub_int - 20250101) / 20000.0 * date_weight
        recency_bonus = max(0.0, min(date_weight, recency_bonus))

        final_score = base_score + keyword_bonus + recency_bonus
        chunk_copy = dict(chunk)
        chunk_copy["rerank_score"] = round(final_score, 4)
        scored_chunks.append(chunk_copy)

    # Sort descending by rerank_score
    scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored_chunks[:top_k]
