import json
import logging
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool
from app.retrieval.chroma_store import chroma_store
from app.retrieval.reranker import rerank_chunks

logger = logging.getLogger(__name__)

# Core internal retrieval function that executes ChromaDB vector search and applies reranking.
# Queries Chroma for candidate pools and returns top-k reranked context chunks.
def search_corpus_func(query: str, top_k: int = 8, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Internal search corpus implementation."""
    raw_results = chroma_store.query(
        query_text=query,
        top_k=top_k * 2,  # retrieve candidate pool before reranking
        where_filter=filters
    )
    reranked = rerank_chunks(query=query, chunks=raw_results, top_k=top_k)
    return reranked

# LangChain tool wrapper function enabling agents to execute corpus vector search queries.
# Parses optional JSON metadata filters and formats reranked chunk results as JSON strings.
@tool
def search_corpus(query: str, top_k: int = 8, filters: Optional[str] = None) -> str:
    """Search the Kestrel internal documentation corpus for relevant evidence chunks.

    Args:
        query: The search query string.
        top_k: Number of ranked chunks to return (default 8).
        filters: Optional filter string or dict.

    Returns:
        JSON string containing array of ranked evidence chunks with chunk_id, title, text, published, version, etc.
    """
    filter_dict = None
    if filters:
        if isinstance(filters, str):
            try:
                filter_dict = json.loads(filters)
            except Exception:
                filter_dict = None
        elif isinstance(filters, dict):
            filter_dict = filters

    chunks = search_corpus_func(query=query, top_k=top_k, filters=filter_dict)
    return json.dumps(chunks, indent=2)
