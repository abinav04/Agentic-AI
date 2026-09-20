from typing import List, Dict, Any, Set

def format_option_a_citations(chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved chunks into Option A citation block:

    Sources:
    - chunk_id — title
    """
    if not chunks:
        return ""

    seen_ids: Set[str] = set()
    citation_lines: List[str] = []

    for chunk in chunks:
        cid = chunk.get("chunk_id", "").strip()
        title = chunk.get("title", "Document").strip()
        if cid and cid not in seen_ids:
            seen_ids.add(cid)
            citation_lines.append(f"- {cid} — {title}")

    if not citation_lines:
        return ""

    return "Sources:\n" + "\n".join(citation_lines)

def build_citation_sources_list(chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Returns a structured list of unique citations."""
    seen_ids: Set[str] = set()
    citations = []
    for chunk in chunks:
        cid = chunk.get("chunk_id", "").strip()
        title = chunk.get("title", "Document").strip()
        if cid and cid not in seen_ids:
            seen_ids.add(cid)
            citations.append({"chunk_id": cid, "title": title})
    return citations
