from typing import List, Dict, Any, Set

# Function to convert retrieved chunk objects into a formatted Option A citation text block.
# Generates markdown bullet points of unique chunk IDs paired with document titles.
def format_option_a_citations(chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved chunks into Option A citation block:

    Sources:
    - chunk_id — title
    """
    if not chunks:
        return ""

    seen_ids: Set[str] = set()
    citation_lines: List[str] = []

    # Loop over chunk items to collect unique chunk IDs and construct clean citation strings.
    # Ignores duplicate chunks to avoid repeating citations in the formatted list.
    for chunk in chunks:
        cid = chunk.get("chunk_id", "").strip()
        title = chunk.get("title", "Document").strip()
        if cid and cid not in seen_ids:
            seen_ids.add(cid)
            citation_lines.append(f"- {cid} — {title}")

    if not citation_lines:
        return ""

    return "Sources:\n" + "\n".join(citation_lines)

# Function to build a structured list of unique citation dictionaries from retrieved chunks.
# Returns list of dicts containing chunk_id and title fields for API response metadata.
def build_citation_sources_list(chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Returns a structured list of unique citations."""
    seen_ids: Set[str] = set()
    citations = []
    # Loop over chunks to assemble unique citation objects without duplicate entries.
    # Populates structured list used in API payloads and UI source cards.
    for chunk in chunks:
        cid = chunk.get("chunk_id", "").strip()
        title = chunk.get("title", "Document").strip()
        if cid and cid not in seen_ids:
            seen_ids.add(cid)
            citations.append({"chunk_id": cid, "title": title})
    return citations
