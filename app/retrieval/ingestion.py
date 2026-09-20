import json
import os
import logging
from typing import List, Dict, Any
from app.config import settings
from app.retrieval.chroma_store import chroma_store

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = [
    "chunk_id", "doc_id", "title", "category",
    "owner", "source_url", "published", "version", "text"
]

def load_corpus_jsonl(file_path: str = None) -> List[Dict[str, Any]]:
    path = file_path or settings.CORPUS_FILE_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Corpus file not found at: {path}")

    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            # Validate required fields
            missing = [field for field in REQUIRED_FIELDS if field not in data]
            if missing:
                raise ValueError(f"Line {idx} in {path} is missing required fields: {missing}")
            # Ensure published and version are stringified for metadata safety
            data["published"] = str(data["published"])
            data["version"] = str(data["version"])
            chunks.append(data)

    logger.info(f"Loaded and validated {len(chunks)} chunks from {path}")
    return chunks

def run_ingestion(force_reindex: bool = False) -> int:
    current_count = chroma_store.count()
    if not force_reindex and current_count == 154:
        logger.info("Chroma index already contains 154 chunks. Skipping re-ingestion.")
        return current_count

    logger.info("Starting corpus ingestion into Chroma vector DB...")
    chunks = load_corpus_jsonl()
    chroma_store.add_chunks(chunks)
    new_count = chroma_store.count()
    logger.info(f"Ingestion finished. Total chunks in vector store: {new_count}")
    return new_count

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_ingestion()
