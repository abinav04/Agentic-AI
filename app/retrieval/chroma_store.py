import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import settings
from app.retrieval.embeddings import chroma_embedding_function

logger = logging.getLogger(__name__)

COLLECTION_NAME = "kestrel_corpus"

# Class to encapsulate ChromaDB client operations and vector collection management.
# Provides low-level interfaces for adding documents, querying vectors, and checking counts.
class ChromaStoreManager:
    """Manages persistent Chroma vector store."""

    # Initializes the persistent Chroma DB client at the configured directory path.
    # Prepares lazy loading container for the vector collection instance.
    def __init__(self, persist_directory: Optional[str] = None):
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIRECTORY
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self._collection = None

    # Lazy-loads or creates the target Chroma vector collection using cosine similarity.
    # Attaches the custom embedding function for automated text vectorization.
    def get_collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=chroma_embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    # Returns the total count of document chunks currently indexed in the collection.
    # Used for status verification and skipping redundant ingestion runs.
    def count(self) -> int:
        collection = self.get_collection()
        return collection.count()

    # Batch inserts document text chunks and associated metadata into the vector database.
    # Processes chunks in small batches to optimize memory usage on constrained environments.
    def add_chunks(self, chunks: List[Dict[str, Any]]):
        collection = self.get_collection()
        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = []
        # Iterate over raw chunk dicts to format key metadata attributes into clean dictionaries.
        # Prepares structured metadata records for storage alongside vector embeddings.
        for chunk in chunks:
            metadatas.append({
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "title": chunk["title"],
                "category": chunk["category"],
                "owner": chunk["owner"],
                "source_url": chunk["source_url"],
                "published": chunk["published"],
                "version": chunk["version"]
            })

        # Add in small batches of 16 to keep memory usage under 150MB on 512MB RAM cloud tiers
        batch_size = 16
        # Loop through chunk data in fixed batch sizes to submit entries to ChromaDB safely.
        # Prevents memory spikes during heavy vector generation and bulk indexing operations.
        for i in range(0, len(chunks), batch_size):
            collection.add(
                ids=ids[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )
        logger.info(f"Successfully indexed {len(chunks)} chunks into Chroma collection '{COLLECTION_NAME}'")

    # Queries vector collection using similarity search with optional metadata filter constraints.
    # Returns ranked matching document chunks with calculated cosine similarity scores.
    def query(
        self,
        query_text: str,
        top_k: int = 8,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        collection = self.get_collection()
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where_filter
        )

        retrieved_chunks = []
        if results and results.get("documents") and len(results["documents"]) > 0:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else []
            ids = results["ids"][0] if results.get("ids") else []
            distances = results["distances"][0] if results.get("distances") else []

            # Iterate through raw Chroma query output arrays to assemble structured chunk dictionaries.
            # Converts raw cosine distance metrics into normalized similarity scores.
            for i in range(len(docs)):
                chunk_meta = metas[i] if i < len(metas) else {}
                dist = distances[i] if i < len(distances) else 0.0
                similarity_score = max(0.0, 1.0 - dist)  # Cosine distance to similarity

                retrieved_chunks.append({
                    "chunk_id": ids[i] if i < len(ids) else chunk_meta.get("chunk_id", ""),
                    "doc_id": chunk_meta.get("doc_id", ""),
                    "title": chunk_meta.get("title", ""),
                    "category": chunk_meta.get("category", ""),
                    "owner": chunk_meta.get("owner", ""),
                    "source_url": chunk_meta.get("source_url", ""),
                    "published": str(chunk_meta.get("published", "")),
                    "version": str(chunk_meta.get("version", "")),
                    "text": docs[i],
                    "similarity_score": round(similarity_score, 4)
                })

        return retrieved_chunks

chroma_store = ChromaStoreManager()
