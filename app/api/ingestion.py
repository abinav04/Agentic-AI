from fastapi import APIRouter
from pydantic import BaseModel
from app.retrieval.ingestion import run_ingestion

router = APIRouter()

class IngestionRequest(BaseModel):
    force_reindex: bool = False

# FastAPI endpoint handler to manually trigger corpus data ingestion into ChromaDB.
# Reads corpus records, generates embeddings, and returns total document chunks indexed.
@router.post("/ingest")
def trigger_ingestion(request: IngestionRequest = IngestionRequest()):
    """Triggers corpus ingestion into Chroma DB."""
    total_chunks = run_ingestion(force_reindex=request.force_reindex)
    return {
        "status": "success",
        "total_chunks_indexed": total_chunks
    }
