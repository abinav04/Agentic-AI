from fastapi import APIRouter
from app.retrieval.chroma_store import chroma_store
from app.config import settings

router = APIRouter()

@router.get("/health")
def get_health_status():
    """Health check endpoint returning system status and vector DB chunk count."""
    try:
        count = chroma_store.count()
        status = "healthy" if count == 154 else "ingestion_needed"
    except Exception as e:
        count = 0
        status = f"unhealthy: {str(e)}"

    return {
        "status": status,
        "chroma_chunks_count": count,
        "expected_chunks_count": 154,
        "primary_provider": settings.LLM_PRIMARY_PROVIDER,
        "fallback_enabled": settings.LLM_FALLBACK_ENABLED,
        "embedding_model": settings.EMBEDDING_MODEL_NAME
    }
