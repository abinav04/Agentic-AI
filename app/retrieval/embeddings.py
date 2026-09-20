import logging
from typing import List, Union
from sentence_transformers import SentenceTransformer
from chromadb import EmbeddingFunction
from app.config import settings

logger = logging.getLogger(__name__)

class LocalEmbeddingManager:
    """Manages local SentenceTransformer embedding model."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self._model = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Initializing local embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        model = self._get_model()
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        model = self._get_model()
        embedding = model.encode(text, show_progress_bar=False, convert_to_numpy=True)
        return embedding.tolist()

class LocalChromaEmbeddingFunction(EmbeddingFunction):
    """Chroma-compatible embedding function using local sentence-transformers."""

    def __init__(self, embedding_manager: LocalEmbeddingManager):
        self.embedding_manager = embedding_manager

    def name(self) -> str:
        return "local_sentence_transformer"

    def __call__(self, input: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(input, str):
            return [self.embedding_manager.embed_query(input)]
        return self.embedding_manager.embed_documents(input)

embedding_manager = LocalEmbeddingManager()
chroma_embedding_function = LocalChromaEmbeddingFunction(embedding_manager)
