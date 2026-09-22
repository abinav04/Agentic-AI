import logging
from typing import List, Union
from sentence_transformers import SentenceTransformer
from chromadb import EmbeddingFunction
from app.config import settings

logger = logging.getLogger(__name__)

# Class to manage loading and inference for local SentenceTransformer text embedding models.
# Provides helper methods to convert single strings or lists of text into dense vector embeddings.
class LocalEmbeddingManager:
    """Manages local SentenceTransformer embedding model."""

    # Initializes the embedding manager with a specified model name.
    # Sets up lazy initialization for the heavy SentenceTransformer model instance.
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self._model = None

    # Lazy-loads and caches the SentenceTransformer model on first usage.
    # Avoids expensive model load overhead during initial module imports.
    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Initializing local embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # Generates dense vector embeddings for a list of document strings.
    # Encodes texts in batch mode and returns vectors as Python float lists.
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        model = self._get_model()
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()

    # Generates a single vector embedding for an input user query string.
    # Uses the local SentenceTransformer model and returns the vector list.
    def embed_query(self, text: str) -> List[float]:
        model = self._get_model()
        embedding = model.encode(text, show_progress_bar=False, convert_to_numpy=True)
        return embedding.tolist()

# Wrapper adapter class making LocalEmbeddingManager compatible with ChromaDB's EmbeddingFunction interface.
# Enables ChromaDB to call embedding logic automatically during document indexing and querying.
class LocalChromaEmbeddingFunction(EmbeddingFunction):
    """Chroma-compatible embedding function using local sentence-transformers."""

    # Initializes the Chroma embedding adapter with a reference to the local embedding manager.
    # Connects ChromaDB calls directly to the local model instance.
    def __init__(self, embedding_manager: LocalEmbeddingManager):
        self.embedding_manager = embedding_manager

    # Returns the identifier name of the custom embedding function implementation.
    # Required by ChromaDB interface for tracking embedding function types.
    def name(self) -> str:
        return "local_sentence_transformer"

    # Callable interface entry point invoked by ChromaDB during query and document indexing.
    # Handles both single string queries and batch lists of text documents seamlessly.
    def __call__(self, input: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(input, str):
            return [self.embedding_manager.embed_query(input)]
        return self.embedding_manager.embed_documents(input)

embedding_manager = LocalEmbeddingManager()
chroma_embedding_function = LocalChromaEmbeddingFunction(embedding_manager)
