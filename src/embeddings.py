"""
Embedding generation for documents.
Uses sentence-transformers for creating embeddings.
"""

from typing import List
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings


class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper for sentence-transformers to work with LangChain."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the embedding model."""
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("Embedding model loaded successfully")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self.model.encode(text)
        return embedding.tolist()

