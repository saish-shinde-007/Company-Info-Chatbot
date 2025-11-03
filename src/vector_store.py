"""
ChromaDB vector store integration for semantic search.
Handles storing and retrieving document embeddings locally.
100% local - no cloud dependencies!
"""

import os
from typing import List, Optional, Union
import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field


class LangChainEmbeddingFunction:
    """EmbeddingFunction wrapper for LangChain embedding objects."""
    
    def __init__(self, embedding_function):
        """
        Initialize with a LangChain-compatible embedding function.
        
        Args:
            embedding_function: Object with embed_query() and embed_documents() methods
        """
        self.embedding_function = embedding_function
    
    def __call__(self, input: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for input.
        
        Args:
            input: Either a single string or a list of strings
            
        Returns:
            For single string: list of floats (embedding vector)
            For list of strings: list of lists of floats (list of embedding vectors)
        """
        if isinstance(input, str):
            # Single query
            return self.embedding_function.embed_query(input)
        else:
            # List of texts
            return self.embedding_function.embed_documents(input)


class ChromaDBVectorStore:
    """ChromaDB vector store wrapper for LangChain - 100% local."""
    
    def __init__(
        self,
        collection_name: str,
        embedding_function,
        persist_directory: str = "./chroma_db"
    ):
        """
        Initialize ChromaDB connection and collection.
        
        Args:
            collection_name: Name of the ChromaDB collection
            embedding_function: Function to generate embeddings
            persist_directory: Directory to persist the database (local storage)
        """
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        
        # Create proper EmbeddingFunction for ChromaDB
        self.chroma_embedding_function = LangChainEmbeddingFunction(embedding_function)
        
        # Create persistent ChromaDB client
        print(f"Initializing ChromaDB (local storage: {persist_directory})...")
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.chroma_embedding_function
            )
            print(f"✓ Using existing ChromaDB collection: {collection_name}")
            print(f"  Current document count: {self.collection.count()}")
        except Exception as e:
            # If collection exists but has incompatible embedding function, delete and recreate
            try:
                existing_collections = [col.name for col in self.client.list_collections()]
                if collection_name in existing_collections:
                    print(f"⚠ Collection exists with incompatible embedding function.")
                    print(f"  Deleting old collection '{collection_name}' and creating a new one...")
                    self.client.delete_collection(name=collection_name)
            except Exception:
                pass  # Ignore errors during cleanup
            
            print(f"Creating new ChromaDB collection: {collection_name}")
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.chroma_embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"✓ Collection {collection_name} created successfully")
    
    def add_documents(self, documents: List[Document], batch_size: int = 100):
        """Add documents to the vector store."""
        print(f"Adding {len(documents)} documents to ChromaDB...")
        
        # Prepare documents for ChromaDB
        texts = []
        metadatas = []
        ids = []
        
        for i, doc in enumerate(documents):
            texts.append(doc.page_content)
            metadatas.append({
                'source': str(doc.metadata.get('source', 'unknown')),
                'file_name': doc.metadata.get('file_name', 'unknown'),
                'file_type': doc.metadata.get('file_type', 'unknown'),
            })
            ids.append(f"doc_{i}_{hash(doc.page_content) % 1000000}")
        
        # Add to ChromaDB in batches
        total_batches = (len(texts) - 1) // batch_size + 1
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            
            self.collection.add(
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            print(f"Uploaded batch {i//batch_size + 1}/{total_batches}")
        
        print(f"✓ Successfully added {len(texts)} documents to ChromaDB")
        print(f"  Total documents in collection: {self.collection.count()}")
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """Search for similar documents."""
        # Build where clause for filtering if provided
        where = None
        if filter:
            where = {}
            for key, value in filter.items():
                where[key] = {"$eq": value}
        
        # Search ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where
        )
        
        # Convert to Document objects
        documents = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i, doc_text in enumerate(results['documents'][0]):
                metadata = {
                    'source': results['metadatas'][0][i].get('source', 'unknown'),
                    'file_name': results['metadatas'][0][i].get('file_name', 'unknown'),
                    'file_type': results['metadatas'][0][i].get('file_type', 'unknown'),
                }
                
                # Add distance as score (ChromaDB returns distances)
                if results.get('distances') and len(results['distances'][0]) > i:
                    # Convert distance to similarity score (cosine distance -> similarity)
                    distance = results['distances'][0][i]
                    similarity = 1 - distance  # Cosine distance to similarity
                    metadata['score'] = similarity
                
                doc = Document(
                    page_content=doc_text,
                    metadata=metadata
                )
                documents.append(doc)
        
        return documents
    
    def as_retriever(self, k: int = 5) -> 'ChromaDBRetriever':
        """Get a retriever interface."""
        return ChromaDBRetriever(vector_store=self, k=k)
    
    def delete_collection(self):
        """Delete the collection (useful for resetting)."""
        self.client.delete_collection(name=self.collection_name)
        print(f"✓ Deleted collection: {self.collection_name}")


class ChromaDBRetriever(BaseRetriever):
    """Retriever interface for ChromaDB."""
    
    vector_store: ChromaDBVectorStore = Field(..., description="The ChromaDB vector store")
    k: int = Field(default=5, description="Number of documents to retrieve")
    
    def __init__(self, vector_store: ChromaDBVectorStore, k: int = 5, **kwargs):
        super().__init__(vector_store=vector_store, k=k, **kwargs)
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query."""
        return self.vector_store.similarity_search(query, k=self.k)
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version (not implemented yet)."""
        return self._get_relevant_documents(query)
