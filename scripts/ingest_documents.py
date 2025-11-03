"""
Document ingestion script.
Processes documents from the documents/ directory and stores them in ChromaDB locally.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_loader import DocumentLoader
from src.embeddings import SentenceTransformerEmbeddings
from src.vector_store import ChromaDBVectorStore

# Load environment variables
load_dotenv()


def main():
    """Main ingestion function."""
    print("=" * 60)
    print("Document Ingestion Script")
    print("=" * 60)
    print()
    
    # Load configuration
    documents_dir = os.getenv("DOCUMENTS_DIR", "./documents")
    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "startup-knowledge-base")
    persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Step 1: Load documents
    print("Step 1: Loading documents...")
    loader = DocumentLoader(documents_dir=documents_dir)
    documents = loader.load_documents()
    
    if not documents:
        print("No documents found. Please add documents to the 'documents/' directory.")
        return
    
    print(f"✓ Loaded {len(documents)} document(s)")
    
    # Step 2: Chunk documents
    print(f"\nStep 2: Chunking documents (size: {chunk_size}, overlap: {chunk_overlap})...")
    chunks = loader.chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    print(f"✓ Created {len(chunks)} chunks")
    
    # Step 3: Initialize embeddings
    print(f"\nStep 3: Initializing embedding model ({embedding_model})...")
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
    print("✓ Embedding model ready")
    
    # Step 4: Initialize ChromaDB
    print(f"\nStep 4: Initializing ChromaDB (collection: {collection_name})...")
    vector_store = ChromaDBVectorStore(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    print("✓ ChromaDB initialized (100% local storage)")
    
    # Step 5: Upload documents
    print(f"\nStep 5: Adding {len(chunks)} chunks to ChromaDB...")
    vector_store.add_documents(chunks)
    
    print("\n" + "=" * 60)
    print("✓ Document ingestion completed successfully!")
    print("=" * 60)
    print(f"\nDocuments are now stored locally in: {persist_directory}")
    print(f"You can now use the chatbot to query your documents.")
    print(f"Run: python chatbot_cli.py")
    print(f"Or: streamlit run app.py")


if __name__ == "__main__":
    main()
