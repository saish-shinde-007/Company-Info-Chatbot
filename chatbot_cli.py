"""
CLI interface for the Startup Knowledge Q&A Chatbot.
"""

import os
from dotenv import load_dotenv
from src.llm_setup import MPTInstructLLM
from src.embeddings import SentenceTransformerEmbeddings
from src.vector_store import ChromaDBVectorStore, ChromaDBRetriever
from src.rag_pipeline import RAGPipeline
from src.langgraph_agent import LangGraphAgent

# Load environment variables
load_dotenv()


class StartupChatbot:
    """Main chatbot class."""
    
    def __init__(self, use_langgraph: bool = True):
        """Initialize the chatbot."""
        print("Initializing Startup Knowledge Chatbot...")
        
        # Load configuration
        collection_name = os.getenv("CHROMA_COLLECTION_NAME", "startup-knowledge-base")
        persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        llm_model = os.getenv("LLM_MODEL", "mosaicml/mpt-7b-instruct")
        device = os.getenv("DEVICE", "auto")  # "auto" will use MPS on Apple Silicon
        
        # Initialize components
        print("Loading embedding model...")
        embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
        
        print("Initializing ChromaDB (100% local storage)...")
        vector_store = ChromaDBVectorStore(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        
        retriever = vector_store.as_retriever(k=5)
        
        print("Loading LLM...")
        llm = MPTInstructLLM(
            model_name=llm_model,
            device=device,
            max_length=512,
            temperature=0.7
        )
        
        # Initialize RAG pipeline
        print("Setting up RAG pipeline...")
        self.rag_pipeline = RAGPipeline(llm=llm, retriever=retriever)
        
        # Initialize LangGraph agent if requested
        self.use_langgraph = use_langgraph
        if use_langgraph:
            print("Setting up LangGraph agent...")
            self.agent = LangGraphAgent(
                rag_pipeline=self.rag_pipeline,
                max_iterations=3
            )
        else:
            self.agent = None
        
        print("✓ Chatbot initialized successfully!\n")
    
    def ask(self, question: str) -> dict:
        """
        Ask a question and get an answer.
        
        Args:
            question: The question to ask
            
        Returns:
            dict with 'answer', 'documents', and optionally 'iterations'
        """
        if self.use_langgraph and self.agent:
            return self.agent.query(question)
        else:
            return self.rag_pipeline.query(question)


def main():
    """Main CLI entry point."""
    print("=" * 60)
    print("Startup Knowledge Q&A Chatbot")
    print("=" * 60)
    print()
    
    try:
        chatbot = StartupChatbot(use_langgraph=True)
        
        print("Chatbot ready! Type your questions (or 'quit' to exit)\n")
        
        while True:
            question = input("You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            print("\nThinking...")
            result = chatbot.ask(question)
            
            print(f"\nAnswer: {result['answer']}")
            
            if 'documents' in result and result['documents']:
                print(f"\nSources ({len(result['documents'])} documents):")
                for i, doc in enumerate(result['documents'][:3], 1):
                    source = doc.metadata.get('file_name', 'Unknown')
                    print(f"  {i}. {source}")
            
            if 'iterations' in result:
                print(f"\n(Completed in {result['iterations']} iteration(s))")
            
            print("\n" + "-" * 60 + "\n")
    
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
