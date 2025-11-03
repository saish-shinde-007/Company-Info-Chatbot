"""
Streamlit web interface for the Startup Knowledge Q&A Chatbot.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from src.llm_setup import MPTInstructLLM
from src.embeddings import SentenceTransformerEmbeddings
from src.vector_store import ChromaDBVectorStore, ChromaDBRetriever
from src.rag_pipeline import RAGPipeline
from src.langgraph_agent import LangGraphAgent

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Startup Knowledge Q&A Chatbot",
    page_icon="🤖",
    layout="wide"
)


@st.cache_resource
def initialize_chatbot(use_langgraph: bool = True):
    """Initialize and cache the chatbot."""
    # Load configuration
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "startup-knowledge-base")
    persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    llm_model = os.getenv("LLM_MODEL", "mosaicml/mpt-7b-instruct")
    device = os.getenv("DEVICE", "auto")  # "auto" will use MPS on Apple Silicon
    
    # Initialize components
    with st.spinner("Loading embedding model..."):
        embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
    
    with st.spinner("Initializing ChromaDB (100% local storage)..."):
        vector_store = ChromaDBVectorStore(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        retriever = vector_store.as_retriever(k=5)
    
    with st.spinner("Loading LLM (this may take a few minutes on first run)..."):
        llm = MPTInstructLLM(
            model_name=llm_model,
            device=device,
            max_length=512,
            temperature=0.7
        )
    
    # Initialize RAG pipeline
    with st.spinner("Setting up RAG pipeline..."):
        rag_pipeline = RAGPipeline(llm=llm, retriever=retriever)
    
    # Initialize LangGraph agent if requested
    agent = None
    if use_langgraph:
        with st.spinner("Setting up LangGraph agent..."):
            agent = LangGraphAgent(
                rag_pipeline=rag_pipeline,
                max_iterations=3
            )
    
    return {
        "rag_pipeline": rag_pipeline,
        "agent": agent,
        "use_langgraph": use_langgraph
    }


def main():
    """Main Streamlit app."""
    st.title("🤖 Startup Knowledge Q&A Chatbot")
    st.markdown("Ask questions about your internal documents and get instant answers!")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        use_langgraph = st.checkbox(
            "Use LangGraph (Multi-step orchestration)",
            value=True,
            help="Enable multi-step query refinement for complex questions"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This chatbot uses:
        - **LangChain** for RAG workflows
        - **LangGraph** for query orchestration
        - **ChromaDB** for semantic search (100% local)
        - **MPT-7B-Instruct** for natural language generation
        """)
    
    # Initialize chatbot
    chatbot_config = initialize_chatbot(use_langgraph=use_langgraph)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if "sources" in message:
                with st.expander("📄 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"{i}. **{source['file']}**")
                        if source.get("score"):
                            st.caption(f"Relevance: {source['score']:.2%}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        answer = "I couldn't generate an answer."
        sources = []
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if chatbot_config["use_langgraph"] and chatbot_config["agent"]:
                        result = chatbot_config["agent"].query(prompt)
                    else:
                        result = chatbot_config["rag_pipeline"].query(prompt)
                    
                    answer = result.get("answer", "I couldn't generate an answer.")
                    
                    st.markdown(answer)
                    
                    # Extract sources
                    if "documents" in result and result["documents"]:
                        for doc in result["documents"]:
                            sources.append({
                                "file": doc.metadata.get("file_name", "Unknown"),
                                "score": doc.metadata.get("score", 0.0)
                            })
                    
                    # Show sources
                    if sources:
                        with st.expander("📄 Sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"{i}. **{source['file']}**")
                                if source.get("score"):
                                    st.caption(f"Relevance: {source['score']:.2%}")
                    
                    # Show iterations if using LangGraph
                    if "iterations" in result:
                        st.caption(f"Completed in {result['iterations']} iteration(s)")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    answer = f"Error occurred: {str(e)}"
        
        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })


if __name__ == "__main__":
    main()
