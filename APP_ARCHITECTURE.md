# Startup Knowledge Chatbot - Architecture & File Guide

## 🎯 What is This App?

This is a **RAG (Retrieval-Augmented Generation) Chatbot** that answers questions from your internal company documents. It's built to run **100% locally** - no cloud services, no API keys needed!

### Key Features:
- 📄 Processes PDFs, Markdown, and Text files
- 🔍 Semantic search using embeddings stored in ChromaDB
- 🤖 Multi-step query orchestration with LangGraph
- 💬 Natural language Q&A interface
- 🖥️ **100% local** - Everything runs on your machine

---

## 🏗️ How It's Created: Architecture Overview

The app follows a **modular architecture** with clear separation of concerns:

```
User Question
    ↓
Interface Layer (app.py / chatbot_cli.py)
    ↓
Orchestration Layer (langgraph_agent.py) [Optional]
    ↓
RAG Pipeline (rag_pipeline.py)
    ↓
    ├──→ Retrieval (vector_store.py) → ChromaDB
    └──→ Generation (llm_setup.py) → MPT-7B-Instruct
```

### Data Flow:
1. **Document Ingestion**: Documents → Chunks → Embeddings → ChromaDB
2. **Query Processing**: Question → Embedding → Similarity Search → Context + Question → LLM → Answer

---

## 📁 File Structure & Importance

### 🎨 **User Interface Files**

#### `app.py` ⭐ **CRITICAL**
**Purpose**: Streamlit web interface - the main user-facing application

**What it does**:
- Creates a beautiful web UI using Streamlit
- Manages chat history and session state
- Initializes all components (LLM, embeddings, vector store, RAG pipeline)
- Handles user questions and displays answers with sources
- Shows document sources with relevance scores

**Why it's important**: This is what users interact with - without it, there's no web interface!

**Key Features**:
- Caches chatbot initialization (expensive to load models)
- Supports both simple RAG and LangGraph orchestration
- Displays sources and relevance scores
- Handles errors gracefully

---

#### `chatbot_cli.py` ⭐ **IMPORTANT**
**Purpose**: Command-line interface for the chatbot

**What it does**:
- Provides a terminal-based interface
- Initializes the chatbot (same as web version)
- Allows interactive Q&A sessions
- Useful for debugging and programmatic access

**Why it's important**: Alternative interface for users who prefer CLI, or for automation/scripts

**Key Classes**:
- `StartupChatbot`: Main chatbot class that can be imported programmatically

---

### 🧠 **Core Processing Files** (`src/`)

#### `src/document_loader.py` ⭐ **CRITICAL**
**Purpose**: Loads and processes documents from various formats

**What it does**:
- Reads PDF, Markdown, and Text files
- Extracts text content from different formats
- Splits documents into chunks (for embedding)
- Adds metadata (file name, type, source)

**Why it's important**: Without this, you can't ingest documents into the system!

**Key Methods**:
- `load_documents()`: Loads all files from documents/ directory
- `chunk_documents()`: Splits large documents into smaller chunks (needed for embeddings)

**Dependencies**: PyPDF, markdown, BeautifulSoup

---

#### `src/embeddings.py` ⭐ **CRITICAL**
**Purpose**: Generates vector embeddings from text

**What it does**:
- Wraps sentence-transformers for LangChain compatibility
- Converts text into numerical vectors (embeddings)
- Used for both documents (during ingestion) and queries (during search)

**Why it's important**: Embeddings enable semantic search - without them, you can't find similar documents!

**Key Methods**:
- `embed_documents()`: Embed multiple texts (for document ingestion)
- `embed_query()`: Embed a single query (for search)

**Model Used**: `sentence-transformers/all-MiniLM-L6-v2` (default, can be changed)

---

#### `src/vector_store.py` ⭐ **CRITICAL**
**Purpose**: Manages ChromaDB - the local vector database

**What it does**:
- Initializes ChromaDB connection (persistent local storage)
- Stores document embeddings
- Performs similarity search (finds relevant documents)
- Provides retriever interface for LangChain

**Why it's important**: This is the "memory" of the system - all your documents are stored here!

**Key Classes**:
- `ChromaDBVectorStore`: Main vector store wrapper
  - `add_documents()`: Store documents with embeddings
  - `similarity_search()`: Find relevant documents for a query
- `ChromaDBRetriever`: LangChain-compatible retriever interface

**Storage**: Everything stored locally in `./chroma_db/` directory

---

#### `src/llm_setup.py` ⭐ **CRITICAL**
**Purpose**: Sets up and wraps the MPT-7B-Instruct language model

**What it does**:
- Loads MPT-7B-Instruct model (runs locally, no API needed!)
- Wraps it to work with LangChain
- Formats prompts for instruction-following
- Handles text generation with configurable parameters

**Why it's important**: This generates the actual answers - the "brain" of the chatbot!

**Key Features**:
- Supports both CPU and GPU (CUDA)
- Formats prompts as instruction-following (MPT format)
- Configurable temperature, max_length, top_p
- Downloads model on first use (~14GB)

**Model**: `mosaicml/mpt-7b-instruct` (free, runs locally)

---

#### `src/rag_pipeline.py` ⭐ **CRITICAL**
**Purpose**: Implements the RAG (Retrieval-Augmented Generation) pipeline

**What it does**:
- Combines retrieval (finding relevant docs) with generation (creating answers)
- Uses LangChain's RetrievalQA chain
- Formats context and question into a prompt
- Returns answer with source documents

**Why it's important**: This is the core logic that makes the chatbot work - it connects retrieval to generation!

**How it works**:
1. Takes a question
2. Retrieves relevant documents using retriever
3. Combines documents as context
4. Generates answer using LLM with context
5. Returns answer + sources

**Prompt Template**: Includes context and question, instructs LLM to use context

---

#### `src/langgraph_agent.py` ⭐ **ENHANCED FEATURE**
**Purpose**: Multi-step query orchestration for complex questions

**What it does**:
- Orchestrates multiple retrieval and generation steps
- Evaluates if answer needs refinement
- Refines queries iteratively for better results
- Handles complex questions that need multiple passes

**Why it's important**: Enables the chatbot to handle complex questions that need multiple reasoning steps!

**Workflow**:
1. Retrieve documents
2. Generate answer
3. Evaluate answer quality
4. If needed, refine query and repeat (up to max_iterations)
5. Return final answer

**Key Nodes**:
- `retrieve`: Find relevant documents
- `answer`: Generate answer from context
- `evaluate`: Check if answer is good enough
- `refine_query`: Improve query for next iteration

---

### 🛠️ **Utility Scripts**

#### `scripts/ingest_documents.py` ⭐ **CRITICAL**
**Purpose**: Processes documents and adds them to ChromaDB

**What it does**:
- Loads all documents from `documents/` directory
- Chunks them into smaller pieces
- Generates embeddings
- Stores everything in ChromaDB

**Why it's important**: You must run this before using the chatbot - it's the "setup" step!

**Process**:
1. Load documents (PDF, MD, TXT)
2. Split into chunks
3. Initialize embedding model
4. Initialize ChromaDB
5. Add documents with embeddings

**Run this**: Before first use, and whenever you add new documents

---

### 📋 **Configuration Files**

#### `requirements.txt` ⭐ **CRITICAL**
**Purpose**: Lists all Python dependencies

**What it contains**:
- LangChain ecosystem (langchain, langgraph, langchain-community)
- ChromaDB (local vector database)
- PyTorch & Transformers (for LLM)
- Sentence Transformers (for embeddings)
- Streamlit (web interface)
- Document processing libraries (pypdf, markdown, etc.)

**Why it's important**: Without this, you can't install dependencies!

---

#### `.env` (not in repo, you create it)
**Purpose**: Configuration settings

**Key Variables**:
- `CHROMA_COLLECTION_NAME`: Database collection name
- `CHROMA_PERSIST_DIR`: Where to store ChromaDB
- `EMBEDDING_MODEL`: Which embedding model to use
- `LLM_MODEL`: Which LLM to use
- `DEVICE`: cpu or cuda
- `CHUNK_SIZE`: Size of document chunks
- `CHUNK_OVERLAP`: Overlap between chunks

**Why it's important**: Allows customization without changing code!

---

#### `README.md` ⭐ **IMPORTANT**
**Purpose**: Documentation and setup instructions

**Why it's important**: Helps users understand how to use the app!

---

### 📂 **Data Directories**

#### `documents/` 📁
**Purpose**: Place your PDF, Markdown, and Text files here

**What happens**: Files here are processed by `ingest_documents.py` and added to ChromaDB

---

#### `chroma_db/` 📁 (auto-created)
**Purpose**: Local ChromaDB storage

**What's stored**: All document embeddings and metadata - this is your "knowledge base"!

**Why it's important**: Persistent storage - your ingested documents stay here even after restarting

---

## 🔄 How Everything Works Together

### Initial Setup Flow:
```
1. User adds documents to documents/
2. Run scripts/ingest_documents.py
   → DocumentLoader loads files
   → Documents split into chunks
   → Embeddings generated
   → Stored in ChromaDB
3. Ready to use!
```

### Query Flow:
```
1. User asks question (via app.py or chatbot_cli.py)
2. Option A: Simple RAG
   → RAGPipeline queries ChromaDB
   → Finds relevant documents
   → LLM generates answer with context
   
3. Option B: LangGraph (enhanced)
   → LangGraphAgent orchestrates
   → Multiple retrieval/generation steps
   → Refines query if needed
   → Returns final answer
```

---

## 🎯 Key Design Decisions

### Why ChromaDB?
- **100% local** - no cloud, no API keys
- Persistent storage - data survives restarts
- Fast similarity search
- Easy to use with LangChain

### Why MPT-7B-Instruct?
- **Free** - no API costs
- **Local** - runs on your machine
- **Open source** - fully controllable
- Good instruction-following capabilities

### Why LangGraph?
- Handles complex, multi-step queries
- Can refine queries iteratively
- Better answers for complex questions
- Optional - simple RAG also works

### Why Sentence Transformers?
- Lightweight embedding model
- Fast inference
- Good quality for semantic search
- Runs locally

---

## 🚀 Getting Started Checklist

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Create `.env` file (or use defaults)
3. ✅ Add documents to `documents/` directory
4. ✅ Run ingestion: `python scripts/ingest_documents.py`
5. ✅ Start chatbot: `streamlit run app.py` or `python chatbot_cli.py`

---

## 🔧 Customization Points

- **Change LLM**: Update `LLM_MODEL` in `.env` or modify `src/llm_setup.py`
- **Change Embeddings**: Update `EMBEDDING_MODEL` in `.env`
- **Adjust Chunking**: Modify `CHUNK_SIZE` and `CHUNK_OVERLAP` in `.env`
- **Add File Types**: Extend `DocumentLoader.SUPPORTED_EXTENSIONS`
- **Modify Prompts**: Edit prompt template in `src/rag_pipeline.py`
- **Change Orchestration**: Modify graph in `src/langgraph_agent.py`

---

## 🎓 Learning Resources

This app demonstrates:
- **RAG Architecture**: Retrieval-Augmented Generation pattern
- **Vector Databases**: Using ChromaDB for semantic search
- **LangChain**: Building LLM applications
- **LangGraph**: Multi-step agent workflows
- **Local LLM Deployment**: Running models without APIs
- **Document Processing**: Handling multiple file formats

---

## 🐛 Troubleshooting

**Common Issues**:
1. **No documents found**: Make sure files are in `documents/` directory
2. **Model download slow**: First run downloads ~14GB model (be patient!)
3. **Out of memory**: Reduce `CHUNK_SIZE` or use smaller models
4. **Import errors**: Make sure all dependencies installed

---

## 📝 Summary

This is a **production-ready RAG chatbot** that:
- Runs 100% locally (private, no costs)
- Uses modern AI stack (LangChain, ChromaDB, Transformers)
- Supports multiple interfaces (Web, CLI)
- Handles complex queries (with LangGraph)
- Is fully customizable and extensible

Each file has a specific, important role in making the chatbot work!

