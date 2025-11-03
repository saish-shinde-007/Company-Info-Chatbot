# Startup Knowledge Q&A Chatbot

A RAG (Retrieval-Augmented Generation) chatbot for answering questions from internal documents, supporting onboarding and documentation search.

## 🚀 Quick Start

**Everything runs on your local machine - completely offline!**

### Option 1: Use the Quick Start Script (Easiest)

```bash
cd ~/Desktop/startup-knowledge-chatbot
./run_local.sh
```

### Option 2: Manual Setup

1. **Install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Create `.env` file** (no API keys needed!):
   ```bash
   # Copy .env.example to .env
   cp .env.example .env
   # That's it! No API keys needed for ChromaDB
   ```

3. **Ingest documents:**
   ```bash
   python scripts/ingest_documents.py
   ```

4. **Run the chatbot:**
   ```bash
   # Web interface (recommended)
   streamlit run app.py
   
   # Or CLI
   python chatbot_cli.py
   ```

📖 **For detailed local setup, see [LOCAL_SETUP.md](LOCAL_SETUP.md)**

## Tech Stack

- **LangChain**: RAG workflows and document processing
- **LangGraph**: Multi-step query orchestration
- **ChromaDB**: Local vector database for semantic search (100% local, no cloud!)
- **MPT-7B-Instruct**: Free LLM that runs locally (no API costs!)
- **Document Support**: PDFs, Markdown, and Text files

## Features

- 📄 Process PDFs, Markdown, and Text files
- 🔍 Semantic search using embeddings stored locally in ChromaDB
- 🤖 Multi-step query orchestration with LangGraph
- 💬 Natural language Q&A interface
- 🖥️ **Runs 100% locally** - No cloud services, no API keys needed!
- 🔄 Supports both CLI and Streamlit web interface

## What Runs Locally?

✅ **Everything runs on your machine:**
- Document processing
- Embedding generation
- Vector database (ChromaDB - stored locally)
- LLM inference (MPT-7B-Instruct)
- User interfaces (Streamlit/CLI)

## Usage Examples

### CLI Usage

```bash
python chatbot_cli.py
# Then type your questions
```

### Streamlit Interface

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### Programmatic Usage

```python
from chatbot_cli import StartupChatbot

chatbot = StartupChatbot()
response = chatbot.ask("What is our company's vacation policy?")
print(response['answer'])
```

## Configuration

Key configuration options in `.env`:

- `CHROMA_COLLECTION_NAME`: Name of ChromaDB collection (default: startup-knowledge-base)
- `CHROMA_PERSIST_DIR`: Local directory to store ChromaDB (default: ./chroma_db)
- `LLM_MODEL`: LLM model to use (default: mosaicml/mpt-7b-instruct)
- `DEVICE`: `cpu` or `cuda` (for GPU)
- `CHUNK_SIZE`: Size of text chunks for embedding (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)

**Note**: No API keys needed! Everything runs locally.

## System Requirements

- **Python**: 3.8 or higher ✅ (You have 3.12.3)
- **RAM**: 8GB minimum (16GB recommended)
- **Disk Space**: ~20GB free (for models and ChromaDB)
- **Optional**: NVIDIA GPU for faster inference

## Notes

- The MPT-7B-Instruct model will be downloaded on first use (~14GB)
- Processing large documents may take time
- ChromaDB stores everything locally in the `chroma_db/` directory
- All your data stays on your machine - completely private!

## License
Free to use and modify.
