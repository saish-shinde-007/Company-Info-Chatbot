# Setup Guide

Follow these steps to set up and run the Startup Knowledge Q&A Chatbot:

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you're using a GPU, you may want to install PyTorch with CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Step 2: Configure Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here
PINECONE_INDEX_NAME=startup-knowledge-base

# LLM Configuration
LLM_MODEL=mosaicml/mpt-7b-instruct
DEVICE=cpu  # Use "cuda" if you have a GPU

# Document Processing
DOCUMENTS_DIR=./documents
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Vector Store Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
DIMENSION=384
```

### Getting a Pinecone API Key:

1. Go to [https://www.pinecone.io/](https://www.pinecone.io/)
2. Sign up for a free account
3. Create a new index (or use the default)
4. Copy your API key from the dashboard

## Step 3: Add Documents

Place your documents (PDFs, Markdown, or Text files) in the `documents/` directory.

Example:
- `documents/company-handbook.pdf`
- `documents/onboarding.md`
- `documents/policies.txt`

A sample document has been included: `documents/sample_company_handbook.md`

## Step 4: Ingest Documents

Run the ingestion script to process documents and create embeddings:

```bash
python scripts/ingest_documents.py
```

This will:
1. Load all documents from the `documents/` directory
2. Split them into chunks
3. Generate embeddings
4. Upload to Pinecone

**Note:** This may take several minutes depending on the number and size of documents.

## Step 5: Run the Chatbot

### Option 1: Streamlit Web Interface (Recommended)

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Option 2: CLI Interface

```bash
python chatbot_cli.py
```

## Troubleshooting

### Model Download Issues

The MPT-7B-Instruct model is ~14GB and will be downloaded on first use. Ensure you have:
- Sufficient disk space
- Stable internet connection
- Patience (download may take 20-30 minutes)

### Pinecone Connection Issues

- Verify your API key is correct
- Check that your Pinecone index name matches the configuration
- Ensure you have sufficient Pinecone credits

### Memory Issues

If you encounter out-of-memory errors:
- Use `DEVICE=cpu` in your `.env` file
- Reduce `CHUNK_SIZE` to process smaller chunks
- Consider using a smaller embedding model

### Import Errors

If you see import errors, make sure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

## Next Steps

- Add more documents to expand the knowledge base
- Customize the prompt templates in `src/rag_pipeline.py`
- Adjust LangGraph workflow in `src/langgraph_agent.py`
- Fine-tune the chunk size and overlap for your documents

