#!/bin/bash

# Quick Start Script for Local Setup (100% Local with ChromaDB)
# Usage: ./run_local.sh

echo "=========================================="
echo "Startup Knowledge Chatbot - Local Setup"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file (100% local - no API keys needed!)..."
    cat > .env << ENVEOF
# ChromaDB Configuration
CHROMA_COLLECTION_NAME=startup-knowledge-base
CHROMA_PERSIST_DIR=./chroma_db

# LLM Configuration (Runs locally)
LLM_MODEL=mosaicml/mpt-7b-instruct
DEVICE=cpu

# Document Processing
DOCUMENTS_DIR=./documents
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Embeddings (Runs locally)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENVEOF
    echo "✓ .env file created"
    echo ""
    echo "✅ No API keys needed - everything runs locally!"
fi

# Check if requirements are installed
echo "📚 Checking dependencies..."
if ! python -c "import langchain" 2>/dev/null; then
    echo "Installing dependencies (this may take a few minutes)..."
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "✓ Dependencies already installed"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Choose an option:"
echo ""
echo "1. Run Streamlit Web Interface (Recommended)"
echo "2. Run CLI Interface"
echo "3. Ingest Documents Only"
echo "4. Exit"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting Streamlit..."
        echo "   Open http://localhost:8501 in your browser"
        echo ""
        streamlit run app.py
        ;;
    2)
        echo ""
        echo "🚀 Starting CLI..."
        echo ""
        python chatbot_cli.py
        ;;
    3)
        echo ""
        echo "📄 Ingesting documents..."
        python scripts/ingest_documents.py
        ;;
    4)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
