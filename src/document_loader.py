"""
Document loader for processing PDFs, Markdown, and Text files.
Handles various document formats and extracts text content.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import markdown
from bs4 import BeautifulSoup


class DocumentLoader:
    """Load and process documents from various formats."""
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.md', '.txt', '.markdown'}
    
    def __init__(self, documents_dir: str = "./documents"):
        self.documents_dir = Path(documents_dir)
        if not self.documents_dir.exists():
            self.documents_dir.mkdir(parents=True, exist_ok=True)
    
    def load_documents(self) -> List[Document]:
        """Load all supported documents from the documents directory."""
        documents = []
        
        if not self.documents_dir.exists():
            print(f"Documents directory '{self.documents_dir}' does not exist.")
            return documents
        
        # Find all supported files
        files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(self.documents_dir.glob(f"**/*{ext}"))
        
        print(f"Found {len(files)} document(s) to process...")
        
        for file_path in files:
            try:
                doc_list = self.load_file(file_path)
                documents.extend(doc_list)
                print(f"✓ Loaded {file_path.name}")
            except Exception as e:
                print(f"✗ Error loading {file_path.name}: {str(e)}")
        
        return documents
    
    def load_file(self, file_path: Path) -> List[Document]:
        """Load a single file based on its extension."""
        ext = file_path.suffix.lower()
        
        if ext == '.pdf':
            return self._load_pdf(file_path)
        elif ext in {'.md', '.markdown'}:
            return self._load_markdown(file_path)
        elif ext == '.txt':
            return self._load_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def _load_pdf(self, file_path: Path) -> List[Document]:
        """Load PDF file."""
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        
        # Add metadata
        for doc in docs:
            doc.metadata['source'] = str(file_path)
            doc.metadata['file_type'] = 'pdf'
            doc.metadata['file_name'] = file_path.name
        
        return docs
    
    def _load_markdown(self, file_path: Path) -> List[Document]:
        """Load Markdown file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Convert markdown to HTML, then extract text
        html = markdown.markdown(content)
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()
        
        doc = Document(
            page_content=text,
            metadata={
                'source': str(file_path),
                'file_type': 'markdown',
                'file_name': file_path.name
            }
        )
        
        return [doc]
    
    def _load_text(self, file_path: Path) -> List[Document]:
        """Load plain text file."""
        loader = TextLoader(str(file_path), encoding='utf-8')
        docs = loader.load()
        
        # Add metadata
        for doc in docs:
            doc.metadata['source'] = str(file_path)
            doc.metadata['file_type'] = 'text'
            doc.metadata['file_name'] = file_path.name
        
        return docs
    
    def chunk_documents(
        self,
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Document]:
        """Split documents into smaller chunks for embedding."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        
        return chunks

