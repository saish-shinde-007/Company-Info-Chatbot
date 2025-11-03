"""
LangChain RAG pipeline for question answering.
Combines retrieval and generation for answering questions.
"""

from typing import List
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document


class RAGPipeline:
    """RAG pipeline using LangChain."""
    
    def __init__(self, llm, retriever, chain_type: str = "stuff"):
        """
        Initialize RAG pipeline.
        
        Args:
            llm: Language model instance
            retriever: Document retriever
            chain_type: Type of chain ("stuff", "map_reduce", "refine", "map_rerank")
        """
        self.llm = llm
        self.retriever = retriever
        self.chain_type = chain_type
        
        # Create prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, say that you don't know.
Be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""
        )
        
        # Initialize QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=chain_type,
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt_template}
        )
    
    def query(self, question: str) -> dict:
        """
        Answer a question using RAG.
        
        Returns:
            dict with 'answer' and 'source_documents' keys
        """
        result = self.qa_chain({"query": question})
        return {
            'answer': result['result'],
            'source_documents': result.get('source_documents', [])
        }
    
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """Get relevant documents for a query."""
        # Update k if retriever supports it
        if hasattr(self.retriever, 'k'):
            self.retriever.k = k
        return self.retriever._get_relevant_documents(query)

