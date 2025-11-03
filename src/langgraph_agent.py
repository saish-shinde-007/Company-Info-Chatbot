"""
LangGraph multi-step query orchestration.
Handles complex queries that may require multiple retrieval and reasoning steps.
"""

from typing import TypedDict, Annotated, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END


class GraphState(TypedDict):
    """State schema for LangGraph."""
    question: str
    documents: Annotated[List[Document], "documents"]
    answer: str
    iteration: int
    max_iterations: int
    needs_refinement: bool


class LangGraphAgent:
    """Multi-step query orchestration using LangGraph."""
    
    def __init__(self, rag_pipeline, max_iterations: int = 3):
        """
        Initialize LangGraph agent.
        
        Args:
            rag_pipeline: RAGPipeline instance
            max_iterations: Maximum number of refinement iterations
        """
        self.rag_pipeline = rag_pipeline
        self.max_iterations = max_iterations
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("generate", self._generate_answer)
        workflow.add_node("evaluate", self._evaluate_answer)
        workflow.add_node("refine_query", self._refine_query)
        
        # Set entry point
        workflow.set_entry_point("retrieve")
        
        # Add edges
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "evaluate")
        
        # Conditional edge from evaluate
        workflow.add_conditional_edges(
            "evaluate",
            self._should_refine,
            {
                "refine": "refine_query",
                "end": END
            }
        )
        
        workflow.add_edge("refine_query", "retrieve")
        
        return workflow.compile()
    
    def _retrieve_documents(self, state: GraphState) -> GraphState:
        """Retrieve relevant documents."""
        question = state["question"]
        documents = self.rag_pipeline.get_relevant_documents(question, k=5)
        state["documents"] = documents
        return state
    
    def _generate_answer(self, state: GraphState) -> GraphState:
        """Generate answer using RAG."""
        question = state["question"]
        result = self.rag_pipeline.query(question)
        state["answer"] = result["answer"]
        if "documents" not in state or not state["documents"]:
            state["documents"] = result.get("source_documents", [])
        return state
    
    def _evaluate_answer(self, state: GraphState) -> GraphState:
        """Evaluate if answer needs refinement."""
        answer = state["answer"]
        iteration = state.get("iteration", 0) + 1
        state["iteration"] = iteration
        
        # Simple heuristic: check if answer indicates uncertainty
        uncertainty_phrases = [
            "i don't know",
            "not provided",
            "not mentioned",
            "unclear",
            "cannot determine"
        ]
        
        answer_lower = answer.lower()
        needs_refinement = (
            iteration < state["max_iterations"] and
            any(phrase in answer_lower for phrase in uncertainty_phrases)
        )
        
        state["needs_refinement"] = needs_refinement
        return state
    
    def _refine_query(self, state: GraphState) -> GraphState:
        """Refine the query for better retrieval."""
        question = state["question"]
        answer = state["answer"]
        
        # Add context from previous answer to refine query
        refined_question = f"{question} Based on: {answer[:200]}"
        state["question"] = refined_question
        return state
    
    def _should_refine(self, state: GraphState) -> str:
        """Determine if query should be refined."""
        if state.get("needs_refinement", False):
            return "refine"
        return "end"
    
    def query(self, question: str) -> dict:
        """
        Execute multi-step query orchestration.
        
        Returns:
            dict with 'answer', 'documents', and 'iterations' keys
        """
        initial_state = {
            "question": question,
            "documents": [],
            "answer": "",
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "needs_refinement": False
        }
        
        final_state = self.graph.invoke(initial_state)
        
        return {
            "answer": final_state["answer"],
            "documents": final_state["documents"],
            "iterations": final_state["iteration"]
        }
