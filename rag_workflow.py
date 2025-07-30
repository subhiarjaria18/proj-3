"""
RAG workflow management using LangGraph

This module implements the core RAG workflow using LangGraph's state management
and graph-based orchestration. It handles the complete flow from question
processing to answer generation, with built-in evaluation and fallback mechanisms.

The LangGraph workflow includes:
- Document retrieval and relevance checking
- Conditional routing between local and online search
- Multi-step answer generation and validation
- Error handling and recovery strategies

This demonstrates practical LangGraph RAG patterns for building robust
question-answering systems with proper workflow orchestration.
"""
import streamlit as st
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph

from state import GraphState
from chains.document_relevance import document_relevance
from chains.evaluate import evaluate_docs
from chains.generate_answer import generate_chain
from chains.question_relevance import question_relevance
from config import TAVILY_SEARCH_RESULTS


class RAGWorkflow:
    """
    Manages the RAG workflow using LangGraph
    
    This class orchestrates the complete RAG pipeline using LangGraph's state
    management system. It handles document processing, question answering,
    and evaluation with proper error handling and fallback mechanisms.
    
    The workflow demonstrates key LangGraph RAG patterns:
    - State-based workflow management
    - Conditional routing based on document availability
    - Multi-step evaluation and quality checks
    - Dynamic fallback to online search when needed
    
    Good for understanding how to build RAG systems with LangGraph in practice.
    """
    
    def __init__(self):
        self.graph = None
        self.retriever = None
        self._current_session_retriever_key = None
    
    def get_graph(self):
        """Get or create the graph instance (cached for performance)"""
        if 'graph_instance' not in st.session_state or st.session_state.graph_instance is None:
            st.session_state.graph_instance = self._create_graph()
        return st.session_state.graph_instance
    
    def set_retriever(self, retriever):
        """Set the document retriever"""
        self.retriever = retriever
        
        # Keep track of which session state retriever we're using
        if retriever is not None:
            current_file_key = st.session_state.get('processed_file')
            self._current_session_retriever_key = current_file_key
            print(f"Retriever set for file: {current_file_key}")
        else:
            self._current_session_retriever_key = None
            print("Retriever cleared")
    
    def get_current_retriever(self):
        """Get the current retriever, with fallback to session state"""
        # First check if we have a retriever set
        if self.retriever is not None:
            return self.retriever
            
        # Fallback to session state retriever
        session_retriever = st.session_state.get('retriever')
        if session_retriever is not None:
            print("Using retriever from session state")
            self.retriever = session_retriever
            return session_retriever
            
        return None
    
    def process_question(self, question):
        """Process a question through the RAG workflow"""
        print(f"STARTING RAG WORKFLOW for question: '{question}'")
        
        # Ensure we have the most current retriever
        current_retriever = self.get_current_retriever()
        self.set_retriever(current_retriever)
        
        graph = self.get_graph()
        result = graph.invoke(input={"question": question})
        
        print(f"RAG WORKFLOW COMPLETED")
        return result
    
    def _create_graph(self):
        """Create and configure the state graph for handling queries"""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("Retrieve Documents", self._retrieve)
        workflow.add_node("Grade Documents", self._evaluate)
        workflow.add_node("Generate Answer", self._generate_answer)
        workflow.add_node("Search Online", self._search_online)

        # Set entry point and edges
        workflow.set_entry_point("Retrieve Documents")
        workflow.add_edge("Retrieve Documents", "Grade Documents")
        workflow.add_conditional_edges(
            "Grade Documents",
            self._any_doc_irrelevant,
            {
                "Search Online": "Search Online",
                "Generate Answer": "Generate Answer",
            },
        )

        workflow.add_conditional_edges(
            "Generate Answer",
            self._check_hallucinations,
            {
                "Hallucinations detected": "Generate Answer",
                "Answers Question": END,
                "Question not addressed": "Search Online",
            },
        )
        workflow.add_edge("Search Online", "Generate Answer")

        return workflow.compile()
    
    def _retrieve(self, state: GraphState):
        """Retrieve documents relevant to the user's question"""
        print("GRAPH STATE: Retrieve Documents")
        question = state["question"]
        
        # Get the current retriever (with fallback to session state)
        current_retriever = self.get_current_retriever()
        
        # Debug: Print retriever status
        print(f"Current retriever status: {current_retriever is not None}")
        
        if current_retriever is None:
            print("No retriever available - going to online search")
            return {"documents": [], "question": question, "online_search": True}
        
        try:
            documents = current_retriever.invoke(question)
            print(f"Retrieved {len(documents)} documents from ChromaDB")
            return {"documents": documents, "question": question}
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            print("Clearing invalid retriever and falling back to online search")
            # Clear the invalid retriever
            self.retriever = None
            st.session_state.retriever = None
            return {"documents": [], "question": question, "online_search": True}
    
    def _evaluate(self, state: GraphState):
        """Filter documents based on their relevance to the question"""
        print("GRAPH STATE: Grade Documents")
        question = state["question"]
        documents = state["documents"]

        # Check if online search is already required
        online_search = state.get("online_search", False)
        print(f"Evaluating {len(documents)} documents, online_search: {online_search}")
        
        filtered_docs = []
        document_evaluations = []
        
        for document in documents:
            response = evaluate_docs.invoke({"question": question, "document": document.page_content})
            document_evaluations.append(response)
            
            result = response.score
            if result.lower() == "yes":
                filtered_docs.append(document)
            else:
                online_search = True
        
        print(f"Filtered to {len(filtered_docs)} relevant documents, online_search: {online_search}")
        
        # Determine search method
        search_method = "online" if online_search else "documents"
        
        return {
            "documents": filtered_docs, 
            "question": question, 
            "online_search": online_search,
            "search_method": search_method,
            "document_evaluations": document_evaluations
        }
    
    def _generate_answer(self, state: GraphState):
        """Generate an answer based on the retrieved documents"""
        print("GRAPH STATE: Generate Answer")
        question = state["question"]
        documents = state["documents"]
        
        print(f"Generating answer using {len(documents)} documents")
        solution = generate_chain.invoke({"context": documents, "question": question})
        print(f"Answer generated: {len(solution)} characters")
        return {"documents": documents, "question": question, "solution": solution}
    
    def _search_online(self, state: GraphState):
        """Search online for additional context if needed"""
        print("GRAPH STATE: Search Online")
        question = state["question"]
        documents = state["documents"]
        
        print(f"Searching online for: {question}")
        tavily_client = TavilySearchResults(k=TAVILY_SEARCH_RESULTS)
        response = tavily_client.invoke({"query": question})
        results = "\n".join([element["content"] for element in response])
        results = Document(page_content=results)
        
        if documents is not None:
            documents.append(results)
            print(f"Added online search results to {len(documents)-1} existing documents")
        else:
            documents = [results]
            print(f"Using only online search results")
        
        # Update search method to indicate online search was used
        return {
            "documents": documents, 
            "question": question, 
            "search_method": "online"
        }
    
    def _any_doc_irrelevant(self, state):
        """Determine whether any document is irrelevant, triggering online search"""
        online_search = state.get("online_search", False)
        next_state = "Search Online" if online_search else "Generate Answer"
        print(f"ROUTING DECISION: Going to '{next_state}' (online_search: {online_search})")
        return next_state
    
    def _check_hallucinations(self, state: GraphState):
        """Check for hallucinations in the generated answers"""
        print("GRAPH STATE: Check Hallucinations")
        question = state["question"]
        documents = state["documents"]
        solution = state["solution"]

        print("Checking document relevance...")
        doc_relevance_score = document_relevance.invoke(
            {"documents": documents, "solution": solution}
        )

        if doc_relevance_score.binary_score:
            print("Document relevance check passed")
            print("Checking question relevance...")
            question_relevance_score = question_relevance.invoke({"question": question, "solution": solution})
            
            # Store the evaluation scores in state
            state["document_relevance_score"] = doc_relevance_score
            state["question_relevance_score"] = question_relevance_score
            
            if question_relevance_score.binary_score:
                print("ROUTING DECISION: Going to 'END' (Answers Question)")
                return "Answers Question"
            else:
                print("ROUTING DECISION: Going to 'Search Online' (Question not addressed)")
                return "Question not addressed"
        else:
            print("ROUTING DECISION: Going to 'Generate Answer' (Hallucinations detected)")
            # Store the document relevance score even if it failed
            state["document_relevance_score"] = doc_relevance_score
            return "Hallucinations detected"
