
"""
Advanced RAG System with LangGraph

This is the main Streamlit application that demonstrates how to build a RAG system
using LangGraph for workflow management. The application handles document uploads,
processes questions, and generates answers using a LangGraph-orchestrated pipeline.

Key components:
- Document processing and chunking
- LangGraph workflow for RAG operations
- Question answering with fallback to online search
- Evaluation and quality assessment
- Real-time user interface

This implementation shows practical patterns for building RAG applications with
LangGraph, including state management, conditional routing, and error handling.
Good for understanding how LangGraph works with RAG systems.
"""
import streamlit as st

# Local imports
from config import QUESTION_PLACEHOLDER
from utils import clear_chroma_db, initialize_session_state
from ui_components import (
    setup_page_config, render_header, render_sidebar, 
    render_upload_section, render_upload_placeholder,
    render_question_section, render_answer_section
)
from document_loader import MultiModalDocumentLoader
from document_processor import DocumentProcessor
from rag_workflow import RAGWorkflow

# Initialize components
document_loader = MultiModalDocumentLoader()
document_processor = DocumentProcessor(document_loader)
rag_workflow = RAGWorkflow()


def handle_question_processing(question):
    """Handle the Q&A processing workflow"""
    # Debug info
    print(f"Processing question: {question}")
    
    with st.container():
        with st.spinner('🧠 Analyzing your question and retrieving relevant information...'):
            # Process the question - workflow will handle retriever automatically
            result = rag_workflow.process_question(question)
        
        # Render answer section (it will handle its own heading)
        render_answer_section(result)
        
        # Show evaluation scores and system information
        if result:
            st.markdown("---")
            st.markdown("### 📊 System Information")
            
            # Search Method Status
            search_method = result.get('search_method', 'Unknown')
            online_search = result.get('online_search', False)
            
            if search_method == 'online' or online_search:
                st.info("🌐 Online Search Used")
            elif search_method == 'documents':
                st.success("📄 Document Search Used")
            else:
                st.warning("❓ Search method not specified")
            
            # Create summary table
            summary_data = []
            
            # Document Evaluations Summary
            if 'document_evaluations' in result and result['document_evaluations']:
                evaluations = result['document_evaluations']
                relevant_count = sum(1 for eval in evaluations if eval.score.lower() == 'yes')
                total_count = len(evaluations)
                summary_data.append(["📋 Document Relevance", f"{relevant_count}/{total_count} relevant"])
                
                # Show average relevance score if available
                if hasattr(evaluations[0], 'relevance_score'):
                    avg_score = sum(eval.relevance_score for eval in evaluations) / len(evaluations)
                    summary_data.append(["📊 Avg. Doc Relevance", f"{avg_score:.2f}"])
            
            # Question-Answer Match
            if 'question_relevance_score' in result:
                q_relevance = result['question_relevance_score']
                if hasattr(q_relevance, 'binary_score'):
                    match_text = "✅ Well Matched" if q_relevance.binary_score else "❌ Poor Match"
                    summary_data.append(["❓ Question Match", match_text])
                if hasattr(q_relevance, 'relevance_score'):
                    summary_data.append(["📈 Question Score", f"{q_relevance.relevance_score:.2f}"])
                if hasattr(q_relevance, 'completeness'):
                    summary_data.append(["📝 Completeness", q_relevance.completeness])
            
            # Document Relevance Grading
            if 'document_relevance_score' in result:
                doc_relevance = result['document_relevance_score']
                if hasattr(doc_relevance, 'binary_score'):
                    grounding_text = "✅ Well Grounded" if doc_relevance.binary_score else "❌ Not Grounded"
                    summary_data.append(["🎯 Answer Grounding", grounding_text])
                if hasattr(doc_relevance, 'confidence'):
                    summary_data.append(["🔒 Confidence", f"{doc_relevance.confidence:.2f}"])
            
            # Display summary table
            if summary_data:
                import pandas as pd
                df = pd.DataFrame(summary_data, columns=["Metric", "Value"])
                st.table(df)
            
            # Show detailed evaluations in expandable section
            with st.expander("🔧 Detailed Evaluation Results"):
                
                # Document Evaluations Table
                if 'document_evaluations' in result and result['document_evaluations']:
                    st.markdown("**📋 Document Evaluation Details:**")
                    
                    eval_data = []
                    for i, eval in enumerate(result['document_evaluations']):
                        row = [f"Document {i+1}", eval.score]
                        
                        if hasattr(eval, 'relevance_score'):
                            row.append(f"{eval.relevance_score:.2f}")
                        else:
                            row.append("N/A")
                        
                        if hasattr(eval, 'coverage_assessment') and eval.coverage_assessment:
                            row.append(eval.coverage_assessment[:50] + "..." if len(eval.coverage_assessment) > 50 else eval.coverage_assessment)
                        else:
                            row.append("N/A")
                        
                        if hasattr(eval, 'missing_information') and eval.missing_information:
                            row.append(eval.missing_information[:50] + "..." if len(eval.missing_information) > 50 else eval.missing_information)
                        else:
                            row.append("N/A")
                        
                        eval_data.append(row)
                    
                    if eval_data:
                        eval_df = pd.DataFrame(eval_data, columns=["Document", "Score", "Relevance", "Coverage", "Missing Info"])
                        st.dataframe(eval_df, use_container_width=True)
                
                # Reasoning Table
                reasoning_data = []
                if 'question_relevance_score' in result and hasattr(result['question_relevance_score'], 'reasoning'):
                    reasoning_data.append(["Question Relevance", result['question_relevance_score'].reasoning])
                
                if 'document_relevance_score' in result and hasattr(result['document_relevance_score'], 'reasoning'):
                    reasoning_data.append(["Document Relevance", result['document_relevance_score'].reasoning])
                
                if reasoning_data:
                    st.markdown("**🧠 Evaluation Reasoning:**")
                    reasoning_df = pd.DataFrame(reasoning_data, columns=["Evaluation Type", "Reasoning"])
                    st.dataframe(reasoning_df, use_container_width=True)


def handle_user_interaction(user_file):
    """Handle user interactions for Q&A"""
    if user_file is None:
        render_upload_placeholder()
        return
    
    # Render question section
    question, ask_button = render_question_section(user_file)
    
    # Process question if submitted
    if ask_button and question.strip():
        handle_question_processing(question)
    elif ask_button and not question.strip():
        st.warning("Please enter a question before clicking Ask.")


def main():
    """Main application function"""
    # Initialize session state and clear DB only once
    initialize_session_state()
    
    # Clear ChromaDB only on first run
    if 'db_cleared' not in st.session_state:
        clear_chroma_db()
        st.session_state.db_cleared = True
        print("ChromaDB cleared on app startup")
    
    # Setup page and render UI
    setup_page_config()
    render_header()
    render_sidebar(document_loader)
    
    # Handle file upload
    user_file = render_upload_section(document_loader)
    
    # Process uploaded file
    if user_file:
        retriever = document_processor.process_file(user_file)
        if retriever:
            st.session_state.retriever = retriever
            print(f"File processed, retriever stored in session state")
        else:
            print(f"File processing failed - no retriever created")
    
    # Handle user interactions
    handle_user_interaction(user_file)


if __name__ == "__main__":
    main()
