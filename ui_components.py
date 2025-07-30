"""
UI components for the Advanced RAG application
"""
import streamlit as st
from config import (
    PAGE_TITLE, PAGE_ICON, LAYOUT, SIDEBAR_STATE, 
    FILE_CATEGORIES, UPLOAD_PLACEHOLDER_TITLE, UPLOAD_PLACEHOLDER_TEXT
)
from utils import format_file_size


def setup_page_config():
    """Sets up Streamlit page settings"""
    st.set_page_config(
        page_title=PAGE_TITLE, 
        page_icon=PAGE_ICON,
        layout=LAYOUT,
        initial_sidebar_state=SIDEBAR_STATE
    )


def render_header():
    """Shows the main header section"""
    st.title(f"{PAGE_ICON} Advanced RAG System")
    st.subheader("Intelligent Document Search & Analysis")
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🔍 **Smart Search**\n\nAdvanced retrieval with fallback to online sources")
    
    with col2:
        st.info("📄 **Multi-Format**\n\nPDF, Word, Excel, Code files supported")
    
    with col3:
        st.info("🤖 **LLM-Powered**\n\nLangGraph workflow with hallucination detection")
    
    st.divider()


def render_sidebar(document_loader):
    """Shows the sidebar with app info and file types"""
    with st.sidebar:
        # App info
        st.markdown("""
        <div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <h4>🔍 Advanced RAG System</h4>
            <p>Upload documents and ask intelligent questions using LLM-powered retrieval.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📋 Supported File Types")
        
        # Organized file type display
        for category, formats in FILE_CATEGORIES.items():
            with st.expander(category, expanded=False):
                for fmt in formats:
                    st.markdown(f"• {fmt}")


def render_upload_section(document_loader):
    """Shows the document upload section"""
    st.markdown("## 📤 Document Upload")
    
    # Upload area with simple styling
    st.info("📁 **Drag & Drop Your Document**\n\nSupported: PDF, Word, Excel, Text, Code files")
    
    # Show current supported extensions
    with st.expander("ℹ️ View All Supported Formats", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Supported extensions:** {document_loader.get_supported_extensions_display()}")
        with col2:
            st.write(f"**Total formats:** {len(document_loader.get_supported_extensions())}")
    
    # File uploader
    user_file = st.file_uploader(
        "Choose a file", 
        type=document_loader.get_supported_extensions(),
        help="Upload any supported document type.",
        label_visibility="collapsed"
    )
    
    return user_file


def render_file_analysis(file_info):
    """Shows file analysis metrics"""
    st.markdown("### 📊 File Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**📄 Filename**")
        st.write(file_info['filename'])
    
    with col2:
        st.markdown("**📏 Size**")
        size_display = format_file_size(file_info['size'])
        st.write(size_display)
    
    with col3:
        st.markdown("**🏷️ Type**")
        st.write(f".{file_info['extension'].upper()}")
    
    with col4:
        st.markdown("**📋 Status**")
        status_icon = "✅" if file_info['is_supported'] else "❌"
        status_text = "Supported" if file_info['is_supported'] else "Unsupported"
        st.write(f"{status_icon} {status_text}")


def render_upload_placeholder():
    """Shows placeholder when no file is uploaded"""
    st.markdown(f"""
    <div style="text-align: center; padding: 3rem; background: #f8fafc; border-radius: 10px; margin: 2rem 0;">
        <h3>{UPLOAD_PLACEHOLDER_TITLE}</h3>
        <p>{UPLOAD_PLACEHOLDER_TEXT}</p>
    </div>
    """, unsafe_allow_html=True)


def render_question_section(user_file):
    """Shows the question input section"""
    st.markdown("---")
    st.markdown("### 💬 Ask Questions About Your Document")
    
    # Display current file info
    file_display = f"📄 **Current Document:** {user_file.name}"
    if hasattr(user_file, 'type') and user_file.type:
        file_display += f" ({user_file.type})"
    st.markdown(file_display)
    
    # Question input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        question = st.text_input(
            'Enter your question:', 
            placeholder="What is the main topic of this document?",
            disabled=not user_file,
            help="Ask any question about the content of your uploaded document"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        ask_button = st.button("Ask", use_container_width=True)
    
    return question, ask_button


def render_answer_section(result):
    """Shows the answer section"""
    st.markdown("### 📝 Answer")
    st.success(result['solution'])
    st.markdown("---")
