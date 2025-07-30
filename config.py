"""
Configuration settings for the Advanced RAG application
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# UI Configuration
PAGE_TITLE = "Advanced RAG"
PAGE_ICON = "🔎"
LAYOUT = "wide"
SIDEBAR_STATE = "expanded"

# File Processing Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
CHROMA_COLLECTION_NAME = "rag-chroma"
CHROMA_PERSIST_DIR = "./.chroma"

# Model Configuration
LLM_TEMPERATURE = 0
TAVILY_SEARCH_RESULTS = 2

# Supported File Types
SUPPORTED_EXTENSIONS = [
    "pdf", "docx", "doc", "csv", "xlsx", "xls", 
    "txt", "md", "py", "js", "html", "xml"
]

# UI Messages
UPLOAD_PLACEHOLDER_TITLE = "📤 Upload a document to get started"
UPLOAD_PLACEHOLDER_TEXT = "Once you upload a file, you'll be able to ask questions about its content."
QUESTION_PLACEHOLDER = "What is the main topic of this document?"

# File Categories for UI Display
FILE_CATEGORIES = {
    "📄 Documents": ["PDF (.pdf)", "Word (.docx, .doc)", "Text (.txt, .md)"],
    "📊 Data Files": ["Excel (.xlsx, .xls)", "CSV (.csv)"],
    "💻 Code Files": ["Python (.py)", "JavaScript (.js)", "HTML (.html)", "XML (.xml)"]
}
