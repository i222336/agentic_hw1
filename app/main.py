"""
Main Entry Point for Semantic Search Engine
CS-4015 Agentic AI - Homework 1 Phase 1

This application demonstrates:
1. GUI-based data selection and management
2. Embedding model and vector store configuration using LangChain
3. Semantic retrieval using LangChain's vector stores
4. Retrieval evaluation and analysis

LangChain Integration:
- Document loaders for PDF processing
- Text splitters for chunking
- HuggingFace embeddings for semantic representation
- FAISS/Chroma vector stores for efficient retrieval
- Similarity search for semantic queries
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.gui import launch_gui


def main():
    """
    Main function to launch the Semantic Search Engine GUI.
    
    The application pipeline:
    1. User selects data source (directory/file/sample data)
    2. User configures embedding model and vector database
    3. System processes documents using LangChain:
       - Loads PDFs using PyPDFLoader
       - Splits into chunks using RecursiveCharacterTextSplitter
       - Generates embeddings using HuggingFaceEmbeddings
       - Stores in FAISS or Chroma vector store
    4. User performs semantic searches
    5. System retrieves and ranks results by relevance
    """
    print("=" * 70)
    print("🤖 AI RESEARCH ASSISTANT - SEMANTIC SEARCH MODULE")
    print("CS-4015 Agentic AI - Homework 1 Phase 1")
    print("=" * 70)
    print("\nLaunching GUI application...")
    print("\nLangChain Components in Use:")
    print("  • Document Loaders (PyPDFLoader, DirectoryLoader)")
    print("  • Text Splitters (RecursiveCharacterTextSplitter)")
    print("  • Embeddings (HuggingFaceEmbeddings)")
    print("  • Vector Stores (FAISS, Chroma)")
    print("  • Retrieval (similarity_search)")
    print("=" * 70)
    print()
    
    try:
        launch_gui()
    except KeyboardInterrupt:
        print("\n\nApplication closed by user.")
    except Exception as e:
        print(f"\n\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

