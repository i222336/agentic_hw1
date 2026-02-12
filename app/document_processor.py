"""
Document Processor Module
Handles loading and processing documents using LangChain components.

Key LangChain Concepts Used:
1. Document Loaders: Load documents from various file formats
2. Text Splitters: Split documents into manageable chunks for embedding
"""

import os
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentProcessor:
    """
    Handles document loading and processing for the semantic search system.
    
    LangChain Components:
    - DirectoryLoader: Loads all documents from a directory
    - PyPDFLoader: Specifically loads PDF documents
    - RecursiveCharacterTextSplitter: Intelligently splits text into chunks
    """
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # LangChain TextSplitter: Splits documents while preserving context
        # RecursiveCharacterTextSplitter tries to keep paragraphs, sentences together
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # Try to split at natural boundaries
        )
    
    def load_documents_from_directory(self, directory_path: str) -> Tuple[List[Document], dict]:
        """
        Load all PDF documents from a directory using LangChain's DirectoryLoader.
        
        Args:
            directory_path: Path to directory containing PDF files
            
        Returns:
            Tuple of (list of Document objects, statistics dictionary)
            
        LangChain Concept:
        - DirectoryLoader: Automatically detects and loads all files of a specific type
        - Returns Document objects which contain 'page_content' and 'metadata'
        """
        try:
            # LangChain DirectoryLoader: Loads all PDFs from directory
            loader = DirectoryLoader(
                directory_path,
                glob="**/*.pdf",  # Pattern to match PDF files
                loader_cls=PyPDFLoader,  # Use PyPDFLoader for each PDF
                show_progress=True
            )
            
            # Load all documents - returns List[Document]
            documents = loader.load()
            
            # Calculate statistics
            stats = {
                "total_documents": len(set([doc.metadata.get('source', '') for doc in documents])),
                "total_pages": len(documents),
                "total_characters": sum(len(doc.page_content) for doc in documents)
            }
            
            return documents, stats
            
        except Exception as e:
            raise Exception(f"Error loading documents: {str(e)}")
    
    def load_single_document(self, file_path: str) -> Tuple[List[Document], dict]:
        """
        Load a single PDF document using LangChain's PyPDFLoader.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (list of Document objects, statistics dictionary)
            
        LangChain Concept:
        - PyPDFLoader: Specialized loader for PDF files
        - Automatically splits PDF into pages
        - Each page becomes a Document object with metadata
        """
        try:
            # LangChain PyPDFLoader: Loads a single PDF file
            loader = PyPDFLoader(file_path)
            
            # Load and split by pages
            documents = loader.load()
            
            stats = {
                "total_documents": 1,
                "total_pages": len(documents),
                "total_characters": sum(len(doc.page_content) for doc in documents)
            }
            
            return documents, stats
            
        except Exception as e:
            raise Exception(f"Error loading document {file_path}: {str(e)}")
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks using LangChain's TextSplitter.
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of split Document objects
            
        LangChain Concept:
        - TextSplitter maintains Document structure (content + metadata)
        - Adds chunk metadata to track which part of original document
        - Preserves semantic meaning by splitting at natural boundaries
        """
        try:
            # LangChain split_documents: Intelligently chunks documents
            # Preserves metadata and adds chunk information
            split_docs = self.text_splitter.split_documents(documents)
            
            return split_docs
            
        except Exception as e:
            raise Exception(f"Error splitting documents: {str(e)}")
    
    def get_document_info(self, file_path: str) -> dict:
        """
        Get information about a document file without fully loading it.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with file information
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_size = os.path.getsize(file_path)
            file_name = os.path.basename(file_path)
            
            return {
                "name": file_name,
                "size_bytes": file_size,
                "size_mb": round(file_size / (1024 * 1024), 2),
                "path": file_path
            }
            
        except Exception as e:
            raise Exception(f"Error getting document info: {str(e)}")
