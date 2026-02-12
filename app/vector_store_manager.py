"""
Vector Store Manager Module
Manages embeddings and vector databases using LangChain.

Key LangChain Concepts:
1. Embeddings: Convert text to vector representations
2. Vector Stores: Store and retrieve embeddings efficiently
3. HuggingFaceEmbeddings: Use Hugging Face models for embeddings
4. FAISS & Chroma: Two different vector database backends
"""

import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from app.config import EMBEDDING_MODELS, VECTOR_STORE_DIR


class VectorStoreManager:
    """
    Manages the creation and interaction with vector stores using LangChain.
    
    LangChain Components:
    - HuggingFaceEmbeddings: Wrapper for Hugging Face embedding models
    - FAISS: Facebook AI Similarity Search - fast in-memory vector store
    - Chroma: Persistent vector store with additional features
    """
    
    def __init__(self):
        """Initialize the Vector Store Manager."""
        self.embeddings = None
        self.vector_store = None
        self.current_model = None
        self.current_db_type = None
    
    def initialize_embeddings(self, model_key: str) -> None:
        """
        Initialize the embedding model using LangChain's HuggingFaceEmbeddings.
        
        Args:
            model_key: Key from EMBEDDING_MODELS config
            
        LangChain Concept:
        - HuggingFaceEmbeddings: LangChain wrapper for sentence-transformers
        - Automatically handles model download and caching
        - Provides consistent interface for different embedding models
        """
        try:
            if model_key not in EMBEDDING_MODELS:
                raise ValueError(f"Invalid model key: {model_key}")
            
            model_info = EMBEDDING_MODELS[model_key]
            model_name = model_info["name"]
            
            # LangChain HuggingFaceEmbeddings: Wraps Hugging Face models
            # model_name: The Hugging Face model identifier
            # model_kwargs: Additional arguments (e.g., device selection)
            # encode_kwargs: Arguments for the encode function
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
                encode_kwargs={'normalize_embeddings': True}  # Normalize for cosine similarity
            )
            
            self.current_model = model_key
            print(f"✓ Initialized embedding model: {model_key}")
            
        except Exception as e:
            raise Exception(f"Error initializing embeddings: {str(e)}")
    
    def create_vector_store(self, documents: List[Document], db_type: str, 
                          collection_name: str = "semantic_search") -> None:
        """
        Create a vector store from documents using LangChain.
        
        Args:
            documents: List of Document objects to embed and store
            db_type: Type of vector database ("FAISS" or "Chroma")
            collection_name: Name for the collection/index
            
        LangChain Concept:
        - from_documents(): Class method that:
          1. Automatically embeds all documents using the embedding model
          2. Creates the vector store
          3. Stores documents with their embeddings and metadata
        """
        try:
            if self.embeddings is None:
                raise ValueError("Embeddings not initialized. Call initialize_embeddings first.")
            
            if not documents:
                raise ValueError("No documents provided to create vector store.")
            
            print(f"Creating {db_type} vector store with {len(documents)} document chunks...")
            
            if db_type == "FAISS":
                # LangChain FAISS: Fast in-memory vector store
                # from_documents: Embeds documents and creates FAISS index
                # Uses Facebook's FAISS library for efficient similarity search
                self.vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
                
            elif db_type == "Chroma":
                # LangChain Chroma: Persistent vector store
                # from_documents: Embeds documents and stores in Chroma DB
                # Supports persistence, metadata filtering, and more
                persist_directory = os.path.join(VECTOR_STORE_DIR, collection_name)
                
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    collection_name=collection_name,
                    persist_directory=persist_directory
                )
                
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
            
            self.current_db_type = db_type
            print(f"✓ Vector store created successfully with {db_type}")
            
        except Exception as e:
            raise Exception(f"Error creating vector store: {str(e)}")
    
    def save_vector_store(self, save_path: str) -> None:
        """
        Save the vector store to disk (primarily for FAISS).
        
        Args:
            save_path: Path to save the vector store
            
        LangChain Concept:
        - FAISS.save_local(): Persists FAISS index to disk
        - Chroma automatically persists if persist_directory is set
        """
        try:
            if self.vector_store is None:
                raise ValueError("No vector store to save.")
            
            if self.current_db_type == "FAISS":
                # FAISS needs explicit saving
                os.makedirs(save_path, exist_ok=True)
                self.vector_store.save_local(save_path)
                print(f"✓ FAISS vector store saved to {save_path}")
                
            elif self.current_db_type == "Chroma":
                # Chroma persists automatically
                print("✓ Chroma vector store is automatically persisted")
            
        except Exception as e:
            raise Exception(f"Error saving vector store: {str(e)}")
    
    def load_vector_store(self, load_path: str, db_type: str) -> None:
        """
        Load a previously saved vector store.
        
        Args:
            load_path: Path to load the vector store from
            db_type: Type of vector database ("FAISS" or "Chroma")
            
        LangChain Concept:
        - FAISS.load_local(): Loads saved FAISS index
        - Chroma(): Connects to existing Chroma database
        """
        try:
            if self.embeddings is None:
                raise ValueError("Embeddings not initialized. Call initialize_embeddings first.")
            
            if db_type == "FAISS":
                # Load FAISS index from disk
                self.vector_store = FAISS.load_local(
                    load_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True  # Required for pickle loading
                )
                
            elif db_type == "Chroma":
                # Connect to existing Chroma database
                self.vector_store = Chroma(
                    persist_directory=load_path,
                    embedding_function=self.embeddings
                )
            
            self.current_db_type = db_type
            print(f"✓ Vector store loaded from {load_path}")
            
        except Exception as e:
            raise Exception(f"Error loading vector store: {str(e)}")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform similarity search to retrieve relevant documents.
        
        Args:
            query: Search query text
            k: Number of most similar documents to retrieve
            
        Returns:
            List of most similar Document objects
            
        LangChain Concept:
        - similarity_search(): Core method for semantic retrieval
        - Automatically:
          1. Embeds the query using the same embedding model
          2. Performs vector similarity search (usually cosine similarity)
          3. Returns top-k most similar documents
        """
        try:
            if self.vector_store is None:
                raise ValueError("No vector store available. Create or load one first.")
            
            # LangChain similarity_search: Semantic search in vector store
            # Returns List[Document] sorted by relevance (most similar first)
            results = self.vector_store.similarity_search(query, k=k)
            
            return results
            
        except Exception as e:
            raise Exception(f"Error performing similarity search: {str(e)}")
    
    def similarity_search_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """
        Perform similarity search and return documents with relevance scores.
        
        Args:
            query: Search query text
            k: Number of most similar documents to retrieve
            
        Returns:
            List of tuples (Document, similarity_score)
            
        LangChain Concept:
        - similarity_search_with_score(): Returns documents with similarity scores
        - Scores indicate how similar each document is to the query
        - Lower scores = more similar (for L2 distance)
        - Higher scores = more similar (for cosine similarity)
        """
        try:
            if self.vector_store is None:
                raise ValueError("No vector store available. Create or load one first.")
            
            # Get results with relevance scores
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            return results
            
        except Exception as e:
            raise Exception(f"Error performing similarity search with scores: {str(e)}")
    
    def get_vector_store_info(self) -> dict:
        """
        Get information about the current vector store.
        
        Returns:
            Dictionary with vector store information
        """
        info = {
            "embedding_model": self.current_model,
            "database_type": self.current_db_type,
            "is_initialized": self.vector_store is not None
        }
        
        return info
