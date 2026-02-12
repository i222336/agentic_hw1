"""
Configuration file for the Semantic Search Module.
Contains all constants, available models, and settings.
"""

import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "Vector_Store")
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")

# Available Hugging Face Embedding Models
# These models are popular for semantic search and retrieval tasks
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "Fast and efficient, 384 dimensions",
        "dimension": 384
    },
    "all-mpnet-base-v2": {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "description": "High quality, 768 dimensions",
        "dimension": 768
    },
    "multi-qa-MiniLM-L6-cos-v1": {
        "name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "description": "Optimized for Q&A, 384 dimensions",
        "dimension": 384
    },
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "description": "Multilingual support, 384 dimensions",
        "dimension": 384
    }
}

# Vector Database Options
VECTOR_DATABASES = ["FAISS", "Chroma"]

# Document Processing Settings
CHUNK_SIZE = 1000  # Size of text chunks for splitting documents
CHUNK_OVERLAP = 200  # Overlap between chunks to maintain context

# Default Retrieval Settings
DEFAULT_TOP_K = 5  # Default number of documents to retrieve
MAX_TOP_K = 20  # Maximum allowed top-k value

# GUI Settings
WINDOW_TITLE = "Semantic Search Engine - AI Research Assistant"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
DARK_BG = "#1e1e1e"  # Dark background
DARK_FG = "#ffffff"  # White foreground
DARK_BUTTON_BG = "#2d2d2d"  # Button background
DARK_ENTRY_BG = "#2d2d2d"  # Entry field background
ACCENT_COLOR = "#007acc"  # Blue accent color

