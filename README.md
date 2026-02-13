# Semantic Search Module - Phase 1 Submission

**Student ID:** 22I-2336
**Section:** CS-8B
**Course:** CS-4015 Agentic AI  
**Assignment:** Homework 1 - Phase 1

## Project Overview

GUI-based semantic search system for academic documents using LangChain, sentence-transformers, and vector databases. Allows users to upload documents, select embedding models and vector stores, then run semantic queries to retrieve relevant information.

## Installation

1. **Activate Virtual Environment:**
```bash
source ~/agentic_venv/bin/activate
```

2. **Install Dependencies (if needed):**
```bash
pip install -r requirement.txt
```

## Usage

### Run GUI Application
```bash
python app/main.py
```

### Run Tests
```bash
# Test all configurations (4 models x 2 databases)
python test_all_configs.py

```

## Project Structure

```
.
├── app/                        # Main application code
│   ├── main.py                # Entry point
│   ├── gui.py                 # GUI interface
│   ├── config.py              # Configuration
│   ├── document_processor.py  # Document loading & chunking
│   ├── vector_store_manager.py # Embeddings & vector stores
│   └── retrieval_engine.py    # Search pipeline
├── data/                       # Document storage (user-provided)
├── Vector_Store/              # Vector database storage
├── test_all_configs.py        # Comprehensive testing script
├── quick_test.py              # Quick test script
├── SUBMISSION_REPORT.txt      # Assignment report (2-3 pages)
└── requirement.txt            # Python dependencies
```

## Features Implemented

### Task 1: GUI-Based Data Selection
- Browse and select document directories
- View dataset information (document count, pages, chunks)
- No hardcoded paths

### Task 2: Embedding and Vector Store Configuration
- 4 embedding model options (HuggingFace)
- 2 vector database options (FAISS, Chroma)
- Automatic embedding generation and storage

### Task 3: Semantic Retrieval
- Query input interface
- Configurable top-k results (1-10)
- Results display with relevance scores and metadata

### Task 4: Retrieval Evaluation
- Tested 8 configurations (4 models x 2 databases)
- 5 academic domain queries tested
- 100% success rate
- Detailed performance analysis in report

## Test Results Summary

**Dataset:** 10 PDFs, 587 pages, 1583 chunks

**Best Performing Model:** paraphrase-multilingual-MiniLM-L12-v2
- Best score: 0.6637 L2 distance (academic policies query)
- Recommended for production use

**Most Consistent Model:** all-MiniLM-L6-v2
- Perfect score parity between FAISS and Chroma
- Best for resource-constrained environments

**Vector Database:** FAISS outperformed ChromaDB in consistency
- FAISS: 100% success, more stable scores
- ChromaDB: 100% success after fixes, occasional variance

## Key Files for Evaluation

1. **SUBMISSION_REPORT.txt** - Assignment report (2-3 pages)
2. **app/main.py** - GUI application entry point
3. **app/gui.py** - GUI implementation
4. **test_all_configs.py** - Comprehensive testing
5. **TEST_RESULTS_FINAL.txt** - Detailed test results

## Technologies Used

- **Framework:** LangChain 1.2.10
- **Embeddings:** sentence-transformers 5.2.2
- **Vector DBs:** FAISS-CPU 1.7.4, ChromaDB 0.4.22
- **Deep Learning:** PyTorch 2.10.0+cpu
- **GUI:** Tkinter with custom dark theme

## Deliverables Checklist

- [x] GUI-based application (app/main.py)
- [x] Complete source code (app/ directory)
- [x] Short report 2-3 pages (SUBMISSION_REPORT.txt)
- [x] Test results and evaluation
- [x] Working installation and setup

