# Semantic Search Engine - Usage Guide

## Quick Start

### Option 1: Using the run script (Recommended)
```bash
chmod +x run.sh
./run.sh
```

### Option 2: Manual setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirement.txt

# Run application
python app/main.py
```

## Using the Application

### Step 1: Select Data Source
1. Click "Select Directory" to choose a folder with PDF files
2. Click "Select Single File" to choose one PDF
3. Click "Use Sample Data" to use pre-loaded documents

The GUI will display document statistics.

### Step 2: Configure Pipeline
1. **Select Embedding Model:**
   - `all-MiniLM-L6-v2`: Fast, good quality (recommended for testing)
   - `all-mpnet-base-v2`: Best quality, slower
   - `multi-qa-MiniLM-L6-cos-v1`: Optimized for Q&A
   - `paraphrase-multilingual-MiniLM`: Supports multiple languages

2. **Select Vector Database:**
   - `FAISS`: Fast in-memory search (recommended for small datasets)
   - `Chroma`: Persistent storage (better for production)

3. Click "⚙ Setup Pipeline" and wait for completion

### Step 3: Perform Searches
1. Enter your query in natural language
2. Set top-k value (number of results)
3. Click "🔎 Search"
4. View ranked results with relevance scores

## Running Evaluations

### Quick Test
```bash
python experiments/evaluate.py --mode quick
```

### Comprehensive Evaluation (Task 4)
```bash
python experiments/evaluate.py --mode full
```

Results are saved to `experiments/evaluation_results.json`

## Project Structure

```
├── app/
│   ├── __init__.py              # Package init
│   ├── config.py                # Configuration
│   ├── document_processor.py    # Document loading (LangChain)
│   ├── vector_store_manager.py  # Embeddings & vector stores
│   ├── retrieval_engine.py      # Search pipeline
│   ├── gui.py                   # User interface
│   └── main.py                  # Entry point
├── data/                        # Sample PDF documents
├── experiments/
│   ├── evaluate.py              # Evaluation scripts
│   └── report/
│       └── report_template.md   # Evaluation report
├── Vector_Store/                # Persistent vector stores
├── embeddings/                  # Cached embedding models
├── requirement.txt              # Dependencies
└── run.sh                       # Run script

```

## LangChain Components Used

### 1. Document Loaders
- `PyPDFLoader`: Load individual PDFs
- `DirectoryLoader`: Batch load multiple files

### 2. Text Splitters
- `RecursiveCharacterTextSplitter`: Intelligent text chunking

### 3. Embeddings
- `HuggingFaceEmbeddings`: Sentence transformer models

### 4. Vector Stores
- `FAISS`: Facebook AI Similarity Search
- `Chroma`: Persistent embedding database

### 5. Retrieval
- `similarity_search()`: Semantic search
- `similarity_search_with_score()`: Search with relevance scores

## Troubleshooting

### Issue: Model download is slow
**Solution:** First-time model download can take time. Models are cached for future use.

### Issue: Memory error
**Solution:** Use a smaller embedding model (all-MiniLM-L6-v2) or process fewer documents.

### Issue: Import errors
**Solution:** Ensure all dependencies are installed:
```bash
pip install --upgrade -r requirement.txt
```

### Issue: tkinter not found
**Solution:** Install tkinter:
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# macOS
brew install python-tk
```

## Sample Queries to Try

1. "What are the academic rules and regulations?"
2. "How do I apply for financial assistance?"
3. "What is the grading policy?"
4. "Tell me about Final Year Project requirements"
5. "What are the PhD admission requirements?"
6. "How can students appeal their grades?"
7. "What is the attendance policy?"
8. "Explain the student code of conduct"

## Customization

### Adding Your Own Documents
1. Place PDF files in a folder
2. Use "Select Directory" in the GUI
3. Setup pipeline with your preferred configuration

### Changing Chunk Size
Edit `app/config.py`:
```python
CHUNK_SIZE = 1000      # Adjust as needed
CHUNK_OVERLAP = 200    # Adjust overlap
```

### Adding New Embedding Models
Edit `app/config.py` and add to `EMBEDDING_MODELS` dictionary.

## Performance Tips

1. **For faster setup:** Use `all-MiniLM-L6-v2` model
2. **For better accuracy:** Use `all-mpnet-base-v2` model
3. **For large datasets:** Use Chroma instead of FAISS
4. **For repeated testing:** Models are cached after first download

## System Requirements

- **Python:** 3.8 or higher
- **RAM:** 4GB minimum, 8GB recommended
- **Disk Space:** 2GB for models and data
- **Internet:** Required for first-time model download

## Academic Integrity

This is an individual assignment for CS-4015 Agentic AI.
Do not share or copy this code.

---

**For questions or issues, refer to the assignment instructions or contact the course instructor.**
