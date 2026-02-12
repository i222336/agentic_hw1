#!/bin/bash

# Complete Setup and Test Script for Semantic Search Engine
# Run this script to set up everything and generate the complete report

echo "========================================"
echo "Semantic Search Engine - Complete Setup"
echo "========================================"
echo ""

PROJECT_DIR="/home/hp/Desktop/agentic_hw_1/hw1-phase-1-semantic-search-module-i220467-main"
cd "$PROJECT_DIR"

# Step 1: Create/activate virtual environment
echo "[1/5] Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  ✓ Virtual environment created"
else
    echo "  ✓ Virtual environment exists"
fi

# Activate venv
source venv/bin/activate
echo "  ✓ Virtual environment activated"

# Step 2: Upgrade pip
echo ""
echo "[2/5] Upgrading pip..."
pip install --upgrade pip setuptools wheel -q
echo "  ✓ Pip upgraded"

# Step 3: Install dependencies in stages
echo ""
echo "[3/5] Installing dependencies (this may take several minutes)..."

echo "  • Installing core packages..."
pip install -q numpy pandas matplotlib seaborn plotly

echo "  • Installing PyTorch (CPU version - lighter)..."
pip install -q torch --index-url https://download.pytorch.org/whl/cpu

echo "  • Installing transformers and sentence-transformers..."
pip install -q transformers sentence-transformers

echo "  • Installing LangChain packages..."
pip install -q langchain langchain-community langchain-core langchain-text-splitters langchain-huggingface

echo "  • Installing vector databases..."
pip install -q faiss-cpu chromadb

echo "  • Installing PDF processing..."
pip install -q pypdf pypdf2

echo "  ✓ All dependencies installed"

# Step 4: Verify installation
echo ""
echo "[4/5] Verifying installation..."
python -c "import sentence_transformers; print(f'  ✓ sentence-transformers: {sentence_transformers.__version__}')"
python -c "import langchain; print(f'  ✓ langchain: {langchain.__version__}')"
python -c "import faiss; print('  ✓ FAISS: OK')"
python -c "import chromadb; print(f'  ✓ ChromaDB: OK')"

# Step 5: Run evaluation
echo ""
echo "[5/5] Running evaluation..."
echo ""
echo "Choose evaluation mode:"
echo "  1) Quick test (fast, basic functionality)"
echo "  2) Full evaluation (comprehensive, with visualizations)"
echo ""
read -p "Enter choice (1 or 2): " choice

if [ "$choice" == "1" ]; then
    echo ""
    echo "Running quick test..."
    python experiments/evaluate.py --mode quick
elif [ "$choice" == "2" ]; then
    echo ""
    echo "Running full evaluation with visualizations..."
    python experiments/run_evaluation_with_viz.py
else
    echo "Invalid choice. Running quick test..."
    python experiments/evaluate.py --mode quick
fi

echo ""
echo "========================================"
echo "Setup and evaluation complete!"
echo "========================================"
echo ""
echo "To run the GUI application:"
echo "  source venv/bin/activate && python app/main.py"
echo ""
echo "To run evaluation again:"
echo "  source venv/bin/activate && python experiments/run_evaluation_with_viz.py"
echo ""

deactivate
