# 🚀 Quick Setup and Testing Guide

## Problem with Current Installation
The sentence-transformers package is having connectivity issues during installation. Here's how to fix it:

## ✅ Step-by-Step Solution

### Option 1: Use the Automated Script (Recommended)
```bash
cd /home/hp/Desktop/agentic_hw_1/hw1-phase-1-semantic-search-module-i220467-main
chmod +x setup_and_test.sh
./setup_and_test.sh
```

This script will:
1. Create/verify virtual environment
2. Install all dependencies (including PyTorch CPU version for faster download)
3. Verify installations
4. Run evaluation (your choice: quick or full)

### Option 2: Manual Installation
```bash
cd /home/hp/Desktop/agentic_hw_1/hw1-phase-1-semantic-search-module-i220467-main

# Activate venv
source venv/bin/activate

# Install PyTorch CPU version (much smaller, faster to download)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Now install sentence-transformers
pip install sentence-transformers scikit-learn scipy

# Install remaining packages
pip install transformers langchain-huggingface

# Verify
python -c "import sentence_transformers; print(sentence_transformers.__version__)"
```

### Option 3: Quick Fix for Existing Installation
If the background installation is still running, wait for it or:
```bash
cd /home/hp/Desktop/agentic_hw_1/hw1-phase-1-semantic-search-module-i220467-main
pkill -f "pip install"  # Kill any stuck pip processes
source venv/bin/activate
pip install --upgrade --force-reinstall sentence-transformers
```

## 🧪 Testing the System

### Quick Test (2-3 minutes)
```bash
source venv/bin/activate
python experiments/evaluate.py --mode quick
```

### Full Evaluation with Visualizations (10-15 minutes)
```bash
source venv/bin/activate
python experiments/run_evaluation_with_viz.py
```

This will generate:
- `experiments/results/evaluation_metrics.json` - Detailed metrics
- `experiments/results/*.png` - Visualization charts
  - Model comparison charts
  - Database performance comparison
  - Query performance analysis
  - Score distribution plots
  - Summary statistics

## 🖥️ Running the GUI Application
```bash
source venv/bin/activate
python app/main.py
```

## 📊 What Gets Generated

### 1. Evaluation Metrics (JSON)
- Model performance comparison
- Database speed comparison
- Query-wise results
- Relevance scores

### 2. Visualizations (PNG files)
- **1_model_comparison.png**: 4 charts comparing embedding models
  - Setup time
  - Query time
  - Relevance scores
  - Dimensions vs performance
  
- **2_database_comparison.png**: FAISS vs Chroma performance
  
- **3_query_performance.png**: Per-query analysis
  
- **4_score_distribution.png**: Statistical distribution
  
- **5_summary_statistics.png**: Overall metrics table

### 3. Report Template
Already created at: `experiments/report/report_template.md`
- Fill in experimental results after running evaluation
- Includes sections for observations and analysis

## 🔧 Troubleshooting

### If "No module named sentence_transformers"
```bash
# Check which Python is being used
which python
python --version

# Make sure you're in venv
source venv/bin/activate

# Reinstall
pip uninstall sentence-transformers -y
pip install sentence-transformers
```

### If models download slowly
First-time model downloads can take 5-10 minutes. They're cached afterward.
Progress bars will show download status.

### If memory errors occur
- Use smaller model: `all-MiniLM-L6-v2` (384 dimensions)
- Process fewer documents
- Close other applications

## 📝 Next Steps After Running

1. Check generated visualizations in `experiments/results/`
2. Open `evaluation_metrics.json` for detailed numbers
3. Complete the report template at `experiments/report/report_template.md`
4. Fill in:
   - Experimental results sections (marked with [Fill in...])
   - Observations about which models worked best
   - Performance comparisons
   - Quality analysis

## 🎯 Assignment Completion Checklist

- [ ] All dependencies installed
- [ ] Quick test runs successfully
- [ ] Full evaluation completed with visualizations
- [ ] GUI application tested
- [ ] Report template filled with actual results
- [ ] All visualizations reviewed and analyzed
- [ ] Code committed to repository

## 📦 Files Created

```
├── app/
│   ├── config.py               ✓ Configuration
│   ├── document_processor.py   ✓ LangChain document loading
│   ├── vector_store_manager.py ✓ Embeddings & vector stores
│   ├── retrieval_engine.py     ✓ Search pipeline
│   ├── gui.py                  ✓ Dark-themed interface
│   └── main.py                 ✓ Entry point
├── experiments/
│   ├── evaluate.py             ✓ Basic evaluation
│   ├── run_evaluation_with_viz.py  ✓ Full evaluation with charts
│   ├── results/                ← Generated metrics & visualizations
│   └── report/
│       └── report_template.md  ✓ Report to complete
├── requirement.txt             ✓ Dependencies
├── setup_and_test.sh           ✓ Automated setup script
└── SETUP_GUIDE.md              ✓ This guide
```

---

**Need help?** All code includes extensive comments explaining LangChain concepts and implementation details.
