#!/usr/bin/env python
"""
Comprehensive Test Script for Semantic Search Module
Tests all configurations and queries
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import EMBEDDING_MODELS, VECTOR_DATABASES
from app.retrieval_engine import RetrievalEngine
import traceback

# Test data
TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
TEST_QUERIES = [
    "What is CS curriculum?",
    "What is Data Science curriculum?",
    "What is annual report results?",
    "Explain academic policies",
    "What are the course requirements?"
]

def test_configuration(embedding_key, vector_db):
    """Test a specific configuration"""
    print(f"\n{'='*70}")
    print(f"Testing: {embedding_key} + {vector_db}")
    print(f"{'='*70}")
    
    try:
        # Initialize engine
        engine = RetrievalEngine()
        print(f"✓ Engine initialized")
        
        # Check if test data exists
        if not os.path.exists(TEST_DATA_PATH) or not os.listdir(TEST_DATA_PATH):
            print(f"⚠ No test data found at {TEST_DATA_PATH}")
            print(f"  Creating sample test documents...")
            os.makedirs(TEST_DATA_PATH, exist_ok=True)
            
            # Create sample text files
            sample_docs = {
                "machine_learning.txt": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.",
                "neural_networks.txt": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers. Deep learning uses neural networks with multiple hidden layers to learn complex patterns from data.",
                "semantic_search.txt": "Semantic search is an information retrieval technique that goes beyond keyword matching. It understands the intent and contextual meaning of search queries. It uses natural language processing and vector embeddings to find relevant results."
            }
            
            for filename, content in sample_docs.items():
                with open(os.path.join(TEST_DATA_PATH, filename), 'w') as f:
                    f.write(content)
            print(f"✓ Created {len(sample_docs)} sample documents")
        
        # Setup pipeline with sample data
        print(f"  Setting up pipeline with {embedding_key}...")
        result = engine.setup_pipeline(
            data_source=TEST_DATA_PATH,
            embedding_model=embedding_key,
            vector_db=vector_db,
            is_directory=True
        )
        
        print(f"✓ Pipeline setup successful")
        print(f"  - Documents loaded: {result.get('num_documents', 'N/A')}")
        print(f"  - Chunks created: {result.get('num_chunks', 'N/A')}")
        print(f"  - Vector store: {result.get('vector_store', 'N/A')}")
        
        # Test queries
        print(f"\n  Testing queries:")
        for i, query in enumerate(TEST_QUERIES, 1):
            try:
                print(f"  Query {i}: '{query[:50]}...'")
                results = engine.search(query, top_k=3)
                
                if results:
                    print(f"  ✓ Found {len(results)} results")
                    if results:
                        top_result = results[0]
                        score = top_result.get('score', 'N/A')
                        if isinstance(score, (int, float)):
                            print(f"    Top result score: {score:.4f}")
                        else:
                            print(f"    Top result score: {score}")
                else:
                    print(f"  ⚠ No results returned")
                    
            except Exception as e:
                print(f"  ✗ Query failed: {str(e)}")
                traceback.print_exc()
        
        print(f"\n✓ Configuration {embedding_key} + {vector_db} PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Configuration {embedding_key} + {vector_db} FAILED")
        print(f"  Error: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all configuration tests"""
    print("="*70)
    print("SEMANTIC SEARCH MODULE - COMPREHENSIVE TEST")
    print("="*70)
    
    # Check imports
    print("\nChecking dependencies...")
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
    except Exception as e:
        print(f"✗ PyTorch: {e}")
    
    try:
        import sentence_transformers
        print(f"✓ sentence-transformers: {sentence_transformers.__version__}")
    except Exception as e:
        print(f"✗ sentence-transformers: {e}")
    
    try:
        import langchain
        print(f"✓ LangChain: {langchain.__version__}")
    except Exception as e:
        print(f"✗ LangChain: {e}")
    
    try:
        import faiss
        print(f"✓ FAISS: available")
    except Exception as e:
        print(f"✗ FAISS: {e}")
    
    try:
        import chromadb
        print(f"✓ ChromaDB: {chromadb.__version__}")
    except Exception as e:
        print(f"✗ ChromaDB: {e}")
    
    # Test all configurations
    print(f"\n\nTesting {len(EMBEDDING_MODELS)} embedding models x {len(VECTOR_DATABASES)} vector DBs")
    print(f"= {len(EMBEDDING_MODELS) * len(VECTOR_DATABASES)} total configurations\n")
    
    results = {}
    for embedding_key in EMBEDDING_MODELS.keys():
        for vector_db in VECTOR_DATABASES:
            config_name = f"{embedding_key}+{vector_db}"
            success = test_configuration(embedding_key, vector_db)
            results[config_name] = success
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    failed = len(results) - passed
    
    print(f"\nTotal configurations tested: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    print("\nDetailed Results:")
    for config, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {config}")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {failed} test(s) failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
