"""
Evaluation Script for Task 4: Retrieval Evaluation and Analysis
This script tests the system with multiple queries and analyzes retrieval quality.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retrieval_engine import RetrievalEngine
from app.config import DATA_DIR, EMBEDDING_MODELS, VECTOR_DATABASES
import json
from datetime import datetime


# Test queries for evaluation
TEST_QUERIES = [
    "What are the academic rules and regulations?",
    "How do I apply for financial assistance?",
    "What is the grading policy?",
    "Tell me about the Final Year Project requirements",
    "What are the PhD admission requirements?",
    "How can students appeal their grades?",
    "What is the attendance policy?",
    "Explain the student code of conduct",
    "What are the faculty responsibilities?",
    "How to submit a thesis for MS program?"
]


def run_comprehensive_evaluation():
    """
    Run comprehensive evaluation across multiple configurations.
    This addresses Task 4 requirements.
    """
    print("=" * 80)
    print("TASK 4: RETRIEVAL EVALUATION AND ANALYSIS")
    print("=" * 80)
    print("\nTesting system with multiple queries, datasets, and embedding models...")
    print()
    
    results_log = {
        "timestamp": datetime.now().isoformat(),
        "evaluations": []
    }
    
    # Test with different embedding models
    models_to_test = list(EMBEDDING_MODELS.keys())[:2]  # Test first 2 models
    
    for model_key in models_to_test:
        print("\n" + "=" * 80)
        print(f"TESTING WITH MODEL: {model_key}")
        print(f"Description: {EMBEDDING_MODELS[model_key]['description']}")
        print("=" * 80)
        
        for db_type in VECTOR_DATABASES:
            print(f"\n>>> Testing with {db_type} vector database...")
            
            try:
                # Initialize engine
                engine = RetrievalEngine()
                
                # Setup pipeline
                stats = engine.setup_pipeline(
                    data_source=DATA_DIR,
                    embedding_model=model_key,
                    vector_db=db_type,
                    is_directory=True
                )
                
                # Run evaluation
                eval_results = engine.evaluate_retrieval(
                    test_queries=TEST_QUERIES[:5],  # Use first 5 queries
                    top_k=3
                )
                
                # Store results
                config_result = {
                    "embedding_model": model_key,
                    "vector_database": db_type,
                    "pipeline_stats": stats,
                    "evaluation": eval_results
                }
                
                results_log["evaluations"].append(config_result)
                
                print(f"\n✓ Evaluation complete for {model_key} + {db_type}")
                
            except Exception as e:
                print(f"\n✗ Error during evaluation: {str(e)}")
                continue
    
    # Save results
    output_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "experiments",
        "evaluation_results.json"
    )
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results_log, f, indent=2)
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_file}")
    print("\nKey Findings Summary:")
    print(f"  • Tested {len(models_to_test)} embedding models")
    print(f"  • Tested {len(VECTOR_DATABASES)} vector databases")
    print(f"  • Evaluated {len(TEST_QUERIES[:5])} test queries")
    print(f"  • Total configurations tested: {len(results_log['evaluations'])}")
    
    return results_log


def quick_test():
    """
    Quick test for basic functionality.
    """
    print("=" * 80)
    print("QUICK FUNCTIONALITY TEST")
    print("=" * 80)
    
    try:
        engine = RetrievalEngine()
        
        # Setup with default configuration
        print("\nSetting up pipeline with sample data...")
        stats = engine.setup_pipeline(
            data_source=DATA_DIR,
            embedding_model="all-MiniLM-L6-v2",
            vector_db="FAISS",
            is_directory=True
        )
        
        print(f"\n✓ Pipeline ready!")
        print(f"  Documents: {stats['documents_loaded']}")
        print(f"  Chunks: {stats['chunks_created']}")
        
        # Test a single query
        print("\n" + "-" * 80)
        print("Testing sample query...")
        test_query = "What are the academic rules?"
        results = engine.search(test_query, top_k=3, return_scores=True)
        
        print(f"\nQuery: '{test_query}'")
        print(f"Retrieved {len(results)} results:\n")
        
        for result in results:
            print(f"[{result['rank']}] {os.path.basename(result['source'])}")
            print(f"    Score: {result['relevance_score']:.4f}")
            print(f"    Preview: {result['content'][:100]}...")
            print()
        
        print("✓ Quick test successful!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluation script for semantic search")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick",
                       help="Test mode: 'quick' for basic test, 'full' for comprehensive evaluation")
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        quick_test()
    else:
        run_comprehensive_evaluation()
