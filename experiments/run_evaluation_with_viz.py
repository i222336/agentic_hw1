"""
Comprehensive Evaluation with Metrics and Visualizations
Generates detailed analysis for the report.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retrieval_engine import RetrievalEngine
from app.config import DATA_DIR, EMBEDDING_MODELS, VECTOR_DATABASES
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style for plots
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

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


def run_comprehensive_evaluation_with_metrics():
    """
    Run comprehensive evaluation and generate visualizations.
    """
    print("=" * 80)
    print("COMPREHENSIVE EVALUATION WITH METRICS & VISUALIZATIONS")
    print("=" * 80)
    print()
    
    # Results storage
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "test_queries": TEST_QUERIES,
        "model_comparisons": [],
        "database_comparisons": [],
        "query_results": []
    }
    
    # Create output directories
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "experiments", "results"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Test each embedding model
    models_to_test = list(EMBEDDING_MODELS.keys())
    
    print("\n" + "=" * 80)
    print("PHASE 1: TESTING EMBEDDING MODELS")
    print("=" * 80)
    
    for model_idx, model_key in enumerate(models_to_test, 1):
        print(f"\n[{model_idx}/{len(models_to_test)}] Testing Model: {model_key}")
        print(f"Description: {EMBEDDING_MODELS[model_key]['description']}")
        print("-" * 80)
        
        # Test with FAISS (faster for comparison)
        try:
            start_time = time.time()
            
            # Initialize engine
            engine = RetrievalEngine()
            
            # Setup pipeline
            print("  • Setting up pipeline...")
            stats = engine.setup_pipeline(
                data_source=DATA_DIR,
                embedding_model=model_key,
                vector_db="FAISS",
                is_directory=True
            )
            
            setup_time = time.time() - start_time
            
            print(f"  • Setup completed in {setup_time:.2f}s")
            print(f"  • Documents: {stats['documents_loaded']}, Chunks: {stats['chunks_created']}")
            
            # Test queries
            query_times = []
            query_scores = []
            
            print("  • Running test queries...")
            for query in TEST_QUERIES[:5]:  # Test with first 5 queries
                query_start = time.time()
                results = engine.search(query, top_k=5, return_scores=True)
                query_time = time.time() - query_start
                
                query_times.append(query_time)
                if results:
                    avg_score = sum(r['relevance_score'] for r in results) / len(results)
                    query_scores.append(avg_score)
            
            # Store model results
            model_results = {
                "model": model_key,
                "dimensions": EMBEDDING_MODELS[model_key]['dimension'],
                "setup_time": setup_time,
                "avg_query_time": np.mean(query_times),
                "std_query_time": np.std(query_times),
                "avg_relevance_score": np.mean(query_scores),
                "documents_processed": stats['documents_loaded'],
                "chunks_created": stats['chunks_created']
            }
            
            all_results["model_comparisons"].append(model_results)
            
            print(f"  ✓ Model evaluation complete")
            print(f"    - Avg query time: {np.mean(query_times):.4f}s")
            print(f"    - Avg relevance score: {np.mean(query_scores):.4f}")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            continue
    
    print("\n" + "=" * 80)
    print("PHASE 2: COMPARING VECTOR DATABASES")
    print("=" * 80)
    
    # Compare FAISS vs Chroma with best model
    best_model = models_to_test[0]  # Use first model for DB comparison
    
    for db_type in VECTOR_DATABASES:
        print(f"\n Testing Vector Database: {db_type}")
        print("-" * 80)
        
        try:
            engine = RetrievalEngine()
            
            start_time = time.time()
            stats = engine.setup_pipeline(
                data_source=DATA_DIR,
                embedding_model=best_model,
                vector_db=db_type,
                is_directory=True
            )
            setup_time = time.time() - start_time
            
            # Test query performance
            query_times = []
            for query in TEST_QUERIES[:3]:
                query_start = time.time()
                results = engine.search(query, top_k=5, return_scores=True)
                query_times.append(time.time() - query_start)
            
            db_results = {
                "database": db_type,
                "model": best_model,
                "setup_time": setup_time,
                "avg_query_time": np.mean(query_times),
                "std_query_time": np.std(query_times)
            }
            
            all_results["database_comparisons"].append(db_results)
            
            print(f"  ✓ Database evaluation complete")
            print(f"    - Setup time: {setup_time:.2f}s")
            print(f"    - Avg query time: {np.mean(query_times):.4f}s")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            continue
    
    print("\n" + "=" * 80)
    print("PHASE 3: DETAILED QUERY ANALYSIS")
    print("=" * 80)
    
    # Detailed analysis with best configuration
    engine = RetrievalEngine()
    engine.setup_pipeline(
        data_source=DATA_DIR,
        embedding_model=best_model,
        vector_db="FAISS",
        is_directory=True
    )
    
    for query in TEST_QUERIES:
        print(f"\n• Query: {query[:60]}...")
        results = engine.search(query, top_k=5, return_scores=True)
        
        query_result = {
            "query": query,
            "num_results": len(results),
            "top_score": results[0]['relevance_score'] if results else 0,
            "avg_score": np.mean([r['relevance_score'] for r in results]) if results else 0,
            "top_source": os.path.basename(results[0]['source']) if results else "N/A",
            "all_scores": [r['relevance_score'] for r in results]
        }
        
        all_results["query_results"].append(query_result)
        print(f"  Top result: {query_result['top_source']}")
        print(f"  Score: {query_result['top_score']:.4f}")
    
    # Save results
    results_file = os.path.join(output_dir, "evaluation_metrics.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    generate_visualizations(all_results, output_dir)
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  • Metrics: evaluation_metrics.json")
    print(f"  • Visualizations: *.png files")
    
    return all_results


def generate_visualizations(results, output_dir):
    """Generate comprehensive visualizations for the report."""
    
    # 1. Model Performance Comparison
    if results["model_comparisons"]:
        print("\n• Generating model comparison charts...")
        
        df_models = pd.DataFrame(results["model_comparisons"])
        
        # Setup time comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Embedding Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Chart 1: Setup Time
        axes[0, 0].bar(df_models['model'], df_models['setup_time'], color='steelblue')
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].set_title('Pipeline Setup Time')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Chart 2: Query Time
        axes[0, 1].bar(df_models['model'], df_models['avg_query_time'], color='coral')
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].set_title('Average Query Time')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Chart 3: Relevance Scores
        axes[1, 0].bar(df_models['model'], df_models['avg_relevance_score'], color='mediumseagreen')
        axes[1, 0].set_xlabel('Model')
        axes[1, 0].set_ylabel('Relevance Score')
        axes[1, 0].set_title('Average Relevance Score (Lower = More Similar)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Chart 4: Dimensions vs Performance
        axes[1, 1].scatter(df_models['dimensions'], df_models['avg_query_time'], 
                          s=200, alpha=0.6, color='purple')
        for idx, row in df_models.iterrows():
            axes[1, 1].annotate(row['model'].split('-')[0], 
                               (row['dimensions'], row['avg_query_time']),
                               fontsize=8)
        axes[1, 1].set_xlabel('Embedding Dimensions')
        axes[1, 1].set_ylabel('Query Time (seconds)')
        axes[1, 1].set_title('Dimensions vs Query Performance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '1_model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Database Comparison
    if results["database_comparisons"]:
        print("• Generating database comparison charts...")
        
        df_db = pd.DataFrame(results["database_comparisons"])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Vector Database Performance Comparison', fontsize=16, fontweight='bold')
        
        # Setup time
        axes[0].bar(df_db['database'], df_db['setup_time'], color=['steelblue', 'coral'])
        axes[0].set_xlabel('Database')
        axes[0].set_ylabel('Time (seconds)')
        axes[0].set_title('Setup Time Comparison')
        
        # Query time
        axes[1].bar(df_db['database'], df_db['avg_query_time'], color=['steelblue', 'coral'])
        axes[1].set_xlabel('Database')
        axes[1].set_ylabel('Time (seconds)')
        axes[1].set_title('Average Query Time')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '2_database_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Query Performance Analysis
    if results["query_results"]:
        print("• Generating query analysis charts...")
        
        df_queries = pd.DataFrame(results["query_results"])
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Query-wise Retrieval Performance', fontsize=16, fontweight='bold')
        
        # Top scores by query
        query_labels = [q[:40] + '...' if len(q) > 40 else q for q in df_queries['query']]
        
        axes[0].barh(query_labels, df_queries['top_score'], color='steelblue')
        axes[0].set_xlabel('Relevance Score (Lower = Better)')
        axes[0].set_ylabel('Query')
        axes[0].set_title('Top Result Relevance Score per Query')
        axes[0].invert_yaxis()
        
        # Average scores
        axes[1].barh(query_labels, df_queries['avg_score'], color='coral')
        axes[1].set_xlabel('Average Relevance Score')
        axes[1].set_ylabel('Query')
        axes[1].set_title('Average Top-5 Relevance Scores')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '3_query_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Score Distribution
    if results["query_results"]:
        print("• Generating score distribution chart...")
        
        all_scores = []
        for qr in results["query_results"]:
            all_scores.extend(qr["all_scores"])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Relevance Score Distribution', fontsize=16, fontweight='bold')
        
        # Histogram
        axes[0].hist(all_scores, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Relevance Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of All Relevance Scores')
        axes[0].axvline(np.mean(all_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(all_scores):.4f}')
        axes[0].legend()
        
        # Box plot
        axes[1].boxplot(all_scores, vert=True)
        axes[1].set_ylabel('Relevance Score')
        axes[1].set_title('Score Distribution Statistics')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '4_score_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Summary Statistics Table
    print("• Generating summary statistics...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    summary_data = []
    summary_data.append(['Metric', 'Value'])
    summary_data.append(['─' * 40, '─' * 20])
    
    if results["model_comparisons"]:
        df_models = pd.DataFrame(results["model_comparisons"])
        summary_data.append(['Total Models Tested', str(len(df_models))])
        summary_data.append(['Fastest Setup', df_models.loc[df_models['setup_time'].idxmin(), 'model']])
        summary_data.append(['Fastest Query', df_models.loc[df_models['avg_query_time'].idxmin(), 'model']])
        summary_data.append(['Best Relevance', df_models.loc[df_models['avg_relevance_score'].idxmin(), 'model']])
    
    if results["query_results"]:
        summary_data.append(['─' * 40, '─' * 20])
        summary_data.append(['Total Queries Tested', str(len(results["query_results"]))])
        summary_data.append(['Avg Relevance Score', f"{np.mean([q['avg_score'] for q in results['query_results']]):.4f}"])
        summary_data.append(['Best Query Score', f"{min([q['top_score'] for q in results['query_results']]):.4f}"])
        summary_data.append(['Worst Query Score', f"{max([q['top_score'] for q in results['query_results']]):.4f}"])
    
    table = ax.table(cellText=summary_data, cellLoc='left', loc='center',
                    colWidths=[0.7, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('steelblue')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Evaluation Summary Statistics', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(output_dir, '5_summary_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ All visualizations generated successfully!")


if __name__ == "__main__":
    results = run_comprehensive_evaluation_with_metrics()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if results["model_comparisons"]:
        print("\nModel Performance:")
        for model in results["model_comparisons"]:
            print(f"  • {model['model']}: {model['avg_query_time']:.4f}s avg query time")
    
    if results["database_comparisons"]:
        print("\nDatabase Performance:")
        for db in results["database_comparisons"]:
            print(f"  • {db['database']}: {db['avg_query_time']:.4f}s avg query time")
    
    print("\n" + "=" * 80)
