"""
Retrieval Engine Module
Orchestrates the semantic search process.

Key LangChain Concepts:
1. Orchestration: Combines document processing, embeddings, and vector stores
2. Retrieval: Core semantic search functionality
"""

from typing import List, Dict, Optional
from langchain_core.documents import Document
from app.document_processor import DocumentProcessor
from app.vector_store_manager import VectorStoreManager
from app.config import DEFAULT_TOP_K


class RetrievalEngine:
    """
    Main engine that orchestrates the semantic search pipeline.
    
    Pipeline Flow:
    1. Load documents (DocumentProcessor)
    2. Create embeddings (VectorStoreManager)
    3. Store in vector database (VectorStoreManager)
    4. Retrieve similar documents for queries
    """
    
    def __init__(self):
        """Initialize the retrieval engine with its components."""
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.documents = None
        self.split_documents = None
        self.is_ready = False
    
    def setup_pipeline(self, data_source: str, embedding_model: str, 
                      vector_db: str, is_directory: bool = True) -> Dict:
        """
        Setup the complete semantic search pipeline.
        
        Args:
            data_source: Path to documents (file or directory)
            embedding_model: Key for embedding model from config
            vector_db: Type of vector database ("FAISS" or "Chroma")
            is_directory: Whether data_source is a directory or single file
            
        Returns:
            Dictionary with pipeline statistics
            
        LangChain Pipeline Steps:
        1. Load documents using DocumentProcessor (LangChain loaders)
        2. Split into chunks using TextSplitter
        3. Initialize embeddings using HuggingFaceEmbeddings
        4. Create vector store and embed all documents
        """
        try:
            print("=" * 60)
            print("SETTING UP SEMANTIC SEARCH PIPELINE")
            print("=" * 60)
            
            # Step 1: Load documents using LangChain loaders
            print("\n[1/4] Loading documents...")
            if is_directory:
                self.documents, doc_stats = self.document_processor.load_documents_from_directory(
                    data_source
                )
            else:
                self.documents, doc_stats = self.document_processor.load_single_document(
                    data_source
                )
            
            print(f"  ✓ Loaded {doc_stats['total_documents']} documents")
            print(f"  ✓ Total pages: {doc_stats['total_pages']}")
            
            # Step 2: Split documents using LangChain TextSplitter
            print("\n[2/4] Splitting documents into chunks...")
            self.split_documents = self.document_processor.split_documents(self.documents)
            print(f"  ✓ Created {len(self.split_documents)} text chunks")
            
            # Step 3: Initialize embeddings using LangChain HuggingFaceEmbeddings
            print("\n[3/4] Initializing embedding model...")
            self.vector_store_manager.initialize_embeddings(embedding_model)
            
            # Step 4: Create vector store using LangChain FAISS/Chroma
            print("\n[4/4] Creating vector store and embedding documents...")
            self.vector_store_manager.create_vector_store(
                documents=self.split_documents,
                db_type=vector_db
            )
            
            self.is_ready = True
            
            # Compile pipeline statistics
            pipeline_stats = {
                "documents_loaded": doc_stats['total_documents'],
                "total_pages": doc_stats['total_pages'],
                "chunks_created": len(self.split_documents),
                "embedding_model": embedding_model,
                "vector_database": vector_db,
                "status": "ready"
            }
            
            print("\n" + "=" * 60)
            print("PIPELINE SETUP COMPLETE!")
            print("=" * 60)
            
            return pipeline_stats
            
        except Exception as e:
            self.is_ready = False
            raise Exception(f"Error setting up pipeline: {str(e)}")
    
    def search(self, query: str, top_k: int = DEFAULT_TOP_K, 
               return_scores: bool = True) -> List[Dict]:
        """
        Perform semantic search for a query.
        
        Args:
            query: Search query text
            top_k: Number of most relevant documents to retrieve
            return_scores: Whether to include relevance scores
            
        Returns:
            List of dictionaries containing retrieved documents and metadata
            
        LangChain Retrieval:
        - Uses vector_store.similarity_search_with_score()
        - Returns documents ranked by semantic similarity
        - Scores indicate relevance (implementation depends on vector store)
        """
        try:
            if not self.is_ready:
                raise ValueError("Pipeline not ready. Run setup_pipeline first.")
            
            print(f"\nSearching for: '{query}' (top-{top_k})")
            
            # Perform semantic search using LangChain vector store
            if return_scores:
                results = self.vector_store_manager.similarity_search_with_score(
                    query, k=top_k
                )
                
                # Format results with scores
                formatted_results = []
                for idx, (doc, score) in enumerate(results, 1):
                    formatted_results.append({
                        "rank": idx,
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(score),
                        "relevance_score": float(score),
                        "source": doc.metadata.get("source", "Unknown")
                    })
            else:
                results = self.vector_store_manager.similarity_search(query, k=top_k)
                
                # Format results without scores
                formatted_results = []
                for idx, doc in enumerate(results, 1):
                    formatted_results.append({
                        "rank": idx,
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": None,
                        "source": doc.metadata.get("source", "Unknown")
                    })
            
            print(f"  ✓ Retrieved {len(formatted_results)} relevant documents")
            
            return formatted_results
            
        except Exception as e:
            raise Exception(f"Error performing search: {str(e)}")
    
    def get_pipeline_info(self) -> Dict:
        """
        Get information about the current pipeline state.
        
        Returns:
            Dictionary with pipeline information
        """
        info = {
            "is_ready": self.is_ready,
            "documents_loaded": len(self.documents) if self.documents else 0,
            "chunks_created": len(self.split_documents) if self.split_documents else 0,
        }
        
        # Add vector store info if available
        if self.is_ready:
            info.update(self.vector_store_manager.get_vector_store_info())
        
        return info
    
    def evaluate_retrieval(self, test_queries: List[str], top_k: int = 5) -> Dict:
        """
        Evaluate retrieval quality across multiple queries.
        
        Args:
            test_queries: List of test queries
            top_k: Number of documents to retrieve per query
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            if not self.is_ready:
                raise ValueError("Pipeline not ready. Run setup_pipeline first.")
            
            print("\n" + "=" * 60)
            print("RETRIEVAL EVALUATION")
            print("=" * 60)
            
            evaluation_results = {
                "total_queries": len(test_queries),
                "top_k": top_k,
                "queries_results": []
            }
            
            for idx, query in enumerate(test_queries, 1):
                print(f"\nQuery {idx}/{len(test_queries)}: {query}")
                
                results = self.search(query, top_k=top_k, return_scores=True)
                
                query_eval = {
                    "query": query,
                    "num_results": len(results),
                    "avg_score": sum(r['relevance_score'] for r in results) / len(results) if results else 0,
                    "top_sources": [r['source'] for r in results[:3]]
                }
                
                evaluation_results["queries_results"].append(query_eval)
                
                # Print top result
                if results:
                    print(f"  Top result: {results[0]['source']}")
                    print(f"  Score: {results[0]['relevance_score']:.4f}")
            
            print("\n" + "=" * 60)
            print("EVALUATION COMPLETE")
            print("=" * 60)
            
            return evaluation_results
            
        except Exception as e:
            raise Exception(f"Error during evaluation: {str(e)}")
