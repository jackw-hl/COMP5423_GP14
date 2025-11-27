"""
Retrieval Module for RAG System
Implements sparse, dense, and multi-vector retrieval methods
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import pickle


@dataclass
class RetrievalResult:
    """Standard retrieval result format"""
    doc_id: str
    score: float
    text: str
    rank: int


class BaseRetriever:
    """Base class for all retrievers"""
    
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        raise NotImplementedError
    
    def batch_retrieve(self, queries: List[str], top_k: int = 10) -> List[List[RetrievalResult]]:
        """Batch retrieval for multiple queries"""
        return [self.retrieve(q, top_k) for q in queries]


class SparseRetriever(BaseRetriever):
    """Sparse retrieval using BM25 or TF-IDF"""
    
    def __init__(self, method: str = "bm25", data_dir: str = "data"):
        """
        Initialize sparse retriever
        
        Args:
            method: "bm25" or "tfidf"
            data_dir: Directory containing collection.jsonl
        """
        self.method = method
        self.data_dir = Path(data_dir)
        
        # Load document collection
        self.documents = self._load_collection()
        self.doc_ids = list(self.documents.keys())
        self.doc_texts = [self.documents[doc_id] for doc_id in self.doc_ids]
        
        # Initialize retriever
        self._init_retriever()
    
    def _load_collection(self) -> Dict[str, str]:
        """Load document collection from JSONL"""
        collection_path = self.data_dir / "collection.jsonl"
        documents = {}
        
        with open(collection_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                documents[doc['id']] = doc['text']
        
        print(f"Loaded {len(documents)} documents from collection")
        return documents
    
    def _init_retriever(self):
        """Initialize BM25 or TF-IDF"""
        try:
            from rank_bm25 import BM25Okapi
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            raise ImportError("Please install: pip install rank-bm25 scikit-learn")
        
        if self.method == "bm25":
            # Tokenize documents for BM25
            tokenized_docs = [doc.lower().split() for doc in self.doc_texts]
            self.retriever = BM25Okapi(tokenized_docs)
            print("BM25 retriever initialized")
            
        elif self.method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words='english'
            )
            self.doc_vectors = self.vectorizer.fit_transform(self.doc_texts)
            print("TF-IDF retriever initialized")
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve top-k documents for query"""
        if self.method == "bm25":
            tokenized_query = query.lower().split()
            scores = self.retriever.get_scores(tokenized_query)
            
        elif self.method == "tfidf":
            from sklearn.metrics.pairwise import cosine_similarity
            query_vector = self.vectorizer.transform([query])
            scores = cosine_similarity(query_vector, self.doc_vectors)[0]
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            doc_id = self.doc_ids[idx]
            results.append(RetrievalResult(
                doc_id=doc_id,
                score=float(scores[idx]),
                text=self.documents[doc_id],
                rank=rank
            ))
        
        return results


class DenseRetriever(BaseRetriever):
    """Dense retrieval using pre-computed embeddings"""
    
    def __init__(self, model_dir: str = "retrieval_model", data_dir: str = "data"):
        """
        Initialize dense retriever
        
        Args:
            model_dir: Directory containing doc_embs.npy and doc_ids.json
            data_dir: Directory containing collection.jsonl
        """
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        
        # Load embeddings and metadata
        self._load_embeddings()
        self._load_collection()
        self._init_model()
    
    def _load_embeddings(self):
        """Load pre-computed document embeddings"""
        emb_path = self.model_dir / "doc_embs.npy"
        ids_path = self.model_dir / "doc_ids.json"
        meta_path = self.model_dir / "meta.json"
        
        self.doc_embeddings = np.load(emb_path)
        
        with open(ids_path, 'r') as f:
            self.doc_ids = json.load(f)
        
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)
        
        print(f"Loaded {len(self.doc_ids)} document embeddings")
        print(f"Model: {self.meta['model_name']}, Dimension: {self.meta['dim']}")
    
    def _load_collection(self):
        """Load document texts"""
        collection_path = self.data_dir / "collection.jsonl"
        self.documents = {}
        
        with open(collection_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                self.documents[doc['id']] = doc['text']
    
    def _init_model(self):
        """Initialize sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install: pip install sentence-transformers")
        
        self.model = SentenceTransformer(self.meta['model_name'])
        print(f"Loaded model: {self.meta['model_name']}")
    
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve top-k documents using dense embeddings"""
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        
        # Compute similarities
        scores = np.dot(self.doc_embeddings, query_embedding)
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            doc_id = self.doc_ids[idx]
            results.append(RetrievalResult(
                doc_id=doc_id,
                score=float(scores[idx]),
                text=self.documents[doc_id],
                rank=rank
            ))
        
        return results


class ColBERTRetriever(BaseRetriever):
    """Multi-vector retrieval using ColBERT"""
    
    def __init__(self, index_path: str, data_dir: str = "data"):
        """
        Initialize ColBERT retriever
        
        Args:
            index_path: Path to ColBERT index
            data_dir: Directory containing collection.jsonl
        """
        self.index_path = index_path
        self.data_dir = Path(data_dir)
        
        # Load collection
        self._load_collection()
        
        # Initialize ColBERT
        self._init_colbert()
    
    def _load_collection(self):
        """Load document collection"""
        collection_path = self.data_dir / "collection.jsonl"
        self.documents = {}
        
        with open(collection_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                self.documents[doc['id']] = doc['text']
        
        print(f"Loaded {len(self.documents)} documents")
    
    def _init_colbert(self):
        """Initialize ColBERT model"""
        try:
            from ragatouille import RAGPretrainedModel
        except ImportError:
            raise ImportError("Please install: pip install ragatouille")
        
        try:
            self.rag = RAGPretrainedModel.from_index(self.index_path)
            print(f"Loaded ColBERT index from {self.index_path}")
        except Exception as e:
            print(f"Warning: Could not load ColBERT index: {e}")
            print("ColBERT retriever will not be available")
            self.rag = None
    
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve using ColBERT"""
        if self.rag is None:
            print("Warning: ColBERT not initialized, returning empty results")
            return []
        
        # Search with ColBERT
        results = self.rag.search(query, k=top_k)
        
        retrieved = []
        for rank, res in enumerate(results, 1):
            doc_id = res['document_id']
            retrieved.append(RetrievalResult(
                doc_id=doc_id,
                score=float(res['score']),
                text=self.documents.get(doc_id, ""),
                rank=rank
            ))
        
        return retrieved


class HybridRetriever(BaseRetriever):
    """Hybrid retrieval combining multiple methods"""
    
    def __init__(self, retrievers: List[Tuple[BaseRetriever, float]]):
        """
        Initialize hybrid retriever
        
        Args:
            retrievers: List of (retriever, weight) tuples
        """
        self.retrievers = retrievers
        self.total_weight = sum(weight for _, weight in retrievers)
    
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Combine results from multiple retrievers"""
        all_scores = {}  # doc_id -> weighted score
        all_texts = {}   # doc_id -> text
        
        # Retrieve from each method
        for retriever, weight in self.retrievers:
            results = retriever.retrieve(query, top_k=top_k * 2)  # Get more candidates
            
            # Normalize and weight scores
            max_score = max([r.score for r in results]) if results else 1.0
            
            for result in results:
                norm_score = result.score / max_score if max_score > 0 else 0
                weighted_score = norm_score * weight
                
                if result.doc_id in all_scores:
                    all_scores[result.doc_id] += weighted_score
                else:
                    all_scores[result.doc_id] = weighted_score
                    all_texts[result.doc_id] = result.text
        
        # Sort by combined score
        sorted_docs = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top-k
        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs[:top_k], 1):
            results.append(RetrievalResult(
                doc_id=doc_id,
                score=score / self.total_weight,  # Normalize by total weight
                text=all_texts[doc_id],
                rank=rank
            ))
        
        return results


class RetrievalManager:
    """Manages multiple retrieval methods"""
    
    def __init__(self, data_dir: str = "data", retrieval_model_dir: str = "retrieval_model"):
        self.data_dir = data_dir
        self.retrieval_model_dir = retrieval_model_dir
        self.retrievers = {}
    
    def initialize_sparse(self, methods: List[str] = ["bm25", "tfidf"]):
        """Initialize sparse retrievers"""
        for method in methods:
            print(f"\nInitializing {method.upper()} retriever...")
            self.retrievers[method] = SparseRetriever(method=method, data_dir=self.data_dir)
    
    def initialize_dense(self):
        """Initialize dense retriever"""
        print("\nInitializing dense retriever...")
        self.retrievers["dense"] = DenseRetriever(
            model_dir=self.retrieval_model_dir,
            data_dir=self.data_dir
        )
    
    def initialize_colbert(self, index_path: str):
        """Initialize ColBERT retriever"""
        print("\nInitializing ColBERT retriever...")
        self.retrievers["colbert"] = ColBERTRetriever(
            index_path=index_path,
            data_dir=self.data_dir
        )
    
    def initialize_hybrid(self, method_weights: Dict[str, float]):
        """
        Initialize hybrid retriever
        
        Args:
            method_weights: Dict mapping method name to weight
                           e.g., {"bm25": 0.5, "dense": 0.5}
        """
        print("\nInitializing hybrid retriever...")
        retriever_pairs = []
        
        for method, weight in method_weights.items():
            if method in self.retrievers:
                retriever_pairs.append((self.retrievers[method], weight))
            else:
                print(f"Warning: {method} not initialized, skipping")
        
        if retriever_pairs:
            self.retrievers["hybrid"] = HybridRetriever(retriever_pairs)
            print(f"Hybrid retriever created with {len(retriever_pairs)} methods")
    
    def get_retriever(self, method: str) -> BaseRetriever:
        """Get retriever by name"""
        if method not in self.retrievers:
            raise ValueError(f"Retriever '{method}' not initialized. Available: {list(self.retrievers.keys())}")
        return self.retrievers[method]
    
    def list_available_methods(self) -> List[str]:
        """List all initialized retrieval methods"""
        return list(self.retrievers.keys())


# Convenience function for quick retrieval
def build_search_engine_for_webui(artifacts: str = None, method: str = "bm25", data_dir: str = "data"):
    """
    Build a search engine for web UI (backward compatibility)
    
    Args:
        artifacts: Not used, kept for compatibility
        method: Retrieval method
        data_dir: Data directory
    
    Returns:
        Object with search(query, topk) method
    """
    class SearchEngineWrapper:
        def __init__(self, retriever):
            self.retriever = retriever
        
        def search(self, query: str, topk: int = 10):
            """Returns List[(doc_id, score, text)]"""
            results = self.retriever.retrieve(query, top_k=topk)
            return [(r.doc_id, r.score, r.text) for r in results]
    
    # Initialize retriever based on method
    if method in ["bm25", "tfidf"]:
        retriever = SparseRetriever(method=method, data_dir=data_dir)
    elif method == "dense":
        retriever = DenseRetriever(data_dir=data_dir)
    elif method == "hybrid":
        # Default hybrid: BM25 + Dense
        bm25 = SparseRetriever(method="bm25", data_dir=data_dir)
        try:
            dense = DenseRetriever(data_dir=data_dir)
            retriever = HybridRetriever([(bm25, 0.5), (dense, 0.5)])
        except:
            print("Warning: Dense retrieval failed, using BM25 only")
            retriever = bm25
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return SearchEngineWrapper(retriever)


if __name__ == "__main__":
    # Example usage
    print("=== Testing Retrieval Module ===\n")
    
    # Test sparse retrieval
    print("Testing BM25...")
    bm25 = SparseRetriever(method="bm25")
    results = bm25.retrieve("Where was Barack Obama born?", top_k=3)
    for r in results:
        print(f"Rank {r.rank}: {r.doc_id} (score: {r.score:.4f})")
        print(f"Text: {r.text[:100]}...\n")
