"""
Integrated RAG System
Combines retrieval, generation, multi-turn dialogue, and agentic workflow
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

from retrieval_module import (
    BaseRetriever,
    RetrievalManager,
    RetrievalResult
)
from generation_module import (
    BaseGenerator,
    GenerationInput,
    MultiTurnDialogManager,
    QueryRefinementEngine,
    ReActEngine
)


@dataclass
class RAGResponse:
    """Response from RAG system"""
    answer: str
    retrieved_docs: List[RetrievalResult]
    original_query: str
    refined_query: Optional[str] = None
    thought_chain: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class IntegratedRAGSystem:
    """
    Complete RAG system with:
    - Multiple retrieval methods (sparse, dense, hybrid, ColBERT)
    - Basic single-turn RAG
    - Multi-turn conversation (Feature A)
    - Agentic workflow with ReAct (Feature B)
    """
    
    def __init__(self,
                 model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 data_dir: str = "data",
                 retrieval_model_dir: str = "retrieval_model",
                 colbert_index_path: str = "retrieval_model/hq_small_collection"):
        """
        Initialize RAG system
        
        Args:
            model_name: Qwen model to use for generation
            data_dir: Directory containing data files
            retrieval_model_dir: Directory containing dense retrieval model
            colbert_index_path: Path to ColBERT index (optional)
        """
        print("=" * 60)
        print("Initializing Integrated RAG System")
        print("=" * 60)
        
        # Initialize retrieval
        print("\n[1/3] Initializing Retrieval Module...")
        self.retrieval_manager = RetrievalManager(
            data_dir=data_dir,
            retrieval_model_dir=retrieval_model_dir
        )
        
        # Initialize sparse methods (BM25 only - TF-IDF is too slow for 144K docs)
        try:
            self.retrieval_manager.initialize_sparse(methods=["bm25"])
            print("Note: TF-IDF skipped for faster startup (use BM25, Dense, or Hybrid instead)")
        except Exception as e:
            print(f"Warning: Failed to initialize sparse retrieval: {e}")
        
        # Initialize dense method
        try:
            self.retrieval_manager.initialize_dense()
        except Exception as e:
            print(f"Warning: Failed to initialize dense retrieval: {e}")

         # Initialize Qwen3 instruction-aware dense retriever (FAISS)
        try:
            self.retrieval_manager.initialize_qwen3_dense(
                model_dir="retrieval_model/dense_instruction_qwen3-0.6b",
                batch_size=16,  # matches DEFAULT_BATCH_SIZE in dense_qwen3.ipynb for qwen3-0.6b
            )
        except Exception as e:
            print(f"Warning: Failed to initialize Qwen3 dense retriever: {e}")
        
        # Initialize ColBERT (if available)
        if colbert_index_path:
            try:
                self.retrieval_manager.initialize_colbert(colbert_index_path)
            except Exception as e:
                print(f"Warning: Failed to initialize ColBERT: {e}")
        
        # Initialize hybrid retrieval
        try:
            available_methods = self.retrieval_manager.list_available_methods()
            
            # Create hybrid based on available methods
            hybrid_weights = {}
            if "bm25" in available_methods:
                hybrid_weights["bm25"] = 0.4
            if "dense" in available_methods:
                hybrid_weights["dense"] = 0.6
            
            if len(hybrid_weights) > 1:
                self.retrieval_manager.initialize_hybrid(hybrid_weights)
                print(f"Hybrid retrieval initialized with: {list(hybrid_weights.keys())}")
        except Exception as e:
            print(f"Warning: Failed to initialize hybrid retrieval: {e}")
        
        # Initialize generation
        print("\n[2/3] Initializing Generation Module...")
        self.generator = BaseGenerator(model_name=model_name)
        
        # Initialize multi-turn components
        print("\n[3/3] Initializing Multi-Turn and Agentic Components...")
        self.dialog_manager = MultiTurnDialogManager()
        # QueryRefinementEngine needs model and tokenizer from notebook implementation
        self.query_refiner = QueryRefinementEngine(self.generator.model, self.generator.tokenizer)
        self.react_engine = ReActEngine(self.generator)
        
        print("\n" + "=" * 60)
        print("RAG System Initialization Complete!")
        print(f"Available retrieval methods: {self.retrieval_manager.list_available_methods()}")
        print(f"Generator model: {model_name}")
        print("=" * 60 + "\n")
    
    def query(self,
              user_query: str,
              retrieval_method: str = "hybrid",
              top_k: int = 5,
              use_multiturn: bool = False,
              use_agentic: bool = False) -> RAGResponse:
        """
        Query the RAG system
        
        Args:
            user_query: User's question
            retrieval_method: Retrieval method to use
            top_k: Number of documents to retrieve
            use_multiturn: Whether to use multi-turn conversation context
            use_agentic: Whether to use agentic workflow (ReAct)
        
        Returns:
            RAGResponse object
        """
        # Step 1: Query refinement (if multi-turn)
        refined_query = user_query
        if use_multiturn:
            dialog_context = self.dialog_manager.get_context_summary()
            if dialog_context:
                refined_query = self.query_refiner.refine_query(user_query, dialog_context)
                print(f"🔄 Refined query: {refined_query}")
        
        # Step 2: Choose workflow
        if use_agentic:
            # Use ReAct agentic workflow
            answer, thought_chain = self._agentic_workflow(
                refined_query,
                retrieval_method,
                top_k
            )
            # Retrieve final docs for display
            retriever = self.retrieval_manager.get_retriever(retrieval_method)
            retrieved_docs = retriever.retrieve(refined_query, top_k=top_k)
            
        else:
            # Basic RAG workflow
            answer, retrieved_docs, thought_chain = self._basic_workflow(
                refined_query,
                retrieval_method,
                top_k,
                use_multiturn
            )
        
        # Step 3: Update conversation history (if multi-turn)
        if use_multiturn:
            doc_texts = [d.text for d in retrieved_docs]
            self.dialog_manager.add_turn(user_query, answer, doc_texts)
        
        # Step 4: Return response
        return RAGResponse(
            answer=answer,
            retrieved_docs=retrieved_docs,
            original_query=user_query,
            refined_query=refined_query if refined_query != user_query else None,
            thought_chain=thought_chain,
            metadata={
                "retrieval_method": retrieval_method,
                "top_k": top_k,
                "use_multiturn": use_multiturn,
                "use_agentic": use_agentic
            }
        )
    
    def _basic_workflow(self,
                       query: str,
                       retrieval_method: str,
                       top_k: int,
                       use_multiturn: bool) -> Tuple[str, List[RetrievalResult], None]:
        """Basic RAG workflow"""
        # Retrieve
        retriever = self.retrieval_manager.get_retriever(retrieval_method)
        retrieved_docs = retriever.retrieve(query, top_k=top_k)
        
        # Generate
        doc_texts = [d.text for d in retrieved_docs]
        
        context = ""
        if use_multiturn:
            context = self.dialog_manager.get_context_summary()
        
        gen_input = GenerationInput(
            query=query,
            retrieved_docs=doc_texts,
            context=context
        )
        
        template = "multiturn_rag" if use_multiturn else "basic_rag"
        answer = self.generator.generate(gen_input, template_type=template)
        
        return answer, retrieved_docs, None
    
    def _agentic_workflow(self,
                         query: str,
                         retrieval_method: str,
                         top_k: int) -> Tuple[str, List[str]]:
        """Agentic workflow using ReAct"""
        # Get retriever
        retriever = self.retrieval_manager.get_retriever(retrieval_method)
        
        # Execute ReAct cycle
        answer, thought_chain = self.react_engine.execute_react_cycle(
            query=query,
            retriever=retriever,
            max_iterations=3
        )
        
        return answer, thought_chain
    
    def multiturn_query(self,
                       user_query: str,
                       retrieval_method: str = "hybrid",
                       top_k: int = 5,
                       use_agentic: bool = False) -> RAGResponse:
        """
        Convenience method for multi-turn query
        
        Args:
            user_query: User's question
            retrieval_method: Retrieval method
            top_k: Number of documents
            use_agentic: Whether to use agentic workflow
        
        Returns:
            RAGResponse object
        """
        return self.query(
            user_query=user_query,
            retrieval_method=retrieval_method,
            top_k=top_k,
            use_multiturn=True,
            use_agentic=use_agentic
        )
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.dialog_manager.clear_history()
        print("Conversation history cleared")
    
    def batch_query(self,
                   queries: List[str],
                   retrieval_method: str = "hybrid",
                   top_k: int = 5) -> List[RAGResponse]:
        """
        Process multiple queries in batch
        
        Args:
            queries: List of queries
            retrieval_method: Retrieval method
            top_k: Number of documents per query
        
        Returns:
            List of RAGResponse objects
        """
        responses = []
        for query in queries:
            response = self.query(
                user_query=query,
                retrieval_method=retrieval_method,
                top_k=top_k,
                use_multiturn=False,
                use_agentic=False
            )
            responses.append(response)
        
        return responses
    
    def evaluate_on_dataset(self,
                           dataset_path: str,
                           retrieval_method: str = "hybrid",
                           top_k: int = 10,
                           max_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate on a dataset (train/validation/test)
        
        Args:
            dataset_path: Path to JSONL file
            retrieval_method: Retrieval method to use
            top_k: Number of documents to retrieve
            max_samples: Maximum number of samples to process
        
        Returns:
            Dict with predictions and statistics
        """
        print(f"\nEvaluating on: {dataset_path}")
        print(f"Retrieval method: {retrieval_method}, top-k: {top_k}")
        
        # Load dataset
        queries = []
        ids = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                item = json.loads(line)
                queries.append(item['text'])
                ids.append(item['id'])
        
        print(f"Processing {len(queries)} queries...")
        
        # Process queries
        predictions = []
        for i, (qid, query) in enumerate(zip(ids, queries), 1):
            if i % 50 == 0:
                print(f"  Processed {i}/{len(queries)}...")
            
            response = self.query(
                user_query=query,
                retrieval_method=retrieval_method,
                top_k=top_k,
                use_multiturn=False,
                use_agentic=False
            )
            
            predictions.append({
                "id": qid,
                "answer": response.answer,
                "supporting_ids": [doc.doc_id for doc in response.retrieved_docs]
            })
        
        print(f"✅ Completed {len(predictions)} predictions")
        
        return {
            "predictions": predictions,
            "num_samples": len(predictions),
            "retrieval_method": retrieval_method,
            "top_k": top_k
        }
    
    def save_predictions(self,
                        predictions: List[Dict[str, Any]],
                        output_path: str):
        """
        Save predictions to JSONL file
        
        Args:
            predictions: List of prediction dicts
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for pred in predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + '\n')
        
        print(f"💾 Saved {len(predictions)} predictions to: {output_path}")


def demo_basic_rag():
    """Demo: Basic single-turn RAG"""
    print("\n" + "="*60)
    print("DEMO: Basic Single-Turn RAG")
    print("="*60)
    
    # Initialize system
    rag = IntegratedRAGSystem(model_name="Qwen/Qwen2.5-0.5B-Instruct")
    
    # Query
    query = "Which airport is located in Maine?"
    print(f"\nQuery: {query}")
    
    response = rag.query(query, retrieval_method="bm25", top_k=3)
    
    print(f"\nAnswer: {response.answer}")
    print(f"\nRetrieved Documents:")
    for doc in response.retrieved_docs:
        print(f"  - [{doc.rank}] {doc.doc_id} (score: {doc.score:.4f})")
        print(f"    {doc.text[:100]}...")


def demo_multiturn():
    """Demo: Multi-turn conversation (Feature A)"""
    print("\n" + "="*60)
    print("DEMO: Multi-Turn Conversation (Feature A)")
    print("="*60)
    
    rag = IntegratedRAGSystem(model_name="Qwen/Qwen2.5-0.5B-Instruct")
    
    # First turn
    print("\n--- Turn 1 ---")
    query1 = "Who is Barack Obama?"
    print(f"User: {query1}")
    response1 = rag.multiturn_query(query1, retrieval_method="bm25")
    print(f"Assistant: {response1.answer}")
    
    # Second turn (with reference)
    print("\n--- Turn 2 ---")
    query2 = "Where was he born?"
    print(f"User: {query2}")
    response2 = rag.multiturn_query(query2, retrieval_method="bm25")
    print(f"Refined query: {response2.refined_query}")
    print(f"Assistant: {response2.answer}")


def demo_agentic():
    """Demo: Agentic workflow with ReAct (Feature B)"""
    print("\n" + "="*60)
    print("DEMO: Agentic Workflow with ReAct (Feature B)")
    print("="*60)
    
    rag = IntegratedRAGSystem(model_name="Qwen/Qwen2.5-0.5B-Instruct")
    
    query = "What is the relationship between machine learning and artificial intelligence?"
    print(f"\nQuery: {query}")
    
    response = rag.query(
        query,
        retrieval_method="bm25",
        use_agentic=True
    )
    
    print(f"\nFinal Answer: {response.answer}")
    
    if response.thought_chain:
        print(f"\nThought Chain:")
        for i, thought in enumerate(response.thought_chain, 1):
            print(f"  {i}. {thought}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        demo_type = sys.argv[1]
        
        if demo_type == "basic":
            demo_basic_rag()
        elif demo_type == "multiturn":
            demo_multiturn()
        elif demo_type == "agentic":
            demo_agentic()
        else:
            print(f"Unknown demo: {demo_type}")
            print("Available demos: basic, multiturn, agentic")
    else:
        # Run basic demo by default
        demo_basic_rag()
