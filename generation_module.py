"""
Generation Module - Extracted from generate module.ipynb
Implements Feature A (Multi-turn) and Feature B (ReAct) for COMP5423 RAG System
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
import re


# ============================================================================
# Core Data Structures
# ============================================================================

@dataclass
class GenerationInput:
    """Input structure for generation"""
    query: str
    retrieved_docs: List[str]
    context: str = ""  # Multi-turn dialogue history


# ============================================================================
# Simple Retriever (from notebook)
# ============================================================================

class SimpleRetriever:
    """Simple retriever wrapper for RAG system"""
    def __init__(self, rag_system, dataset):
        self.rag = rag_system
        self.dataset = dataset

    def search(self, query: str, k: int = 10):
        """Search and return passages with metadata"""
        results = self.rag.search(query, k=k)
        passages = []
        docs_metadata = []

        for res in results:
            doc_id = res['document_id']
            doc_text = next(
                item['text'] for item in self.dataset['collection']
                if item['id'] == doc_id
            )
            passages.append(doc_text)
            docs_metadata.append({
                'id': doc_id,
                'score': res['score'],
                'text': doc_text[:100] + "..."
            })

        return passages, docs_metadata


# ============================================================================
# Qwen Generator (from notebook)
# ============================================================================

class QwenGenerator:
    """Qwen-based answer generator"""
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, input_data: GenerationInput) -> str:
        """Generate answer based on query, docs, and context"""
        passages = "\n".join([f"[{i+1}] {p}" for i, p in enumerate(input_data.retrieved_docs)])

        # Use English version of the prompt from notebook
        prompt = f"""You are a professional AI assistant. Answer based on the evidence and conversation history.

Current question: {input_data.query}

Conversation history:
{input_data.context if input_data.context else 'None'}

Evidence passages:
{passages}

Requirements:
- If passages are not relevant, say "Cannot determine based on available evidence"
- Answer concisely in English
Answer:"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=4096,
            truncation=True,
            padding=True
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.3,
            top_p=0.85,
            pad_token_id=self.tokenizer.eos_token_id
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Answer:" in response:
            response = response.split("Answer:", 1)[-1].strip()
        return response


# ============================================================================
# Feature A: Multi-Turn Components (from notebook)
# ============================================================================

class MultiTurnDialogManager:
    """Manages multi-turn conversation history"""
    def __init__(self, max_turns=10, max_context_tokens=4000):
        self.dialog_history = []
        self.max_turns = max_turns
        self.max_context_tokens = max_context_tokens
        self.retrieved_docs_cache = {}

    def add_turn(self, query, response, retrieved_docs):
        """Add a conversation turn"""
        turn_record = {
            'query': query,
            'response': response,
            'retrieved_docs': retrieved_docs,
            'timestamp': time.time()
        }
        self.dialog_history.append(turn_record)

        if len(self.dialog_history) > self.max_turns:
            self.dialog_history.pop(0)

    def get_context_summary(self):
        """Generate conversation context summary"""
        if not self.dialog_history:
            return ""

        recent_turns = self.dialog_history[-3:]
        context_parts = []

        for turn in recent_turns:
            context_parts.append(f"User: {turn['query']}")
            context_parts.append(f"System: {turn['response'][:100]}...")

        return "\n".join(context_parts)


class QueryRefinementEngine:
    """Refines queries based on conversation context"""
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def refine_query(self, current_query, dialog_context):
        """Refine query based on dialogue context"""
        if not dialog_context:
            return current_query

        prompt_template = """Based on the current question and conversation history, rewrite the question as a complete, independent query.
Resolve all references (e.g., "it", "this") so the rewritten query can be understood without the conversation history.

Current question: {current_query}

Conversation history: {dialog_context}

Rewritten complete query:
"""
        prompt = prompt_template.format(
            dialog_context=dialog_context,
            current_query=current_query
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            temperature=0.1,
            pad_token_id=self.tokenizer.eos_token_id
        )

        refined_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if prompt in refined_query:
            refined_query = refined_query[len(prompt):].strip()
        else:
            refined_query = refined_query.strip().split('\n')[-1].strip()

        if refined_query.startswith("Rewritten complete query:"):
            refined_query = refined_query[len("Rewritten complete query:"):].strip()

        return refined_query.strip()


class MultiTurnRetrievalOptimizer:
    """Optimizes retrieval with conversation context"""
    def __init__(self, base_retriever):
        self.retriever = base_retriever
        self.doc_seen_tracker = {}

    def retrieve_with_context(self, refined_query, dialog_manager, k=10):
        """Context-aware retrieval"""
        all_docs, metadata = self.retriever.search(refined_query, k=int(k*1.5))

        filtered_results = []
        for doc in metadata:
            filtered_results.append(doc)

        filtered_results.sort(key=lambda x: x['score'], reverse=True)
        return [i['text'] for i in filtered_results[:k]]


class MultiTurnRAGSystem:
    """Multi-turn RAG system (Feature A)"""
    def __init__(self, retriever, generator, qwen_model, tokenizer):
        self.retriever = retriever
        self.generator = generator
        self.dialog_manager = MultiTurnDialogManager()
        self.query_refiner = QueryRefinementEngine(qwen_model, tokenizer)
        self.retrieval_optimizer = MultiTurnRetrievalOptimizer(retriever)

    def chat(self, user_query):
        """Process multi-turn conversation"""
        dialog_context = self.dialog_manager.get_context_summary()

        if dialog_context:
            refined_query = self.query_refiner.refine_query(user_query, dialog_context)
        else:
            refined_query = user_query

        retrieved_docs = self.retrieval_optimizer.retrieve_with_context(
            refined_query, self.dialog_manager
        )

        generation_input = GenerationInput(
            query=user_query,
            context=dialog_context,
            retrieved_docs=retrieved_docs,
        )

        response = self.generator.generate(generation_input)
        self.dialog_manager.add_turn(user_query, response, retrieved_docs)

        return {
            'response': response,
            'refined_query': refined_query,
            'retrieved_docs': retrieved_docs[:3]
        }


# ============================================================================
# Feature B: ReAct Components (from notebook)
# ============================================================================

class PromptTemplates:
    """Prompt templates for ReAct"""
    
    def react_plan_template(self, query: str, context: List[str]) -> str:
        ctx = "\n".join(context) if context else "None"
        return (
            "You are a ReAct (Reasoning-Acting-Reflecting) agent.\n"
            "Given the question and current context, output ONLY the next step plan.\n\n"
            f"Question: {query}\n"
            f"Context:\n{ctx}\n\n"
            "Plan:"
        )

    def react_reflection_template(self, query: str, retrieved_docs: List[str], plan: str) -> str:
        passages = "\n".join([f"[{i+1}] {p}" for i, p in enumerate(retrieved_docs)]) if retrieved_docs else "None"
        return (
            "You are executing a ReAct (Reasoning-Acting) cycle.\n"
            "Based on the following information, decide if you have enough evidence to answer the original question.\n\n"
            f"Original question: {query}\n\n"
            f"Retrieval plan: {plan}\n\n"
            f"Retrieved results:\n{passages}\n\n"
            "Output strictly in this format:\n"
            "- Decision: Continue retrieval / Generate final answer\n"
            "- Reason: Brief explanation\n"
            "- (If continue) New query: ...\n"
            "- (If can answer) Final answer: ...\n\n"
            "Your output:"
        )


class BaseGenerator:
    """
    Base generator supporting multiple template types
    Compatible with rag_system.py while using notebook's implementation
    """
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_name = model_name
        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        print(f"Model loaded on {self.model.device}\n")
        
        # Create QwenGenerator for standard generation
        self.qwen_generator = QwenGenerator(self.model, self.tokenizer)
        self.prompt_templates = PromptTemplates()

        self.generation_config = {
            "max_new_tokens": 512,
            "temperature": 0.3,
            "top_p": 0.85,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id
        }

    def generate(self, generation_input: GenerationInput, template_type: str = "single_round") -> str:
        """Generate with different template types"""
        if template_type == "react_plan":
            prompt = self.prompt_templates.react_plan_template(
                generation_input.query,
                generation_input.context.split('\n') if generation_input.context else []
            )
        elif template_type == "react_reflection":
            prompt = generation_input.query
        elif template_type == "single_round":
            # Use QwenGenerator from notebook for standard generation
            return self.qwen_generator.generate(generation_input)
        else:
            # Default single round
            passages = "\n".join([f"[{i+1}] {p}" for i, p in enumerate(generation_input.retrieved_docs)])
            prompt = f"""You are a helpful QA assistant. Answer based on the passages.

Question: {generation_input.query}

Passages:
{passages}

Answer:"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=4096,
            truncation=True,
            padding=True
        ).to(self.model.device)

        outputs = self.model.generate(**inputs, **self.generation_config)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Answer:" in text:
            text = text.split("Answer:", 1)[-1].strip()
        elif "Plan:" in text:
            text = text.split("Plan:", 1)[-1].strip()

        return text.strip()


class ReActEngine:
    """ReAct agent engine (Feature B) - from notebook"""
    def __init__(self, generator: BaseGenerator):
        self.generator = generator
        self.prompt_templates = PromptTemplates()
        self.thought_chain: List[str] = []

    def _generate_plan(self, state: Dict[str, Any]) -> str:
        """Generate retrieval plan"""
        gi = GenerationInput(
            query=state["query"],
            retrieved_docs=[],
            context="\n".join(state.get("context", []))
        )
        raw_output = self.generator.generate(gi, template_type="react_plan")

        if "Plan:" in raw_output:
            plan = raw_output.split("Plan:", 1)[-1].strip()
        else:
            plan = raw_output
            instruction_prefix = "You are a ReAct"
            if plan.startswith(instruction_prefix):
                plan = plan[len(instruction_prefix):].strip()

        return plan

    def _execute_action(self, plan: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute retrieval action based on plan"""
        plan_lower = plan.lower()
        has_retrieval_intent = (
            any(kw in plan_lower for kw in ["search", "retrieval", "find", "look", "query"]) 
        )

        queries = []
        if has_retrieval_intent:
            lines = re.split(r'[\n；;•·\-–—]+\s*', plan.strip())
            candidate_queries = []

            for line in lines:
                line = line.strip()
                if not line or len(line) < 3:
                    continue
                cleaned = re.sub(r'^[\s\-•\d\.\)\(（）【】\[\]""''""\'\*\u2022]+\s*', '', line)
                if cleaned.endswith((".", "。", "?", "？", ":", "：")):
                    continue
                candidate_queries.append(cleaned)

            if candidate_queries:
                queries = candidate_queries[:3]
            else:
                words = re.findall(r'[a-zA-Z0-9\u4e00-\u9fa5]{2,}', plan)
                queries = [" ".join(words[-6:])] if words else [state["query"]]
        else:
            queries = [state["query"]]

        all_docs = []
        used_queries = []
        for q in queries:
            docs, _ = state["retriever"].search(q, k=3)
            all_docs.extend(docs)
            used_queries.append(q)

        unique_docs = list(dict.fromkeys(all_docs))

        print(f"\n🔍 Retrieved {len(unique_docs)} documents")

        return {
            "retrieved_docs": unique_docs[:10],
            "plan_query": " | ".join(used_queries)
        }

    def _reflect_on_result(self, action_result: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Reflect on retrieval results"""
        docs = action_result.get("retrieved_docs", [])
        plan = action_result.get("plan_query", "")

        reflection_prompt = self.prompt_templates.react_reflection_template(
            query=state["query"],
            retrieved_docs=docs,
            plan=plan
        )

        gi = GenerationInput(query=reflection_prompt, retrieved_docs=[], context="")
        reflection_output = self.generator.generate(gi, template_type="react_reflection")

        reflection_lower = reflection_output.lower()

        if any(kw in reflection_lower for kw in ["final answer", "sufficient", "complete"]):
            answer = reflection_output
            for marker in ["Final answer:", "Answer:"]:
                if marker in reflection_output:
                    answer = reflection_output.split(marker, 1)[-1].strip()
                    break
            return {"is_final": True, "answer": answer, "reflection": reflection_output}

        elif any(kw in reflection_lower for kw in ["continue retrieval", "need more", "insufficient"]):
            new_query = state["query"]
            for marker in ["New query:", "new query:"]:
                if marker in reflection_output:
                    parts = reflection_output.split(marker, 1)
                    if len(parts) > 1:
                        new_query = parts[1].split("\n")[0].strip()
                        break
            return {"is_final": False, "next_query": new_query, "reflection": reflection_output}

        else:
            if docs:
                gi_answer = GenerationInput(query=state["query"], retrieved_docs=docs, context="")
                answer = self.generator.generate(gi_answer)
                return {"is_final": True, "answer": answer, "reflection": reflection_output}
            else:
                return {"is_final": False, "reflection": reflection_output}

    def execute_react_cycle(self, query: str, retriever: Any, max_iterations: int = 3) -> str:
        """Execute full ReAct cycle"""
        current_query = query
        current_context = []

        print(f"\n🚀 Starting ReAct cycle for: '{query}'")

        for step in range(max_iterations):
            print(f"\n🔁 Iteration {step+1}/{max_iterations}")

            plan = self._generate_plan({"query": current_query, "context": current_context, "retriever": retriever})
            print(f"📝 Plan: {plan}")

            action_result = self._execute_action(plan, {"query": current_query, "retriever": retriever})

            reflection = self._reflect_on_result(action_result, {"query": current_query, "retriever": retriever})

            if reflection.get("is_final", False):
                print("✅ Final answer generated")
                return reflection["answer"]

            if "next_query" in reflection:
                current_query = reflection["next_query"]

        return "Maximum iterations reached without final answer."
