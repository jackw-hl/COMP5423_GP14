"""
Simple Web UI for RAG System - Simplified for compatibility
"""

import gradio as gr
from rag_system import IntegratedRAGSystem

# Global RAG system
rag_system = None


def initialize_rag(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    """Initialize RAG system"""
    global rag_system
    print("Initializing RAG system...")
    rag_system = IntegratedRAGSystem(
        model_name=model_name,
        data_dir="data",
        retrieval_model_dir="retrieval_model"
    )
    print("✅ RAG system ready!")


def query_rag(question, method, top_k, multiturn, agentic):
    """Process a query"""
    if not question or not question.strip():
        return "Please enter a question.", ""
    
    try:
        response = rag_system.query(
            user_query=question.strip(),
            retrieval_method=method,
            top_k=int(top_k),
            use_multiturn=multiturn,
            use_agentic=agentic
        )
        
        # Format answer
        answer = response.answer
        if response.refined_query and response.refined_query != response.original_query:
            answer += f"\n\n💡 Refined query: {response.refined_query}"
        
        # Format docs
        docs_text = f"**Retrieved {len(response.retrieved_docs)} documents:**\n\n"
        for doc in response.retrieved_docs[:5]:
            docs_text += f"- **[{doc.rank}] {doc.doc_id}** (score: {doc.score:.4f})\n"
            docs_text += f"  {doc.text[:200]}...\n\n"
        
        return answer, docs_text
        
    except Exception as e:
        return f"❌ Error: {str(e)}", ""


def clear_chat():
    """Clear conversation"""
    rag_system.clear_conversation()
    return "", ""


def create_simple_ui():
    """Create simplified Gradio UI"""
    
    methods = rag_system.retrieval_manager.list_available_methods()
    
    with gr.Blocks(title="RAG System") as demo:
        gr.Markdown("# 🤖 RAG System - COMP5423 Group 14")
        
        with gr.Row():
            with gr.Column():
                question = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask something...",
                    lines=3
                )
                
                with gr.Row():
                    method = gr.Dropdown(
                        choices=methods,
                        value=methods[0],
                        label="Retrieval Method"
                    )
                    top_k = gr.Slider(1, 10, 5, step=1, label="Top-K")
                
                with gr.Row():
                    multiturn = gr.Checkbox(label="Multi-turn (Feature A)")
                    agentic = gr.Checkbox(label="Agentic (Feature B)")
                
                with gr.Row():
                    submit = gr.Button("🚀 Ask", variant="primary")
                    clear = gr.Button("Clear")
                
                answer = gr.Textbox(label="Answer", lines=10)
                
            with gr.Column():
                docs = gr.Markdown("Retrieved documents will appear here")
        
        submit.click(
            query_rag,
            inputs=[question, method, top_k, multiturn, agentic],
            outputs=[answer, docs]
        )
        
        clear.click(clear_chat, outputs=[answer, docs])
        
        gr.Examples(
            examples=[
                ["Which airport is located in Maine?"],
                ["Where is Barack Obama from?"],
            ],
            inputs=question
        )
    
    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    
    initialize_rag(args.model)
    demo = create_simple_ui()
    demo.launch(server_name="127.0.0.1", server_port=args.port)
