"""
Enhanced Simple Web UI for RAG System
- Keep the SAME generation call relationship as the simplified UI:
    rag_system.query(user_query, retrieval_method, top_k, use_multiturn, use_agentic)
- Bring back most UX from the original UI:
  * Chat history (multi-turn display)
  * Collapsible retrieved docs (each doc folded by default)
  * Collapsible workflow / agentic log panels
"""

import gradio as gr
from html import escape
from typing import List, Dict, Any, Tuple
from rag_system import IntegratedRAGSystem

# Global RAG system
rag_system = None


def initialize_rag(model_name="Qwen/Qwen2.5-0.5B-Instruct",gen_backend="local"):
    """Initialize RAG system (load once)"""
    global rag_system
    print("Initializing RAG system...")
    rag_system = IntegratedRAGSystem(
        model_name=model_name,
        data_dir="data",
        retrieval_model_dir="retrieval_model",
        gen_backend=gen_backend,
    )
    print("✅ RAG system ready!")


def init_history() -> List[Dict[str, Any]]:
    return []


def _safe_get(obj, name, default=None):
    return getattr(obj, name, default)


def _format_chat_message(answer: str, original_query: str, refined_query: str) -> str:
    msg = answer or ""
    if refined_query and original_query and refined_query.strip() != original_query.strip():
        msg += f"\n\n---\nRefined query: `{refined_query}`"
    return msg


def build_docs_html(retrieved_docs: List[Any], max_chars: int = 450) -> str:
    n = len(retrieved_docs or [])
    if n == 0:
        return "<p>No documents retrieved.</p>"

    blocks = []
    for i, d in enumerate(retrieved_docs, 1):
        doc_id = escape(str(_safe_get(d, "doc_id", f"doc_{i}")))
        score = _safe_get(d, "score", None)
        rank = _safe_get(d, "rank", None)

        text = str(_safe_get(d, "text", "") or "")
        snippet = escape(text[:max_chars])

        meta = []
        if rank is not None:
            meta.append(f"rank={rank}")
        if score is not None:
            try:
                meta.append(f"score={float(score):.4f}")
            except Exception:
                meta.append(f"score={escape(str(score))}")

        meta_txt = (", ".join(meta)) if meta else "no-score"

        blocks.append(
            f"""
            <details>
              <summary><b>[{i}] {doc_id}</b> <span style="opacity:.8">({meta_txt})</span></summary>
              <pre style="white-space: pre-wrap; font-size: 0.9em; margin-top: 6px;">{snippet}...</pre>
            </details>
            """
        )

    return f"""
    <details open>
      <summary><b>Retrieved {n} documents</b> (each folded by default)</summary>
      <div style="margin-top:8px;">{''.join(blocks)}</div>
    </details>
    """


def build_workflow_html(
    response: Any,
    method: str,
    top_k: int,
    multiturn: bool,
    agentic: bool,
) -> str:
    original_query = _safe_get(response, "original_query", "") or ""
    refined_query = _safe_get(response, "refined_query", "") or ""

    # Step 0: query & config
    step0 = f"""
    <details open>
      <summary><b>Step 0: Query & Config</b></summary>
      <div style="margin-top:6px; line-height:1.5;">
        <div><b>retrieval_method</b>: <code>{escape(str(method))}</code></div>
        <div><b>top_k</b>: <code>{int(top_k)}</code></div>
        <div><b>multi-turn</b>: <code>{str(bool(multiturn))}</code></div>
        <div><b>agentic</b>: <code>{str(bool(agentic))}</code></div>
        <hr style="opacity:.25"/>
        <div><b>Original query</b>:</div>
        <pre style="white-space: pre-wrap; font-size: 0.9em;">{escape(str(original_query))}</pre>
        <div><b>Refined query</b>:</div>
        <pre style="white-space: pre-wrap; font-size: 0.9em;">{escape(str(refined_query))}</pre>
      </div>
    </details>
    """

    # Step 1: retrieval (reuse docs panel content, but here we only put a short index list)
    docs = _safe_get(response, "retrieved_docs", []) or []
    idx_lines = []
    for i, d in enumerate(docs, 1):
        doc_id = escape(str(_safe_get(d, "doc_id", f"doc_{i}")))
        score = _safe_get(d, "score", None)
        if score is None:
            idx_lines.append(f"<li>[{i}] <code>{doc_id}</code></li>")
        else:
            try:
                idx_lines.append(f"<li>[{i}] <code>{doc_id}</code> (score={float(score):.4f})</li>")
            except Exception:
                idx_lines.append(f"<li>[{i}] <code>{doc_id}</code> (score={escape(str(score))})</li>")

    step1 = f"""
    <details>
      <summary><b>Step 1: Retrieval Summary</b></summary>
      <div style="margin-top:6px;">
        <div>Docs count: <b>{len(docs)}</b></div>
        <ul style="margin-top:6px;">{''.join(idx_lines) or "<li>No docs.</li>"}</ul>
      </div>
    </details>
    """

    # Step 2+: agentic trace if available, otherwise a safe placeholder
    # (We do NOT change the pipeline; just display what is available from response.)
    possible_trace_fields = [
        "workflow_log",
        "agentic_log",
        "trace",
        "steps",
        "debug_log",
        "reasoning_log",
    ]
    trace_val = None
    for f in possible_trace_fields:
        v = _safe_get(response, f, None)
        if v:
            trace_val = v
            break

    if trace_val is None:
        if agentic:
            trace_txt = (
                "Agentic mode is ON, but this response object does not expose a step-by-step trace field.\n"
                "You can optionally add a `workflow_log/trace/steps` field inside your RAG response to show here."
            )
        else:
            trace_txt = "Agentic mode is OFF."
    else:
        # Make it readable even if it's a list/dict
        if isinstance(trace_val, (list, dict)):
            trace_txt = str(trace_val)
        else:
            trace_txt = str(trace_val)

    step2 = f"""
    <details>
      <summary><b>Step 2: Agentic / Workflow Log</b></summary>
      <pre style="white-space: pre-wrap; font-size: 0.9em; margin-top: 6px;">{escape(trace_txt)}</pre>
    </details>
    """

    return step0 + "\n" + step1 + "\n" + step2


def query_rag(
    question: str,
    method: str,
    top_k: int,
    multiturn: bool,
    agentic: bool,
    history_state: List[Dict[str, Any]],
):
    """Process a query (KEEP the same rag_system.query call relationship)"""
    if rag_system is None:
        return [], "<p>❌ RAG system not initialized.</p>", "<p>❌ RAG system not initialized.</p>", history_state

    if not question or not question.strip():
        return [], "<p>Please enter a question.</p>", "<p></p>", history_state

    try:
        response = rag_system.query(
            user_query=question.strip(),
            retrieval_method=method,
            top_k=int(top_k),
            use_multiturn=bool(multiturn),
            use_agentic=bool(agentic),
        )

        answer = _safe_get(response, "answer", "") or ""
        original_query = _safe_get(response, "original_query", question.strip()) or question.strip()
        refined_query = _safe_get(response, "refined_query", "") or ""

        # Update UI history (NOT changing RAG internal conversation logic)
        new_turn = {
            "user": question.strip(),
            "answer": answer,
            "original_query": original_query,
            "refined_query": refined_query,
        }
        new_history = (history_state or []) + [new_turn]

        # chat_display = []
        # for t in new_history:
        #     chat_display.append(
        #         (t["user"], _format_chat_message(t["answer"], t["original_query"], t["refined_query"]))
        #     )
        chat_display = []
        for t in new_history:
            chat_display.append({"role": "user", "content": t["user"]})
            chat_display.append({
                "role": "assistant",
                "content": _format_chat_message(t["answer"], t["original_query"], t["refined_query"])
            })



        docs_html = build_docs_html(_safe_get(response, "retrieved_docs", []) or [])
        workflow_html = build_workflow_html(
            response=response,
            method=method,
            top_k=int(top_k),
            multiturn=bool(multiturn),
            agentic=bool(agentic),
        )

        return chat_display, docs_html, workflow_html, new_history

    except Exception as e:
        err = escape(str(e))
        return [], f"<p>❌ Error: {err}</p>", f"<p>❌ Error: {err}</p>", history_state


def clear_all_ui():
    """Clear conversation (both RAG internal state and UI panels)."""
    if rag_system is not None:
        rag_system.clear_conversation()
    return [], "<p>No query yet.</p>", "<p>Workflow will appear here.</p>", init_history()


def create_simple_ui():
    """Create enhanced Gradio UI (layout stays similar to the simplified one)"""
    methods = rag_system.retrieval_manager.list_available_methods()

    with gr.Blocks(title="RAG System") as demo:
        gr.Markdown(
            "# 🤖 RAG System - COMP5423 Group 14\n\n"
            "### LLM using: " + rag_system.using_model
        )

        history_state = gr.State(init_history())

        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot(
                    label="Dialogue",
                    height=420,
                    type="messages",      
                    allow_tags=False,     
                )


                question = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask something...",
                    lines=3,
                )

                with gr.Row():
                    method = gr.Dropdown(
                        choices=methods,
                        value=methods[0] if methods else None,
                        label="Retrieval Method",
                    )
                    top_k = gr.Slider(1, 20, 5, step=1, label="Top-K")

                with gr.Row():
                    multiturn = gr.Checkbox(label="Multi-turn (Feature A)")
                    agentic = gr.Checkbox(label="Agentic (Feature B)")

                with gr.Row():
                    submit = gr.Button("🚀 Ask", variant="primary")
                    clear = gr.Button("Clear")

            with gr.Column():
                docs = gr.HTML("<p>No query yet.</p>", label="Retrieved Documents")
                workflow = gr.HTML("<p>Workflow will appear here.</p>", label="Agentic Workflow / Reasoning Log")

        submit.click(
            query_rag,
            inputs=[question, method, top_k, multiturn, agentic, history_state],
            outputs=[chatbot, docs, workflow, history_state],
        )

        clear.click(
            clear_all_ui,
            inputs=None,
            outputs=[chatbot, docs, workflow, history_state],
        )

        gr.Examples(
            examples=[
                ["Which airport is located in Maine?"],
                ["Where is Barack Obama from?"],
            ],
            inputs=question,
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--backend", choices = ["local","api"],default="local")
    args = parser.parse_args()

    initialize_rag(args.model,gen_backend=args.backend)
    demo = create_simple_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
