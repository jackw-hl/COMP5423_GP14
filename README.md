# COMP5423 Group 14 - RAG System

Integrated Retrieval-Augmented Generation system for HotpotQA dataset.

## Project Structure

```
├── data/                           # HotpotQA dataset files
│   ├── collection.jsonl           # 144K document collection
│   ├── train.jsonl                # 12K training samples
│   ├── validation.jsonl           # 1.5K validation samples  
│   └── test.jsonl                 # 1K test samples
├── retrieval_model/               # Pre-computed embeddings
│   ├── doc_embs.npy              # GTE-ModernBERT embeddings (dim 768)
│   ├── doc_ids.json              # Document ID mapping
│   └── meta.json                 # Model metadata
├── eval/                          # Evaluation scripts (existing)
│   ├── eval_hotpotqa.py          # QA metrics (EM, F1)
│   └── eval_retrieval.py         # Retrieval metrics (MAP, NDCG)
├── generate module.ipynb          # Original notebook (reference)
├── generation_module.py           # Feature A & B (from notebook)
├── retrieval_module.py            # Multiple retrieval methods
├── rag_system.py                  # Integrated RAG system
└── app.py                         # Web UI with all features
```

## Features

### 1.1 Multiple Retrieval Methods
- **BM25**: Sparse lexical matching
- **TF-IDF**: Term frequency-inverse document frequency
- **Dense**: GTE-ModernBERT semantic embeddings (pre-computed)
- **Hybrid**: Weighted combination of BM25 + Dense

### 1.2 RAG System
- **Basic RAG**: Single-turn question answering
- **Feature A**: Multi-turn conversation with query refinement
- **Feature B**: ReAct agentic workflow (Plan-Action-Reflect)

### 1.3 Web UI
- Interactive Gradio interface with workflow visualization
- Supports all retrieval methods and features

## Setup

### Prerequisites
- Python 3.9+
- Virtual environment recommended

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd Group14
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download pre-computed embeddings** (if not included)
   - Place GTE embeddings in `retrieval_model/`
   - Required files: `doc_embs.npy`, `doc_ids.json`, `meta.json`

## Quick Start

### Launch Web UI
```bash
source .venv/bin/activate  # If not already activated
python app.py

# Options:
# --model Qwen/Qwen2.5-0.5B-Instruct  (fast, default)
# --model Qwen/Qwen2.5-1.5B-Instruct  (better quality)
# --port 7860                          (default port)
# --backend 'api' or 'local' (defaut local)
```

Opens at **http://127.0.0.1:7860**

### Using the Web Interface
The web UI provides:
- **Query Input**: Enter your question
- **Retrieval Method**: Choose BM25, Dense (GTE), or Hybrid
- **Top-K**: Number of documents to retrieve (1-20)
- **Multi-turn**: Enable conversation mode (Feature A)
- **Agentic**: Enable ReAct workflow (Feature B)

### Evaluation
Use the evaluation scripts in `eval/` directory:
```bash
# QA metrics
python eval/eval_hotpotqa.py \
  --gold data/test.jsonl \
  --pred your_predictions.jsonl

# Retrieval metrics  
python eval/eval_retrieval.py \
  --gold data/test.jsonl \
  --pred your_predictions.jsonl
```

## Implementation Details

### Generation Module (`generation_module.py`)
Based on `generate module.ipynb`, implements:
- `QwenGenerator`: Answer generation using Qwen2.5 models
- `MultiTurnDialogManager`: Conversation history tracking
- `QueryRefinementEngine`: Query rewriting for multi-turn
- `MultiTurnRetrievalOptimizer`: Context-aware retrieval
- `ReActEngine`: Plan-Action-Reflect agentic workflow

### Retrieval Module (`retrieval_module.py`)
- Abstract `BaseRetriever` interface
- `SparseRetriever`: BM25 and TF-IDF implementations
- `DenseRetriever`: Loads pre-computed GTE embeddings
- `HybridRetriever`: Weighted score fusion
- `RetrievalManager`: Orchestrates all methods

### RAG System (`rag_system.py`)
Integrates retrieval + generation:
- `query()`: Main RAG interface
- `multiturn_query()`: Conversation support
- `batch_query()`: Batch processing
- `evaluate_on_dataset()`: Dataset evaluation

## Models

### Retrieval
- **GTE-ModernBERT**: `Alibaba-NLP/gte-modernbert-base` (768-dim)

### Generation
- **Qwen2.5-0.5B-Instruct**: Fast demo model
- **Qwen2.5-1.5B-Instruct**: Balanced performance
- **Qwen2.5-3B**: Higher quality (slower)
- **Qwen2.5-7B-Instruct**: Best quality (requires more memory)

## Notes

- Pre-computed embeddings in `retrieval_model/` avoid re-encoding 144K documents
- Existing evaluation scripts in `eval/` are preserved and used
- Web UI supports all features with workflow visualization
- System outputs predictions in format compatible with evaluation scripts

## Group Members
Group 14 - COMP5423 Fall 2025
