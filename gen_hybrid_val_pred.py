import argparse
import json
from pathlib import Path

from retrieval_module import RetrievalManager


def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} examples from {path}")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Generate hybrid retrieval predictions on validation set."
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="data/validation.jsonl",
        help="Path to validation jsonl file.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing collection.jsonl.",
    )
    parser.add_argument(
        "--retrieval_model_dir",
        type=str,
        default="retrieval_model",
        help="Directory containing dense retrieval artifacts.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="val_hybrid_pred.jsonl",
        help="Where to save the prediction file.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of docs to retrieve per query.",
    )
    parser.add_argument(
        "--bm25_weight",
        type=float,
        default=0.5,
        help="Weight of BM25 in hybrid retriever.",
    )
    parser.add_argument(
        "--dense_weight",
        type=float,
        default=0.5,
        help="Weight of dense retriever.",
    )

    args = parser.parse_args()

    # Load validation data
    val_data = read_jsonl(args.val_path)
    total = len(val_data)

    # Initialize hybrid retriever
    manager = RetrievalManager(
        data_dir=args.data_dir,
        retrieval_model_dir=args.retrieval_model_dir,
    )

    manager.initialize_sparse(methods=["bm25"])
    manager.initialize_dense()

    manager.initialize_hybrid(
        {
            "bm25": args.bm25_weight,
            "dense": args.dense_weight,
        }
    )

    hybrid = manager.get_retriever("hybrid")

    print("\nRunning hybrid retrieval on validation set...")
    print(f"Total queries: {total}")

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fout:
        for i, ex in enumerate(val_data, start=1):

            qid = ex["id"]
            question = ex["text"]
            answer = ex.get("answer", "")

            results = hybrid.retrieve(question, top_k=args.top_k)

            retrieved_docs = [[r.doc_id, float(r.score)] for r in results]

            record = {
                "id": qid,
                "question": question,
                "answer": answer,
                "retrieved_docs": retrieved_docs,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Simple progress print
            if i % 50 == 0 or i == total:
                print(f"Processed {i}/{total} queries")

    print(f"\nFinished! Saved hybrid predictions to: {out_path}")


if __name__ == "__main__":
    main()
