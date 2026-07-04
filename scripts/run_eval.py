#!/usr/bin/env python
"""
Run the full RAG evaluation pipeline.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --chunker sentence
    python scripts/run_eval.py --chunker semantic --rerank
    python scripts/run_eval.py --compare          # show comparison across built strategies
    python scripts/run_eval.py --k 5 --fresh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    CHUNKER_STRATEGIES,
    TOP_K,
    get_chunk_map_path,
    get_eval_csv,
    get_eval_results_file,
    get_index_path,
    get_traces_file,
)
from src.embeddings import load_model
from src.metrics import RetrievalMetrics, aggregate_metrics
from src.pipeline import run_pipeline
from src.reranker import get_reranker
from src.retriever import Retriever
from src.tracing import clear_traces


def main(k: int, fresh: bool, use_cross_encoder: bool, chunker: str) -> dict:
    """Run evaluation for one chunking strategy; return aggregate metric dict."""
    traces_file = get_traces_file(chunker)
    results_file = get_eval_results_file(chunker)

    if fresh:
        clear_traces(traces_file)
        print(f"Cleared previous traces for strategy='{chunker}'.")

    index_path = get_index_path(chunker)
    if not index_path.exists():
        print(f"\n[!] No FAISS index found for strategy='{chunker}'.")
        print(f"    Run:  python scripts/build_index.py --chunker {chunker}")
        return {}

    eval_csv = get_eval_csv(chunker)
    if not eval_csv.exists():
        print(f"\n[!] No eval set found for strategy='{chunker}': {eval_csv}")
        print(f"    Run:  python scripts/remap_eval_set.py --to {chunker}")
        return {}

    print(f"\nLoading embedding model and FAISS index  (strategy='{chunker}') ...")
    model = load_model()
    retriever = Retriever.from_disk(
        top_k=k,
        model=model,
        index_path=index_path,
        chunk_map_path=get_chunk_map_path(chunker),
    )
    reranker = get_reranker(use_cross_encoder=use_cross_encoder)

    print(f"Reranker: {'CrossEncoder' if use_cross_encoder else 'NoOp (pass-through)'}")
    print(f"Top-K:    {k}\n")

    results = run_pipeline(
        retriever=retriever,
        reranker=reranker,
        eval_csv=eval_csv,
        k=k,
        verbose=True,
        results_path=results_file,
        traces_path=traces_file,
    )

    rm_list = [
        RetrievalMetrics(
            recall_at_k=r["recall_at_k"],
            precision_at_k=r["precision_at_k"],
            mrr=r["mrr"],
            ndcg_at_k=r["ndcg_at_k"],
            k=k,
            retrieved_ids=[],
            relevant_ids=[],
            hits=[],
        )
        for r in results
    ]
    agg = aggregate_metrics(rm_list)

    avg_faith = sum(r["faithfulness_score"] for r in results) / len(results)
    avg_cost = sum(r["estimated_cost_usd"] for r in results)
    avg_lat = sum(r["total_ms"] for r in results) / len(results)

    print("\n" + "=" * 60)
    print(f"Results — strategy='{chunker}'")
    print("=" * 60)
    print(f"  Recall@{k}:    {agg['recall_at_k']:.4f}")
    print(f"  Precision@{k}: {agg['precision_at_k']:.4f}")
    print(f"  MRR:         {agg['mrr']:.4f}")
    print(f"  nDCG@{k}:     {agg['ndcg_at_k']:.4f}")
    print(f"  Faithfulness: {avg_faith:.4f}")
    print(f"  Avg Latency:  {avg_lat:.1f} ms")
    print(f"  Total Cost:   ${avg_cost:.6f}")

    return {
        "strategy": chunker,
        "recall_at_k": agg["recall_at_k"],
        "precision_at_k": agg["precision_at_k"],
        "mrr": agg["mrr"],
        "ndcg_at_k": agg["ndcg_at_k"],
        "faithfulness": avg_faith,
        "avg_latency_ms": avg_lat,
        "total_cost_usd": avg_cost,
    }


def compare(k: int) -> None:
    """Load existing result files for all built strategies and print a table."""
    import json

    print("\n" + "=" * 60)
    print("Chunking Strategy Comparison")
    print("=" * 60)

    rows = []
    for strategy in CHUNKER_STRATEGIES:
        path = get_eval_results_file(strategy)
        if not path.exists():
            print(f"  [{strategy}] No results found — skipping.")
            continue

        results = []
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    results.append(json.loads(line))

        if not results:
            continue

        n = len(results)
        row = {
            "Strategy": strategy,
            f"Recall@{k}": sum(r["recall_at_k"] for r in results) / n,
            f"Prec@{k}":   sum(r["precision_at_k"] for r in results) / n,
            "MRR":         sum(r["mrr"] for r in results) / n,
            f"nDCG@{k}":   sum(r["ndcg_at_k"] for r in results) / n,
            "Faith.":      sum(r["faithfulness_score"] for r in results) / n,
            "Avg ms":      sum(r["total_ms"] for r in results) / n,
        }
        rows.append(row)

    if not rows:
        print("  No results available. Run run_eval.py --chunker <strategy> first.")
        return

    # Pretty-print table
    headers = list(rows[0].keys())
    col_w = {h: max(len(h), max(len(_fmt(r[h])) for r in rows)) for h in headers}

    header_line = "  " + "  ".join(h.ljust(col_w[h]) for h in headers)
    print(header_line)
    print("  " + "-" * (len(header_line) - 2))
    for row in rows:
        print("  " + "  ".join(_fmt(row[h]).ljust(col_w[h]) for h in headers))

    print("\nOpen the dashboard:  streamlit run app.py")


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation pipeline.")
    parser.add_argument("--k", type=int, default=TOP_K, help="Top-K chunks to retrieve")
    parser.add_argument("--fresh", action="store_true", help="Clear previous traces before running")
    parser.add_argument("--rerank", action="store_true", help="Use CrossEncoder reranker")
    parser.add_argument(
        "--chunker",
        choices=list(CHUNKER_STRATEGIES),
        default="word",
        help="Chunking strategy whose index to load (default: word)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print a side-by-side comparison of all available strategy results",
    )
    args = parser.parse_args()

    if args.compare:
        compare(k=args.k)
    else:
        main(k=args.k, fresh=args.fresh, use_cross_encoder=args.rerank, chunker=args.chunker)
        print("\nOpen the dashboard:  streamlit run app.py")
