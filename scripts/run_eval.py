#!/usr/bin/env python
"""
Run the full RAG evaluation pipeline.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --k 5 --fresh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import TOP_K
from src.embeddings import load_model
from src.metrics import aggregate_metrics, RetrievalMetrics
from src.pipeline import run_pipeline
from src.reranker import get_reranker
from src.retriever import Retriever
from src.tracing import clear_traces, TRACES_FILE


def main(k: int, fresh: bool, use_cross_encoder: bool) -> None:
    print("=" * 60)
    print("RAG Eval Dashboard — Run Evaluation")
    print("=" * 60)

    if fresh:
        clear_traces()
        print("Cleared previous traces.")

    print("\nLoading embedding model and FAISS index ...")
    model = load_model()
    retriever = Retriever.from_disk(top_k=k, model=model)
    reranker = get_reranker(use_cross_encoder=use_cross_encoder)

    print(f"Reranker: {'CrossEncoder' if use_cross_encoder else 'NoOp (pass-through)'}")
    print(f"Top-K:    {k}\n")

    results = run_pipeline(
        retriever=retriever,
        reranker=reranker,
        k=k,
        verbose=True,
    )

    # Print aggregate summary
    from src.metrics import RetrievalMetrics
    import dataclasses

    # Rebuild RetrievalMetrics objects for aggregation
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

    print("\n" + "=" * 60)
    print("Aggregate Results")
    print("=" * 60)
    print(f"  Recall@{k}:    {agg['recall_at_k']:.4f}")
    print(f"  Precision@{k}: {agg['precision_at_k']:.4f}")
    print(f"  MRR:         {agg['mrr']:.4f}")
    print(f"  nDCG@{k}:     {agg['ndcg_at_k']:.4f}")
    avg_faith = sum(r["faithfulness_score"] for r in results) / len(results)
    avg_cost = sum(r["estimated_cost_usd"] for r in results)
    avg_lat = sum(r["total_ms"] for r in results) / len(results)
    print(f"  Faithfulness: {avg_faith:.4f}")
    print(f"  Avg Latency:  {avg_lat:.1f} ms")
    print(f"  Total Cost:   ${avg_cost:.6f}")
    print("\nOpen the dashboard:  streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation pipeline.")
    parser.add_argument("--k", type=int, default=TOP_K, help="Top-K chunks to retrieve")
    parser.add_argument("--fresh", action="store_true", help="Clear previous traces before running")
    parser.add_argument("--rerank", action="store_true", help="Use CrossEncoder reranker")
    args = parser.parse_args()
    main(k=args.k, fresh=args.fresh, use_cross_encoder=args.rerank)
