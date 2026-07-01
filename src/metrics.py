"""Retrieval quality metrics: Recall@K, Precision@K, MRR, nDCG@K."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    k: int
    retrieved_ids: list[str]
    relevant_ids: list[str]
    hits: list[str]          # intersection


def compute_metrics(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 5,
) -> RetrievalMetrics:
    """Compute retrieval metrics for a single query.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs (position matters for MRR/nDCG).
        relevant_ids:  Ground-truth relevant chunk IDs (from eval_set.csv).
        k:             Cutoff for all metrics.
    """
    if not relevant_ids:
        return RetrievalMetrics(
            recall_at_k=0.0,
            precision_at_k=0.0,
            mrr=0.0,
            ndcg_at_k=0.0,
            k=k,
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            hits=[],
        )

    retrieved_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = [cid for cid in retrieved_k if cid in relevant_set]

    recall = len(hits) / len(relevant_set)
    precision = len(hits) / k if k > 0 else 0.0

    mrr = _mrr(retrieved_k, relevant_set)
    ndcg = _ndcg(retrieved_k, relevant_set, k)

    return RetrievalMetrics(
        recall_at_k=round(recall, 4),
        precision_at_k=round(precision, 4),
        mrr=round(mrr, 4),
        ndcg_at_k=round(ndcg, 4),
        k=k,
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
        hits=hits,
    )


def aggregate_metrics(all_metrics: list[RetrievalMetrics]) -> dict[str, float]:
    """Return macro-averaged metrics across all queries."""
    if not all_metrics:
        return {}
    keys = ["recall_at_k", "precision_at_k", "mrr", "ndcg_at_k"]
    return {
        key: round(sum(getattr(m, key) for m in all_metrics) / len(all_metrics), 4)
        for key in keys
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def _ndcg(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, cid in enumerate(retrieved[:k], start=1)
        if cid in relevant
    )
    # Ideal DCG: all relevant docs appear at the top positions
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
