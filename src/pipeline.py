"""End-to-end evaluation runner: retrieval → reranking → generation → judging."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.chunking import Chunk
from src.config import EVAL_CSV, EVAL_RESULTS_FILE, PRECISION_K, RECALL_K, TOP_K, TRACES_FILE
from src.cost import CostEstimate, estimate_cost
from src.generator import GenerationResult, generate_answer
from src.judge import FaithfulnessResult, judge_faithfulness
from src.metrics import RetrievalMetrics, compute_metrics
from src.reranker import NoOpReranker
from src.retriever import Retriever
from src.tracing import LatencyBreakdown, QueryTrace, save_trace


def run_pipeline(
    retriever: Retriever,
    reranker=None,
    eval_csv: Path = EVAL_CSV,
    results_path: Path = EVAL_RESULTS_FILE,
    traces_path: Path = TRACES_FILE,
    k: int = TOP_K,
    verbose: bool = True,
) -> list[dict]:
    """
    Run the full RAG evaluation loop over all queries in eval_set.csv.
    Returns a list of result dicts (one per query).
    """
    if reranker is None:
        reranker = NoOpReranker()

    eval_df = _load_eval_set(eval_csv)
    results: list[dict] = []

    for _, row in eval_df.iterrows():
        if verbose:
            print(f"  Evaluating: {row['query_id']} — {row['query'][:60]}...")

        result = _evaluate_query(row, retriever, reranker, k=k, traces_path=traces_path)
        results.append(result)

    _save_results(results, results_path)
    if verbose:
        print(f"\nSaved {len(results)} results → {results_path}")
    return results


def _evaluate_query(row: pd.Series, retriever: Retriever, reranker, k: int, traces_path: Path = TRACES_FILE) -> dict:
    query_id = str(row["query_id"])
    query = str(row["query"])
    relevant_ids = _parse_relevant_ids(str(row.get("relevant_chunk_ids", "")))

    t0 = time.perf_counter()

    # ── Retrieve ─────────────────────────────────────────────────────────────
    retrieved = retriever.retrieve(query)
    t_retrieve = time.perf_counter()

    # ── Rerank ───────────────────────────────────────────────────────────────
    reranked = reranker.rerank(query, retrieved)
    t_rerank = time.perf_counter()

    # ── Generate ─────────────────────────────────────────────────────────────
    gen: GenerationResult = generate_answer(query, reranked[:k])
    t_generate = time.perf_counter()

    # ── Judge ─────────────────────────────────────────────────────────────────
    judgment: FaithfulnessResult = judge_faithfulness(gen.answer, reranked[:k])
    t_judge = time.perf_counter()

    # ── Metrics ──────────────────────────────────────────────────────────────
    retrieved_ids = [c.chunk_id for c, _ in reranked]
    metrics: RetrievalMetrics = compute_metrics(retrieved_ids, relevant_ids, k=k)

    # ── Cost & latency ────────────────────────────────────────────────────────
    cost: CostEstimate = estimate_cost(gen.input_tokens, gen.output_tokens)
    latency = LatencyBreakdown(
        retrieval_ms=round((t_retrieve - t0) * 1000, 1),
        rerank_ms=round((t_rerank - t_retrieve) * 1000, 1),
        generation_ms=round((t_generate - t_rerank) * 1000, 1),
        judging_ms=round((t_judge - t_generate) * 1000, 1),
        total_ms=round((t_judge - t0) * 1000, 1),
    )

    # ── Trace ─────────────────────────────────────────────────────────────────
    trace = QueryTrace(
        query_id=query_id,
        query=query,
        retrieved_chunks=[
            {"chunk_id": c.chunk_id, "text": c.text[:300], "score": round(s, 4)}
            for c, s in retrieved
        ],
        reranked_chunks=[
            {"chunk_id": c.chunk_id, "text": c.text[:300], "score": round(s, 4)}
            for c, s in reranked
        ],
        generated_answer=gen.answer,
        cited_chunk_ids=gen.cited_chunk_ids,
        faithfulness_judgment=asdict(judgment),
        retrieval_metrics=asdict(metrics),
        latency=latency,
        cost=asdict(cost),
        model_used=gen.model_used,
        judge_model=judgment.judge_model,
    )
    save_trace(trace, path=traces_path)

    result = {
        "query_id": query_id,
        "query": query,
        "recall_at_k": metrics.recall_at_k,
        "precision_at_k": metrics.precision_at_k,
        "mrr": metrics.mrr,
        "ndcg_at_k": metrics.ndcg_at_k,
        "faithfulness_score": judgment.faithfulness_score,
        "verdict": judgment.verdict,
        "retrieval_ms": latency.retrieval_ms,
        "generation_ms": latency.generation_ms,
        "judging_ms": latency.judging_ms,
        "total_ms": latency.total_ms,
        "input_tokens": cost.input_tokens,
        "output_tokens": cost.output_tokens,
        "estimated_cost_usd": cost.estimated_cost_usd,
        "model_used": gen.model_used,
    }
    return result


def _load_eval_set(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Eval set not found: {path}\n"
            "Run  python scripts/sample_eval_set.py  to generate one,\n"
            "or edit data/eval/eval_set.csv manually."
        )
    df = pd.read_csv(path)
    required = {"query_id", "query", "relevant_chunk_ids"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"eval_set.csv is missing columns: {missing}")
    return df


def _parse_relevant_ids(raw: str) -> list[str]:
    """Parse pipe-separated chunk IDs, e.g. 'chunk_001|chunk_004'."""
    if not raw or str(raw).strip().lower() in ("", "nan", "none"):
        return []
    return [cid.strip() for cid in raw.split("|") if cid.strip()]


def _save_results(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
