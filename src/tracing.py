"""Structured per-query trace logging to runs/traces.jsonl."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import TRACES_FILE


@dataclass
class LatencyBreakdown:
    retrieval_ms: float
    rerank_ms: float
    generation_ms: float
    judging_ms: float
    total_ms: float


@dataclass
class QueryTrace:
    query_id: str
    query: str
    retrieved_chunks: list[dict]       # [{chunk_id, text, score}, ...]
    reranked_chunks: list[dict]
    generated_answer: str
    cited_chunk_ids: list[str]
    faithfulness_judgment: dict        # FaithfulnessResult as dict
    retrieval_metrics: dict            # RetrievalMetrics as dict
    latency: LatencyBreakdown
    cost: dict                         # CostEstimate as dict
    model_used: str
    judge_model: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    extra: dict[str, Any] = field(default_factory=dict)


def save_trace(trace: QueryTrace, path: Path = TRACES_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_trace_to_dict(trace)) + "\n")


def load_traces(path: Path = TRACES_FILE) -> list[dict]:
    if not path.exists():
        return []
    traces = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


def clear_traces(path: Path = TRACES_FILE) -> None:
    if path.exists():
        path.unlink()


def _trace_to_dict(trace: QueryTrace) -> dict:
    d = asdict(trace)
    # Flatten nested dataclass fields that asdict already handles
    return d
