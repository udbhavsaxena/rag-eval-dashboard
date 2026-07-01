"""Central configuration for the RAG eval pipeline."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"
EVAL_CSV = DATA_DIR / "eval" / "eval_set.csv"
RUNS_DIR = ROOT / "runs"
TRACES_FILE = RUNS_DIR / "traces.jsonl"
EVAL_RESULTS_FILE = RUNS_DIR / "eval_results.jsonl"
INDEX_PATH = EMBEDDINGS_DIR / "faiss.index"
CHUNK_MAP_PATH = EMBEDDINGS_DIR / "chunk_map.jsonl"

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 400          # tokens (approximated as words)
CHUNK_OVERLAP: int = 80

# ── Embedding ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"   # fast, local, no API key needed

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K: int = 5

# ── Generation ───────────────────────────────────────────────────────────────
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── Cost model (OpenAI gpt-4o-mini prices as of mid-2024) ────────────────────
# Prices per 1 000 tokens in USD
COST_PER_1K_INPUT: float = 0.00015
COST_PER_1K_OUTPUT: float = 0.00060

# ── Evaluation ───────────────────────────────────────────────────────────────
RECALL_K: int = 5
PRECISION_K: int = 5
MRR_K: int = 10
NDCG_K: int = 10
