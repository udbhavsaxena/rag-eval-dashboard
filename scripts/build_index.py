#!/usr/bin/env python
"""
Build the FAISS index from raw documents.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --chunker sentence
    python scripts/build_index.py --chunker semantic
    python scripts/build_index.py --model all-MiniLM-L6-v2 --chunker word
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunking import chunk_documents_strategy
from src.config import (
    CHUNKER_STRATEGIES,
    get_chunk_map_path,
    get_chunks_file,
    get_index_path,
)
from src.embeddings import embed_chunks, load_model
from src.ingest import load_documents
from src.vector_store import FAISSStore


def main(model_name: str, strategy: str) -> None:
    print("=" * 60)
    print(f"RAG Eval Dashboard — Build Index  [chunker={strategy}]")
    print("=" * 60)

    print("\n[1/4] Loading documents from data/raw/ ...")
    docs = load_documents()
    print(f"      Loaded {len(docs)} document(s).")
    for d in docs:
        print(f"        • {d.source}  ({len(d.text.split())} words)")

    # The embedding model is needed for step 3 always; for semantic chunking it
    # is also needed in step 2, so we load it before chunking.
    print(f"\n[2a]  Loading embedding model '{model_name}' ...")
    model = load_model(model_name)

    print(f"\n[2/4] Chunking documents  (strategy='{strategy}') ...")
    chunks = chunk_documents_strategy(docs, strategy=strategy, model=model)

    chunks_file = get_chunks_file(strategy)
    _save_chunks(chunks, chunks_file)
    print(f"      Created {len(chunks)} chunks.")

    print(f"\n[3/4] Embedding chunks with '{model_name}' ...")
    embeddings = embed_chunks(chunks, model=model)
    print(f"      Embedding shape: {embeddings.shape}")

    print("\n[4/4] Building and saving FAISS index ...")
    store = FAISSStore()
    store.build(chunks, embeddings)
    store.save(
        index_path=get_index_path(strategy),
        chunk_map_path=get_chunk_map_path(strategy),
    )

    print("\nDone! Index is ready.")
    print(f"Next step:  python scripts/run_eval.py --chunker {strategy}")


def _save_chunks(chunks, path: Path) -> None:
    import json
    from dataclasses import asdict

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk)) + "\n")
    print(f"      Saved {len(chunks)} chunks → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index for RAG eval.")
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--chunker",
        choices=list(CHUNKER_STRATEGIES),
        default="word",
        help="Chunking strategy to use (default: word)",
    )
    args = parser.parse_args()
    main(args.model, args.chunker)
