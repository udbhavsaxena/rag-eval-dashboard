#!/usr/bin/env python
"""
Build the FAISS index from raw documents.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --model all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunking import chunk_documents, save_chunks
from src.embeddings import embed_chunks, load_model
from src.ingest import load_documents
from src.vector_store import FAISSStore


def main(model_name: str) -> None:
    print("=" * 60)
    print("RAG Eval Dashboard — Build Index")
    print("=" * 60)

    print("\n[1/4] Loading documents from data/raw/ ...")
    docs = load_documents()
    print(f"      Loaded {len(docs)} document(s).")
    for d in docs:
        print(f"        • {d.source}  ({len(d.text.split())} words)")

    print("\n[2/4] Chunking documents ...")
    chunks = chunk_documents(docs)
    save_chunks(chunks)
    print(f"      Created {len(chunks)} chunks.")

    print(f"\n[3/4] Embedding chunks with '{model_name}' ...")
    model = load_model(model_name)
    embeddings = embed_chunks(chunks, model=model)
    print(f"      Embedding shape: {embeddings.shape}")

    print("\n[4/4] Building and saving FAISS index ...")
    store = FAISSStore()
    store.build(chunks, embeddings)
    store.save()

    print("\nDone! Index is ready.")
    print("Next step:  python scripts/run_eval.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index for RAG eval.")
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name (default: all-MiniLM-L6-v2)",
    )
    args = parser.parse_args()
    main(args.model)
