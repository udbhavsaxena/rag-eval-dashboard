"""Embed chunks using sentence-transformers (runs fully locally)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.config import EMBEDDING_MODEL
from src.chunking import Chunk


def load_model(model_name: str = EMBEDDING_MODEL):
    """Lazy-load the SentenceTransformer model (downloads once, cached locally)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is not installed.\n"
            "Run:  pip install sentence-transformers"
        )
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def embed_chunks(
    chunks: Sequence[Chunk],
    model=None,
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """Return (N, D) float32 embedding matrix for *chunks*."""
    if model is None:
        model = load_model()
    texts = [c.text for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity via inner product
    )
    return embeddings.astype(np.float32)


def embed_query(query: str, model=None) -> np.ndarray:
    """Return (1, D) float32 embedding for a single query string."""
    if model is None:
        model = load_model()
    vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vec.astype(np.float32)
