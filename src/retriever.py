"""Retrieve top-k chunks for a query."""

from __future__ import annotations

from src.chunking import Chunk
from src.config import TOP_K
from src.embeddings import embed_query
from src.vector_store import FAISSStore


class Retriever:
    def __init__(self, store: FAISSStore, model=None, top_k: int = TOP_K) -> None:
        self.store = store
        self.model = model
        self.top_k = top_k

    def retrieve(self, query: str) -> list[tuple[Chunk, float]]:
        """Return [(chunk, similarity_score), ...] for *query*."""
        vec = embed_query(query, self.model)
        return self.store.search(vec, self.top_k)

    @classmethod
    def from_disk(cls, top_k: int = TOP_K, model=None) -> "Retriever":
        store = FAISSStore.load()
        return cls(store=store, model=model, top_k=top_k)
