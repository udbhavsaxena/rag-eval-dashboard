"""Retrieve top-k chunks for a query."""

from __future__ import annotations

from pathlib import Path

from src.chunking import Chunk
from src.config import CHUNK_MAP_PATH, INDEX_PATH, TOP_K
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
    def from_disk(
        cls,
        top_k: int = TOP_K,
        model=None,
        index_path: Path | None = None,
        chunk_map_path: Path | None = None,
    ) -> "Retriever":
        store = FAISSStore.load(
            index_path=index_path or INDEX_PATH,
            chunk_map_path=chunk_map_path or CHUNK_MAP_PATH,
        )
        return cls(store=store, model=model, top_k=top_k)
