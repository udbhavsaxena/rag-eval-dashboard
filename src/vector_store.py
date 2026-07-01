"""FAISS-backed vector store with persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from src.chunking import Chunk
from src.config import CHUNK_MAP_PATH, INDEX_PATH


class FAISSStore:
    """Thin wrapper around faiss.IndexFlatIP for normalized embeddings."""

    def __init__(self) -> None:
        self._index = None
        self._chunks: list[Chunk] = []

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        import faiss  # imported lazily so the module loads without faiss installed

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)   # inner product == cosine on unit vecs
        self._index.add(embeddings)
        self._chunks = list(chunks)

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(
        self,
        index_path: Path = INDEX_PATH,
        chunk_map_path: Path = CHUNK_MAP_PATH,
    ) -> None:
        import faiss

        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))

        with chunk_map_path.open("w", encoding="utf-8") as fh:
            for chunk in self._chunks:
                fh.write(
                    json.dumps(
                        {
                            "chunk_id": chunk.chunk_id,
                            "doc_id": chunk.doc_id,
                            "source": chunk.source,
                            "title": chunk.title,
                            "text": chunk.text,
                            "chunk_index": chunk.chunk_index,
                            "start_word": chunk.start_word,
                            "end_word": chunk.end_word,
                        }
                    )
                    + "\n"
                )
        print(f"Saved FAISS index → {index_path}")
        print(f"Saved chunk map   → {chunk_map_path}")

    # ── Load ──────────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        index_path: Path = INDEX_PATH,
        chunk_map_path: Path = CHUNK_MAP_PATH,
    ) -> "FAISSStore":
        import faiss

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}\n"
                "Run  python scripts/build_index.py  first."
            )

        store = cls()
        store._index = faiss.read_index(str(index_path))

        with chunk_map_path.open(encoding="utf-8") as fh:
            store._chunks = [Chunk(**json.loads(line)) for line in fh if line.strip()]

        return store

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Return [(chunk, score), ...] sorted by descending similarity."""
        scores, indices = self._index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._chunks[idx], float(score)))
        return results

    def __len__(self) -> int:
        return len(self._chunks)
