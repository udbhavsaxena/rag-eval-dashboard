"""Optional reranking step (no-op by default; swap in a cross-encoder here)."""

from __future__ import annotations

from src.chunking import Chunk


class NoOpReranker:
    """Pass-through reranker that preserves retrieval order."""

    def rerank(
        self,
        query: str,  # noqa: ARG002
        results: list[tuple[Chunk, float]],
    ) -> list[tuple[Chunk, float]]:
        return results


class CrossEncoderReranker:
    """Reranker using a sentence-transformers cross-encoder.

    Requires:  pip install sentence-transformers
    A reasonable model: cross-encoder/ms-marco-MiniLM-L-6-v2
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("pip install sentence-transformers")
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[tuple[Chunk, float]],
    ) -> list[tuple[Chunk, float]]:
        if not results:
            return results
        pairs = [(query, chunk.text) for chunk, _ in results]
        scores = self._model.predict(pairs)
        reranked = sorted(zip([c for c, _ in results], scores), key=lambda x: x[1], reverse=True)
        return [(chunk, float(score)) for chunk, score in reranked]


def get_reranker(use_cross_encoder: bool = False):
    """Factory: return the appropriate reranker based on config."""
    if use_cross_encoder:
        return CrossEncoderReranker()
    return NoOpReranker()
