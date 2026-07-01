"""Split documents into overlapping chunks with stable IDs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_FILE
from src.ingest import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    title: str
    text: str
    chunk_index: int   # position within the document
    start_word: int
    end_word: int


def chunk_documents(
    docs: Sequence[Document],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Return overlapping word-level chunks for every document."""
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(_chunk_document(doc, chunk_size, overlap))
    return chunks


def _chunk_document(doc: Document, size: int, overlap: int) -> list[Chunk]:
    words = _split_words(doc.text)
    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(words):
        end = min(start + size, len(words))
        text = " ".join(words[start:end])
        chunk_id = _stable_id(doc.doc_id, idx)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                source=doc.source,
                title=doc.title,
                text=text,
                chunk_index=idx,
                start_word=start,
                end_word=end,
            )
        )
        idx += 1
        if end == len(words):
            break
        start = end - overlap

    return chunks


def _split_words(text: str) -> list[str]:
    # Preserve paragraph boundaries by normalising whitespace but keeping tokens
    return re.split(r"\s+", text.strip())


def _stable_id(doc_id: str, idx: int) -> str:
    """Deterministic chunk ID based on doc_id and position."""
    raw = f"{doc_id}::{idx}"
    digest = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"{doc_id}_chunk{idx:04d}_{digest}"


def save_chunks(chunks: list[Chunk], path: Path = CHUNKS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk)) + "\n")
    print(f"Saved {len(chunks)} chunks → {path}")


def load_chunks(path: Path = CHUNKS_FILE) -> list[Chunk]:
    if not path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {path}\n"
            "Run  python scripts/build_index.py  first."
        )
    chunks = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(Chunk(**json.loads(line)))
    return chunks
