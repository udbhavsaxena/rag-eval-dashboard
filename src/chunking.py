"""Split documents into overlapping chunks with stable IDs.

Three strategies are provided:

- ``word``     (default) — fixed-size word windows with overlap; fast but may
                           cut mid-sentence.
- ``sentence`` — groups whole sentences until the word budget is met; overlaps
                 by keeping the last N sentences in the next chunk.
- ``semantic`` — embeds every sentence, detects topic shifts via cosine-
                 similarity drops, and starts a new chunk at each shift.
                 Requires a sentence-transformers model to be passed in.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_FILE
from src.ingest import Document

# Abbreviations whose trailing period must NOT trigger a sentence split.
_ABBREVS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr",
    "vs", "i.e", "e.g", "etc", "fig", "approx",
    "no", "vol", "dept", "est",
}


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    title: str
    text: str
    chunk_index: int   # position within the document
    start_word: int    # word (or sentence) index of first token in this chunk
    end_word: int      # word (or sentence) index one past the last token


# ── Strategy 1: word-level (original) ────────────────────────────────────────

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


# ── Strategy 2: sentence-aware ────────────────────────────────────────────────

def chunk_documents_sentence(
    docs: Sequence[Document],
    chunk_size: int = CHUNK_SIZE,
    overlap_sentences: int = 2,
) -> list[Chunk]:
    """Return sentence-boundary-respecting chunks for every document.

    Sentences are accumulated until adding the next one would exceed
    *chunk_size* words.  The last *overlap_sentences* sentences of each chunk
    are prepended to the next one so context is not lost at boundaries.
    """
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(_chunk_document_sentence(doc, chunk_size, overlap_sentences))
    return chunks


def _chunk_document_sentence(doc: Document, size: int, overlap: int) -> list[Chunk]:
    sentences = _split_sentences(doc.text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    idx = 0
    i = 0  # index into sentences[]

    while i < len(sentences):
        group: list[str] = []
        word_count = 0
        j = i

        # Greedily accumulate sentences until the word budget runs out.
        while j < len(sentences):
            sent_words = len(sentences[j].split())
            if group and word_count + sent_words > size:
                break
            group.append(sentences[j])
            word_count += sent_words
            j += 1

        text = " ".join(group)
        chunks.append(Chunk(
            chunk_id=_stable_id(doc.doc_id, idx),
            doc_id=doc.doc_id,
            source=doc.source,
            title=doc.title,
            text=text,
            chunk_index=idx,
            start_word=i,
            end_word=j,
        ))
        idx += 1

        if j >= len(sentences):
            break
        # Overlap: start next chunk from (j - overlap), but always advance.
        i = max(i + 1, j - overlap)

    return chunks


# ── Strategy 3: semantic ──────────────────────────────────────────────────────

def chunk_documents_semantic(
    docs: Sequence[Document],
    model,
    chunk_size: int = CHUNK_SIZE,
    breakpoint_threshold: float = 0.4,
) -> list[Chunk]:
    """Return semantically coherent chunks using embedding similarity.

    Each sentence is embedded; when the cosine similarity between two adjacent
    sentences falls below *breakpoint_threshold* a topic shift is detected and a
    new chunk is started.  *chunk_size* acts as a hard upper bound so a single
    chunk never grows unbounded.

    Args:
        model: A loaded sentence-transformers model (already loaded by
               build_index.py, so no extra download is needed).
        breakpoint_threshold: Cosine-similarity value below which a new chunk
               begins.  Lower → more chunks, more granular.  Typical range
               0.3 – 0.6.
    """
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(_chunk_document_semantic(doc, model, chunk_size, breakpoint_threshold))
    return chunks


def _chunk_document_semantic(
    doc: Document, model, size: int, threshold: float
) -> list[Chunk]:
    import numpy as np  # only needed when this strategy is selected

    sentences = _split_sentences(doc.text)
    if not sentences:
        return []

    # Embed all sentences at once — much faster than one-by-one.
    embeddings = model.encode(
        sentences, normalize_embeddings=True, show_progress_bar=False, batch_size=64
    )

    chunks: list[Chunk] = []
    idx = 0
    group_start = 0
    group: list[str] = [sentences[0]]
    word_count = len(sentences[0].split())

    for i in range(1, len(sentences)):
        sim = float(np.dot(embeddings[i - 1], embeddings[i]))  # cosine (unit vecs)
        next_words = len(sentences[i].split())

        # Flush current group on topic-shift OR word-budget overflow.
        if sim < threshold or word_count + next_words > size:
            chunks.append(Chunk(
                chunk_id=_stable_id(doc.doc_id, idx),
                doc_id=doc.doc_id,
                source=doc.source,
                title=doc.title,
                text=" ".join(group),
                chunk_index=idx,
                start_word=group_start,
                end_word=i,
            ))
            idx += 1
            group_start = i
            group = [sentences[i]]
            word_count = next_words
        else:
            group.append(sentences[i])
            word_count += next_words

    # Flush last group.
    if group:
        chunks.append(Chunk(
            chunk_id=_stable_id(doc.doc_id, idx),
            doc_id=doc.doc_id,
            source=doc.source,
            title=doc.title,
            text=" ".join(group),
            chunk_index=idx,
            start_word=group_start,
            end_word=len(sentences),
        ))

    return chunks


# ── Strategy dispatcher ───────────────────────────────────────────────────────

def chunk_documents_strategy(
    docs: Sequence[Document],
    strategy: str = "word",
    model=None,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Dispatch to the appropriate chunking strategy.

    Args:
        strategy: ``"word"``, ``"sentence"``, or ``"semantic"``.
        model:    Required only for ``"semantic"`` — the loaded
                  sentence-transformers model.
    """
    if strategy == "word":
        return chunk_documents(docs, chunk_size=chunk_size, overlap=overlap)
    if strategy == "sentence":
        return chunk_documents_sentence(docs, chunk_size=chunk_size)
    if strategy == "semantic":
        if model is None:
            raise ValueError("semantic chunking requires 'model' to be provided")
        return chunk_documents_semantic(docs, model=model, chunk_size=chunk_size)
    raise ValueError(f"Unknown chunking strategy: {strategy!r}. "
                     f"Choose from 'word', 'sentence', 'semantic'.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_words(text: str) -> list[str]:
    # Preserve paragraph boundaries by normalising whitespace but keeping tokens
    return re.split(r"\s+", text.strip())


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences, respecting common abbreviations.

    Strategy: tokenise on any run of whitespace that follows a sentence-ending
    punctuation mark (.  !  ?) AND precedes an uppercase letter or digit.
    Tokens whose last word before the period is a known abbreviation are left
    intact so "e.g. hypertension" doesn't become two sentences.
    """
    # First, replace paragraph breaks with a special sentinel so they always
    # trigger a split regardless of the letter case that follows.
    normalised = re.sub(r"\n{2,}", " <PARA> ", text.strip())
    normalised = re.sub(r"\s+", " ", normalised)

    # Candidate split points: period/!/? followed by space + uppercase/digit.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\<])", normalised)

    sentences: list[str] = []
    for part in parts:
        # Re-expand paragraph sentinels into a split.
        sub = [s.strip() for s in part.split("<PARA>") if s.strip()]
        for s in sub:
            # Check if the sentence ends with a known abbreviation — if so,
            # merge it back with the *next* sentence (undo the split).
            if sentences and _ends_with_abbreviation(sentences[-1]):
                sentences[-1] = sentences[-1] + " " + s
            else:
                sentences.append(s)

    return [s for s in sentences if s]


def _ends_with_abbreviation(sentence: str) -> bool:
    """Return True when *sentence* ends with a known abbreviation word."""
    # Match the last word before a trailing period.
    m = re.search(r"(\w+)\.\s*$", sentence)
    if not m:
        return False
    return m.group(1).lower() in _ABBREVS


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
