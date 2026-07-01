"""Load raw documents from data/raw into Document objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from src.config import RAW_DIR


@dataclass
class Document:
    doc_id: str
    source: str        # relative path from data/raw
    title: str
    text: str
    metadata: dict = field(default_factory=dict)


def load_documents(raw_dir: Path = RAW_DIR) -> list[Document]:
    """Read all .txt and .md files from *raw_dir* and return Document objects."""
    docs: list[Document] = []
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}\n"
            "Create data/raw/ and add .txt or .md guideline files."
        )

    for path in sorted(_iter_files(raw_dir)):
        rel = path.relative_to(raw_dir)
        doc_id = _path_to_id(rel)
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        docs.append(
            Document(
                doc_id=doc_id,
                source=str(rel),
                title=path.stem.replace("_", " ").replace("-", " ").title(),
                text=text,
            )
        )

    if not docs:
        raise ValueError(
            f"No .txt or .md files found in {raw_dir}.\n"
            "Add clinical guideline text files and re-run build_index.py."
        )
    return docs


def _iter_files(directory: Path) -> Iterator[Path]:
    for suffix in ("*.txt", "*.md"):
        yield from directory.rglob(suffix)


def _path_to_id(rel: Path) -> str:
    return rel.with_suffix("").as_posix().replace("/", "_").replace(" ", "_").lower()
