#!/usr/bin/env python
"""
Remap eval_set.csv ground-truth chunk IDs for a new chunking strategy.

When you rebuild with sentence or semantic chunking the chunk IDs change
because the text boundaries differ.  This script finds the best-matching
chunk in the new strategy for every annotated word-strategy chunk ID, using
Jaccard similarity on word tokens.

Run AFTER building the new strategy's index:
    python scripts/remap_eval_set.py --to sentence
    python scripts/remap_eval_set.py --to semantic

Output:
    data/eval/eval_set_sentence.csv  (or _semantic.csv)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.config import (
    CHUNKER_STRATEGIES,
    EVAL_CSV,
    get_chunks_file,
)


def jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def load_chunks_dict(strategy: str) -> dict[str, dict]:
    """Return {chunk_id: chunk_dict} for *strategy*."""
    path = get_chunks_file(strategy)
    if not path.exists():
        raise FileNotFoundError(
            f"Chunks file not found for strategy '{strategy}': {path}\n"
            f"Run:  python scripts/build_index.py --chunker {strategy}"
        )
    chunks: dict[str, dict] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                c = json.loads(line)
                chunks[c["chunk_id"]] = c
    return chunks


def best_match(
    original_text: str,
    candidates: list[dict],
    doc_id: str,
) -> str | None:
    """Return the chunk_id from *candidates* that best overlaps *original_text*."""
    doc_candidates = [c for c in candidates if c["doc_id"] == doc_id]
    if not doc_candidates:
        return None
    scored = [(jaccard(original_text, c["text"]), c["chunk_id"]) for c in doc_candidates]
    scored.sort(reverse=True)
    return scored[0][1]


def remap(from_strategy: str, to_strategy: str) -> None:
    print(f"Remapping eval_set from '{from_strategy}' → '{to_strategy}' ...")

    source_chunks = load_chunks_dict(from_strategy)
    target_chunks = load_chunks_dict(to_strategy)
    target_list = list(target_chunks.values())

    eval_df = pd.read_csv(EVAL_CSV)

    remapped_rows = []
    for _, row in eval_df.iterrows():
        raw = str(row.get("relevant_chunk_ids", ""))
        original_ids = [cid.strip() for cid in raw.split("|") if cid.strip()
                        and cid.strip().lower() not in ("nan", "none", "")]

        new_ids: list[str] = []
        for orig_id in original_ids:
            if orig_id not in source_chunks:
                print(f"  [warn] chunk not found in source: {orig_id}")
                continue
            orig_text = source_chunks[orig_id]["text"]
            orig_doc = source_chunks[orig_id]["doc_id"]
            match = best_match(orig_text, target_list, orig_doc)
            if match:
                new_ids.append(match)
                print(f"  {orig_id}  →  {match}")
            else:
                print(f"  [warn] no match found for {orig_id}")

        remapped_rows.append({
            **row.to_dict(),
            "relevant_chunk_ids": "|".join(new_ids),
        })

    out_path = EVAL_CSV.parent / f"eval_set_{to_strategy}.csv"
    pd.DataFrame(remapped_rows).to_csv(out_path, index=False)
    print(f"\nSaved remapped eval set → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remap eval_set.csv chunk IDs for a new strategy.")
    parser.add_argument(
        "--from",
        dest="from_strategy",
        default="word",
        choices=[s for s in CHUNKER_STRATEGIES],
        help="Source strategy whose chunk IDs are currently in eval_set.csv (default: word)",
    )
    parser.add_argument(
        "--to",
        dest="to_strategy",
        required=True,
        choices=[s for s in CHUNKER_STRATEGIES if s != "word"],
        help="Target strategy to remap chunk IDs for",
    )
    args = parser.parse_args()
    remap(args.from_strategy, args.to_strategy)
