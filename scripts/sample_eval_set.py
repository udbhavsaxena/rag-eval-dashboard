#!/usr/bin/env python
"""
Generate a starter eval_set.csv and print chunk IDs so you can annotate it.

Usage:
    python scripts/sample_eval_set.py          # print chunk IDs
    python scripts/sample_eval_set.py --create # write data/eval/eval_set.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.chunking import load_chunks
from src.config import EVAL_CSV


SAMPLE_QUERIES = [
    {
        "query_id": "q001",
        "query": "What are the recommended blood pressure targets for hypertensive patients?",
        "relevant_chunk_ids": "",
        "expected_answer_notes": "Should mention systolic/diastolic targets, possibly <130/80 for high-risk patients",
    },
    {
        "query_id": "q002",
        "query": "What lifestyle modifications are recommended for type 2 diabetes management?",
        "relevant_chunk_ids": "",
        "expected_answer_notes": "Diet, exercise, weight loss, smoking cessation",
    },
    {
        "query_id": "q003",
        "query": "What are the first-line antibiotic choices for community-acquired pneumonia?",
        "relevant_chunk_ids": "",
        "expected_answer_notes": "Amoxicillin or macrolide for outpatients; beta-lactam + macrolide for inpatients",
    },
    {
        "query_id": "q004",
        "query": "When should statin therapy be initiated for cardiovascular risk reduction?",
        "relevant_chunk_ids": "",
        "expected_answer_notes": "Based on 10-year CVD risk score, LDL levels, and risk factors",
    },
    {
        "query_id": "q005",
        "query": "What screening tests are recommended for colorectal cancer?",
        "relevant_chunk_ids": "",
        "expected_answer_notes": "Colonoscopy, stool tests, flexible sigmoidoscopy; starting age 45",
    },
    {
        "query_id": "q006",
        "query": "What are the diagnostic criteria for metabolic syndrome?",
        "relevant_chunk_ids": "",
        "expected_answer_notes": "Waist circumference, triglycerides, HDL, BP, fasting glucose",
    },
    {
        "query_id": "q007",
        "query": "How should asthma severity be classified?",
        "relevant_chunk_ids": "",
        "expected_answer_notes": "Intermittent, mild/moderate/severe persistent based on frequency and lung function",
    },
    {
        "query_id": "q008",
        "query": "What are the recommended vaccinations for adults over 65?",
        "relevant_chunk_ids": "",
        "expected_answer_notes": "Influenza, pneumococcal, shingles (zoster), COVID-19 boosters",
    },
    {
        "query_id": "q009",
        "query": "What is the initial treatment approach for acute low back pain?",
        "relevant_chunk_ids": "",
        "expected_answer_notes": "NSAIDs, heat, continued activity, avoiding bed rest; imaging not recommended early",
    },
    {
        "query_id": "q010",
        "query": "What are the criteria for diagnosing heart failure with reduced ejection fraction?",
        "relevant_chunk_ids": "",
        "expected_answer_notes": "EF < 40%, symptoms, echocardiography",
    },
]


def print_chunks_for_annotation() -> None:
    """Help you find the right chunk IDs for annotation."""
    chunks = load_chunks()
    print(f"\nFound {len(chunks)} chunks. Listing first 50 with a text preview:\n")
    print(f"{'CHUNK ID':<50}  {'PREVIEW'}")
    print("-" * 100)
    for chunk in chunks[:50]:
        preview = chunk.text[:80].replace("\n", " ")
        print(f"{chunk.chunk_id:<50}  {preview}")

    print("\n" + "=" * 60)
    print("Search chunks by keyword:")
    print("  python -c \"")
    print("  import sys; sys.path.insert(0,'.')")
    print("  from src.chunking import load_chunks")
    print("  chunks = load_chunks()")
    print("  kw = 'blood pressure'")
    print("  for c in chunks:")
    print("      if kw.lower() in c.text.lower():")
    print("          print(c.chunk_id, c.text[:100])")
    print("  \"")


def create_eval_csv() -> None:
    EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(SAMPLE_QUERIES)
    df.to_csv(EVAL_CSV, index=False)
    print(f"Created eval set → {EVAL_CSV}")
    print(f"Rows: {len(df)}")
    print("\nNext: edit data/eval/eval_set.csv to fill in 'relevant_chunk_ids'")
    print("Use  python scripts/sample_eval_set.py  (without --create) to browse chunk IDs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--create",
        action="store_true",
        help="Write the sample eval_set.csv file",
    )
    args = parser.parse_args()

    if args.create:
        create_eval_csv()
    else:
        try:
            print_chunks_for_annotation()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Run  python scripts/build_index.py  first.")
