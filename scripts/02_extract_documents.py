#!/usr/bin/env python3
"""Extract downloaded raw sources into dataset/processed/documents_extracted.jsonl."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parser import extract_documents_from_catalog, write_jsonl

CATALOG = ROOT / "dataset" / "source_catalog.csv"
OUT = ROOT / "dataset" / "processed" / "documents_extracted.jsonl"


def main() -> int:
    docs = extract_documents_from_catalog(CATALOG, ROOT)
    write_jsonl(OUT, docs)
    print(f"Wrote {len(docs)} extracted document records to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
