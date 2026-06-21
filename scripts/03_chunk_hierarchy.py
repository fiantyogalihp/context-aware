#!/usr/bin/env python3
"""Build hierarchy-preserving chunks from extracted or starter documents."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parser import build_chunks, read_jsonl, write_jsonl

EXTRACTED = ROOT / "dataset" / "processed" / "documents_extracted.jsonl"
STARTER = ROOT / "dataset" / "processed" / "documents.jsonl"
OUT = ROOT / "dataset" / "processed" / "chunks.jsonl"
LEGACY_OUT = ROOT / "dataset" / "processed" / "chunks_rebuilt.jsonl"


def main() -> int:
    src = EXTRACTED if EXTRACTED.exists() and EXTRACTED.stat().st_size > 0 else STARTER
    chunks = build_chunks(read_jsonl(src))
    write_jsonl(OUT, chunks)
    write_jsonl(LEGACY_OUT, chunks)
    print(f"Wrote {len(chunks)} chunks to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
