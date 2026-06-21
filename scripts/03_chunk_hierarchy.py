#!/usr/bin/env python3
"""Build hierarchy-preserving chunks from extracted or starter documents."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parser import build_chunks, read_jsonl, write_jsonl

EXTRACTED = ROOT / "dataset" / "processed" / "documents_extracted.jsonl"
STARTER = ROOT / "dataset" / "processed" / "documents.jsonl"
OUT = ROOT / "dataset" / "processed" / "chunks.jsonl"
LEGACY_OUT = ROOT / "dataset" / "processed" / "chunks_rebuilt.jsonl"


def select_source(source: str) -> Path:
    if source == "starter":
        return STARTER
    if source == "extracted":
        return EXTRACTED
    return EXTRACTED if EXTRACTED.exists() and EXTRACTED.stat().st_size > 0 else STARTER


def main() -> int:
    parser = argparse.ArgumentParser(description="Build hierarchy-preserving JSONL chunks.")
    parser.add_argument("--source", choices=["auto", "starter", "extracted"], default="auto")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--legacy-out", type=Path, default=LEGACY_OUT)
    args = parser.parse_args()

    src = select_source(args.source)
    if not src.exists() or src.stat().st_size == 0:
        raise FileNotFoundError(f"Chunk source is missing or empty: {src}")
    chunks = build_chunks(read_jsonl(src))
    write_jsonl(args.out, chunks)
    write_jsonl(args.legacy_out, chunks)
    print(f"Wrote {len(chunks)} chunks from {src} to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
