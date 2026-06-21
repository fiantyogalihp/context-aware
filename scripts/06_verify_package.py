
#!/usr/bin/env python3
"""Verify required files and schemas for the packaged thesis starter project."""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = [
    "dataset/source_catalog.csv",
    "dataset/processed/documents.jsonl",
    "dataset/processed/chunks.jsonl",
    "dataset/eval/qa_eval.jsonl",
    "dataset/eval/sample_model_outputs.jsonl",
    "src/evaluator.py",
    "src/simple_retriever.py",
    "src/parser.py",
    "src/hybrid_retriever.py",
    "src/vector_retriever.py",
    "src/generator.py",
    "config/evaluator_thresholds.yaml",
    "scripts/01_download_sources.py",
    "scripts/02_extract_documents.py",
    "scripts/03_chunk_hierarchy.py",
    "scripts/05_evaluate_examples.py",
    "scripts/pipeline.py",
    "scripts/run_tests.py",
    "README.md",
]

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    missing = [p for p in REQUIRED_FILES if not (ROOT/p).exists()]
    if missing:
        raise SystemExit("Missing files: " + ", ".join(missing))
    catalog = list(csv.DictReader(open(ROOT/"dataset/source_catalog.csv", encoding="utf-8")))
    chunks = read_jsonl(ROOT/"dataset/processed/chunks.jsonl")
    qas = read_jsonl(ROOT/"dataset/eval/qa_eval.jsonl")
    assert len(catalog) >= 8, "source_catalog should contain at least 8 official-source entries"
    assert len(chunks) >= 15, "starter chunks should contain at least 15 chunks"
    assert len(qas) >= 20, "qa_eval should contain at least 20 eval cases"
    for chunk in chunks:
        assert chunk.get("source"), f"chunk {chunk.get('chunk_id')} missing source"
        assert isinstance(chunk.get("metadata"), dict), f"chunk {chunk.get('chunk_id')} missing metadata object"
        assert chunk.get("content"), f"chunk {chunk.get('chunk_id')} missing content"
    chunk_ids = {c["chunk_id"] for c in chunks}
    for q in qas:
        for cid in q.get("expected_context_ids", []):
            assert cid in chunk_ids, f"QA {q['question_id']} references missing chunk_id {cid}"
    print("Package verification OK")
    print(f"sources={len(catalog)}, chunks={len(chunks)}, eval_cases={len(qas)}")

if __name__ == "__main__":
    main()
