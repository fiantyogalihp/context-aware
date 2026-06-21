#!/usr/bin/env python3
"""Master pipeline for reconstruction, indexing, and evaluation."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_script(script: str, *args: str) -> int:
    print(f"\n=== {script} {' '.join(args)} ===".strip())
    return subprocess.call([sys.executable, str(ROOT / script), *args])


def build_index(chunks_path: Path, recreate: bool) -> int:
    from src.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever.from_jsonl(chunks_path)
    retriever.build_index(recreate=recreate)
    print(f"Built local embedding cache for {len(retriever.chunks)} chunks from {chunks_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run thesis RAG pipeline stages.")
    parser.add_argument("--download", action="store_true", help="download official catalog sources")
    parser.add_argument("--extract", action="store_true", help="extract raw documents and rebuild chunks")
    parser.add_argument("--build-index", action="store_true", help="build local embedding cache for hybrid retrieval")
    parser.add_argument("--run-eval", action="store_true", help="run evaluator or Mimo E2E tests")
    parser.add_argument("--engine", choices=["starter", "mimo"], default="starter")
    parser.add_argument("--eval-set", type=Path, default=ROOT / "dataset" / "eval" / "qa_eval.jsonl")
    parser.add_argument("--chunks", type=Path, default=ROOT / "dataset" / "processed" / "chunks.jsonl")
    parser.add_argument("--recreate-index", action="store_true")
    parser.add_argument("--strict", action="store_true", help="fail when expected eval routes do not all pass")
    args = parser.parse_args()

    if args.download:
        code = run_script("scripts/01_download_sources.py")
        if code:
            return code
        args.extract = True

    if args.extract:
        for script in ("scripts/02_extract_documents.py", "scripts/03_chunk_hierarchy.py"):
            code = run_script(script)
            if code:
                return code

    if args.build_index:
        code = build_index(args.chunks, recreate=args.recreate_index)
        if code:
            return code

    if args.run_eval:
        if args.engine == "mimo":
            extra = ["--skip-build-index"] if args.build_index else []
            output = ROOT / "reports" / "rag_eval_results.jsonl"
            debug_output = ROOT / "reports" / "rag_eval_debug.jsonl"
            summary_output = ROOT / "reports" / "rag_eval_summary.json"
            strict = ["--strict"] if args.strict else []
            return run_script(
                "scripts/run_tests.py",
                "--engine",
                "mimo",
                "--eval-set",
                str(args.eval_set),
                "--chunks",
                str(args.chunks),
                "--output",
                str(output),
                "--debug-output",
                str(debug_output),
                "--summary-output",
                str(summary_output),
                *extra,
                *strict,
            )
        return run_script(
            "src/evaluator.py",
            "--input",
            str(ROOT / "dataset" / "eval" / "sample_model_outputs.jsonl"),
            "--chunks",
            str(args.chunks),
        )

    if not any((args.download, args.extract, args.build_index, args.run_eval)):
        return run_script(
            "src/evaluator.py",
            "--input",
            str(ROOT / "dataset" / "eval" / "sample_model_outputs.jsonl"),
            "--chunks",
            str(args.chunks),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
