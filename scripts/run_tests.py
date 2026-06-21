#!/usr/bin/env python3
"""Run validation-set experiments with starter or Xiaomi Mimo generation."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluator import evaluate, result_to_dict
from src.parser import read_jsonl, write_jsonl


def expected_contexts(row: Dict, chunks_by_id: Dict[str, Dict]) -> List[Dict]:
    return [chunks_by_id[cid] for cid in row.get("expected_context_ids", []) if cid in chunks_by_id]


def run_starter(rows: List[Dict], chunks_by_id: Dict[str, Dict]) -> List[Dict]:
    outputs = []
    for row in rows:
        contexts = expected_contexts(row, chunks_by_id)
        outputs.append({
            "question_id": row.get("question_id"),
            "question": row["question"],
            "answer": row.get("reference_answer", ""),
            "contexts": contexts,
            "retrieved_context_ids": [c["chunk_id"] for c in contexts],
        })
    return outputs


def run_mimo(rows: List[Dict], chunks_path: Path, top_k: int, *, build_index: bool) -> List[Dict]:
    from src.generator import XiaomiMimoClient
    from src.hybrid_retriever import HybridRetriever

    client = XiaomiMimoClient.from_env()
    retriever = HybridRetriever.from_jsonl(chunks_path)
    if build_index:
        retriever.build_index(recreate=False)
    outputs = []
    for row in rows:
        contexts = retriever.retrieve(row["question"], top_k=top_k)
        answer = client.generate_answer(row["question"], contexts)
        outputs.append({
            "question_id": row.get("question_id"),
            "question": row["question"],
            "answer": answer,
            "contexts": contexts,
            "retrieved_context_ids": [c["chunk_id"] for c in contexts],
        })
    return outputs


def context_debug_summary(contexts: List[Dict]) -> List[Dict]:
    summaries = []
    for ctx in contexts:
        text = str(ctx.get("text") or ctx.get("content") or "")
        summaries.append({
            "chunk_id": ctx.get("chunk_id"),
            "title": ctx.get("title"),
            "section": ctx.get("section"),
            "rrf_score": ctx.get("rrf_score"),
            "bm25_rank": ctx.get("bm25_rank"),
            "vector_rank": ctx.get("vector_rank"),
            "text_preview": text[:500],
        })
    return summaries


def quality_summary(results: List[Dict]) -> Dict:
    expected = [row for row in results if row.get("expected_route")]
    route_counts = Counter(row["route"] for row in results)
    expected_counts = Counter(row["expected_route"] for row in expected)
    pass_count = sum(1 for row in expected if row.get("pass"))
    metric_keys = ("attribution_score", "specificity_score", "context_quality_score")
    averages = {
        key: round(sum(float(row.get(key, 0.0)) for row in results) / max(1, len(results)), 3)
        for key in metric_keys
    }
    return {
        "total_cases": len(results),
        "expected_cases": len(expected),
        "passed_cases": pass_count,
        "pass_rate": round(pass_count / max(1, len(expected)), 3) if expected else None,
        "route_counts": dict(route_counts),
        "expected_route_counts": dict(expected_counts),
        "metric_averages": averages,
        "failed_question_ids": [
            row.get("question_id") for row in expected if not row.get("pass")
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG validation tests.")
    parser.add_argument("--engine", choices=["starter", "mimo"], default="starter")
    parser.add_argument("--eval-set", type=Path, default=ROOT / "dataset" / "eval" / "qa_eval.jsonl")
    parser.add_argument("--chunks", type=Path, default=ROOT / "dataset" / "processed" / "chunks.jsonl")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "rag_eval_results.jsonl")
    parser.add_argument("--debug-output", type=Path, help="write generated answers and retrieved context previews")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "reports" / "rag_eval_summary.json")
    parser.add_argument("--strict", action="store_true", help="return non-zero when expected routes do not all match")
    parser.add_argument("--skip-build-index", action="store_true", help="defer embedding-cache build until first retrieval")
    args = parser.parse_args()

    rows = read_jsonl(args.eval_set)
    chunks = read_jsonl(args.chunks)
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}

    if args.engine == "mimo":
        generated = run_mimo(rows, args.chunks, args.top_k, build_index=not args.skip_build_index)
    else:
        generated = run_starter(rows, chunks_by_id)

    results = []
    debug_rows = []
    pass_count = 0
    for row, output in zip(rows, generated):
        ev = evaluate(output["question"], output["answer"], output["contexts"])
        result = result_to_dict(ev)
        result.update({
            "question_id": row.get("question_id"),
            "expected_route": row.get("expected_route"),
            "retrieved_context_ids": output["retrieved_context_ids"],
        })
        if row.get("expected_route"):
            result["pass"] = ev.route == row["expected_route"]
            pass_count += int(result["pass"])
        results.append(result)
        debug_rows.append({
            **result,
            "question": output["question"],
            "answer": output["answer"],
            "contexts": context_debug_summary(output["contexts"]),
        })

    write_jsonl(args.output, results)
    summary = quality_summary(results)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.debug_output:
        write_jsonl(args.debug_output, debug_rows)
        print(f"Wrote debug traces to {args.debug_output}")
    print(f"Wrote {len(results)} evaluation results to {args.output}")
    print(f"Wrote quality summary to {args.summary_output}")
    if any("expected_route" in row for row in rows):
        print(f"Routes passed: {pass_count}/{len(rows)}")
        return 0 if not args.strict or pass_count == len(rows) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
