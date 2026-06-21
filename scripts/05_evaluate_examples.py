
#!/usr/bin/env python3
"""Run deterministic evaluator on included sample outputs."""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluator import evaluate, result_to_dict

CHUNKS = ROOT / "dataset" / "processed" / "chunks.jsonl"
SAMPLES = ROOT / "dataset" / "eval" / "sample_model_outputs.jsonl"
OUT = ROOT / "reports" / "sample_evaluation_results.jsonl"

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    chunks = {c["chunk_id"]: c for c in read_jsonl(CHUNKS)}
    samples = read_jsonl(SAMPLES)
    results=[]; pass_count=0
    for s in samples:
        ctxs = [chunks[cid] for cid in s.get("retrieved_context_ids", []) if cid in chunks]
        ev = evaluate(s["question"], s["answer"], ctxs)
        rd = result_to_dict(ev)
        rd.update({"case_id": s["case_id"], "expected_route": s["expected_route"], "pass": ev.route == s["expected_route"]})
        pass_count += int(rd["pass"])
        results.append(rd)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")
    print(f"Samples passed: {pass_count}/{len(samples)}")
    for r in results:
        print(f"{r['case_id']}: route={r['route']} expected={r['expected_route']} A={r['attribution_score']} S={r['specificity_score']} C={r['context_quality_score']} pass={r['pass']}")
    return 0 if pass_count == len(samples) else 1

if __name__ == "__main__":
    raise SystemExit(main())
