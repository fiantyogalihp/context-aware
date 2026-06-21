
#!/usr/bin/env python3
"""Regenerate/validate evaluation-set structure. Starter qa_eval.jsonl is already included."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "dataset" / "eval" / "qa_eval.jsonl"
REQUIRED = {"question_id", "question", "expected_context_ids", "reference_answer", "expected_route", "risk_type"}

def main():
    rows=[]
    for line in open(QA, encoding="utf-8"):
        if line.strip(): rows.append(json.loads(line))
    bad=[]
    for r in rows:
        missing = REQUIRED - set(r)
        if missing: bad.append((r.get("question_id"), sorted(missing)))
        if r.get("expected_route") not in {"ACCEPT","REVIEW","REJECT"}:
            bad.append((r.get("question_id"), ["invalid expected_route"]))
    if bad:
        raise SystemExit(f"Invalid eval rows: {bad}")
    print(f"Evaluation set OK: {len(rows)} cases")

if __name__ == "__main__":
    main()
