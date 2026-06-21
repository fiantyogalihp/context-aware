
#!/usr/bin/env python3
"""One-command runner.

Default mode validates the included starter dataset and evaluator without internet.
Use --download to fetch official raw sources first, then rebuild extracted docs/chunks.
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def run(script):
    print("\n===", script, "===")
    return subprocess.call([sys.executable, str(ROOT / script)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="download official raw documents before extraction")
    args = ap.parse_args()
    scripts = []
    if args.download:
        scripts += ["scripts/01_download_sources.py", "scripts/02_extract_documents.py", "scripts/03_chunk_hierarchy.py"]
    scripts += ["scripts/04_build_eval_set.py", "scripts/05_evaluate_examples.py", "scripts/06_verify_package.py"]
    code = 0
    for s in scripts:
        rc = run(s)
        if rc != 0:
            code = rc
            break
    return code

if __name__ == "__main__":
    raise SystemExit(main())
