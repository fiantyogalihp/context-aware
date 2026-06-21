# Repository Guidelines

## Project Structure & Module Organization

This repository implements an offline-first public-health RAG starter with deterministic evaluation.

- `src/parser.py`: PDF/HTML extraction, table-aware parsing, chunk building, JSONL IO.
- `src/hybrid_retriever.py`: BM25 + dense embeddings + RRF + context pruning.
- `src/vector_retriever.py`: compatibility re-export for older imports.
- `src/generator.py`: Xiaomi Mimo 2.5 wrapper for OpenAI-compatible chat completions.
- `src/evaluator.py`: pure-Python attribution, specificity, context-quality scoring, and hard-fail rules.
- `scripts/`: download, extraction, chunking, testing, verification, and pipeline orchestration.
- `dataset/`: source catalog, raw inputs, extracted docs, processed chunks, eval sets.
- `reports/`: generated evaluation traces and summaries.
- `tests/`: offline pytest checks.

## Build, Test, and Development Commands

Run from repo root. Use `venv/bin/python` if the shell has not activated the project virtualenv.

- `venv/bin/python run_all.py` runs the starter offline path: eval build, evaluator, verification.
- `venv/bin/python run_all.py --download` downloads sources, then rebuilds extracted docs and chunks.
- `venv/bin/python scripts/pipeline.py --download --extract --build-index --run-eval --engine mimo` runs the full end-to-end pipeline.
- `venv/bin/python scripts/run_tests.py --engine mimo --eval-set dataset/eval/qa_eval.jsonl --output reports/rag_eval_results.jsonl --debug-output reports/rag_eval_debug.jsonl --summary-output reports/rag_eval_summary.json --skip-build-index` runs retrieval, generation, and evaluation with MiMo.
- `venv/bin/python src/evaluator.py --input dataset/eval/sample_model_outputs.jsonl` runs the deterministic evaluator on the sample fixture.
- `venv/bin/python -m pytest -q` runs unit tests in the active environment.

## Coding Style & Naming Conventions

- Use 4-space indentation and `snake_case`.
- Keep scripts small, executable, and self-contained.
- Prefer explicit helpers over heavy abstraction.
- Preserve existing JSONL schemas and uppercase underscore chunk IDs.
- Keep new text outputs ASCII unless source content requires otherwise.

## Testing Guidelines

- Add tests under `tests/` with names like `test_*.py`.
- Favor fixture-based assertions using `dataset/processed/` and `dataset/eval/`.
- Keep parser, retriever, and evaluator tests offline and reproducible.
- When changing chunking or retrieval, verify both `chunks.jsonl` shape and evaluator routing.

## Commit & Pull Request Guidelines

No commit history is exposed in this checkout, so use concise imperative subjects such as `Add table-aware parser`.

- Keep each commit focused on one logical change.
- In PRs, describe the user-visible effect, note dataset regeneration, and list the commands you ran.
- Include screenshots only for visual artifacts; otherwise attach sample output or file paths.

## Security & Configuration Tips

- Treat `dataset/raw/` as downloaded source material and avoid manual edits unless you are curating inputs.
- Keep `MIMO_API_KEY`, `MIMO_BASE_URL`, and `MIMO_MODEL` in local environment config only.
- `MIMO_BASE_URL` must point to the OpenAI-compatible base URL, not `/chat/completions`.
- Preserve evaluator hard-fail behavior for diagnosis, dosage, and other individual medical advice.
