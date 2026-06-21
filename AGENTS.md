# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python starter package for action-driven evaluation over official public-health RAG content.

- `src/`: core logic, including `parser.py`, `hybrid_retriever.py`, `vector_retriever.py`, `generator.py`, and `evaluator.py`.
- `scripts/`: ordered pipeline stages from download through verification.
- `dataset/`: source catalog, raw downloads, extracted docs, processed chunks, and eval sets.
- `reports/`: generated evaluation outputs and package reports.
- `tests/`: deterministic pytest checks for parser, retriever, and evaluator behavior.
- `run_all.py`: one-command offline runner.

## Build, Test, and Development Commands

Run commands from the repo root. Prefer `venv/bin/python` when the shell does not activate the project virtualenv.

- `venv/bin/python run_all.py` runs the offline validation path.
- `venv/bin/python run_all.py --download` downloads official sources and rebuilds extracted docs and chunks.
- `venv/bin/python scripts/pipeline.py --download --extract --build-index --run-eval --engine mimo` runs the end-to-end thesis pipeline.
- `venv/bin/python src/evaluator.py --input dataset/eval/sample_model_outputs.jsonl` runs the evaluator on the sample fixture.
- `venv/bin/python -m pytest -q` runs unit tests if `pytest` is installed in the active environment.

## Coding Style & Naming Conventions

The codebase follows plain Python 3 style and keeps dependencies lightweight.

- Use 4-space indentation and `snake_case` for functions, variables, and filenames.
- Keep scripts executable and self-contained when they are part of the pipeline.
- Prefer short, explicit helper functions over heavy abstraction.
- Preserve existing JSONL schemas and uppercase, underscore-separated chunk IDs.

## Testing Guidelines

Tests are intentionally small and deterministic.

- Put new tests under `tests/` and name them `test_*.py`.
- Favor sample-based assertions against the checked-in JSONL fixtures in `dataset/processed/` and `dataset/eval/`.
- Keep evaluator and retrieval tests offline and reproducible; avoid network calls in test code.

## Commit & Pull Request Guidelines

No commit history is exposed in this checkout, so use concise imperative commit subjects such as `Add table-aware parser`.

- Keep commits focused on one logical change.
- In PRs, describe the change, note any dataset regeneration, and mention commands you ran.
- Include screenshots only for visual artifacts; otherwise include relevant sample output or file paths.

## Security & Configuration Tips

- Treat `dataset/raw/` as downloaded source material and avoid manual edits unless you are intentionally curating inputs.
- Keep `MIMO_API_KEY` and `MIMO_BASE_URL` in local environment configuration only.
- The evaluator hard-rejects unsafe medical advice; preserve that behavior when changing thresholds or heuristics.

