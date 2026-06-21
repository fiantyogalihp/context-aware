# Repository Guidelines

## Project Structure & Module Organization
This repository is a Python starter package for an action-driven evaluation layer over official public-health RAG content.

- `src/`: core logic, including `evaluator.py` and `simple_retriever.py`.
- `scripts/`: ordered pipeline steps from download to verification.
- `dataset/`: source catalog, raw downloads, extracted docs, processed chunks, and eval sets.
- `reports/`: generated evaluation outputs and package reports.
- `tests/`: pytest-style checks for the evaluator.
- `run_all.py`: one-command pipeline runner.

## Build, Test, and Development Commands
Use commands from the repo root.

- `python run_all.py` runs the offline validation path: eval-set validation, evaluator execution, and package verification.
- `python run_all.py --download` downloads official sources first, then rebuilds extracted docs and chunks.
- `python scripts/01_download_sources.py` through `python scripts/06_verify_package.py` run individual pipeline stages.
- `python -m pytest -q` runs unit tests if `pytest` is installed in your environment.

## Coding Style & Naming Conventions
The codebase follows plain Python 3 style with no formatter or linter configured in `requirements.txt`.

- Use 4-space indentation and `snake_case` for functions, variables, and module filenames.
- Keep scripts executable and self-contained when they are part of the pipeline.
- Prefer short, explicit helper functions over heavy abstraction.
- Match existing JSONL field names exactly; chunk IDs use uppercase, underscore-separated identifiers.

## Testing Guidelines
Tests are intentionally small and deterministic.

- Put new tests under `tests/` and name them `test_*.py`.
- Favor sample-based assertions against the checked-in JSONL fixtures in `dataset/processed/` and `dataset/eval/`.
- Keep evaluator tests offline and reproducible; avoid network calls in test code.

## Commit & Pull Request Guidelines
No commit history is exposed in this checkout, so use concise imperative commit subjects such as `Add chunk validation`.

- Keep commits focused on one logical change.
- In PRs, describe the change, note any dataset regeneration, and mention commands you ran.
- Include screenshots only for visual artifacts; otherwise include relevant sample output or file paths.

## Security & Configuration Tips
- Treat `dataset/raw/` as downloaded source material and avoid manual edits unless you are intentionally curating inputs.
- The evaluator hard-rejects unsafe medical advice; preserve that behavior when changing thresholds or heuristics.
