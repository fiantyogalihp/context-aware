# Package report

Created: 2026-06-16

## Included

- Official-source catalog entries: 9
- Starter document records: 21
- Starter chunks: 21
- Evaluation cases: 20
- Sample model outputs: 6

## Important limitation

The file-building environment could not download raw files from government domains. Therefore, this ZIP includes verified official text snapshots, a complete source catalog, and a download/rebuild pipeline. Run `python run_all.py --download` on a machine with internet access to fetch full official PDFs/HTML and rebuild chunks.

## Safety scope

This package is scoped to public health education, public service navigation, and official document lookup. It rejects diagnosis, prescription, dosing, and individual clinical advice.

## Verification result

Local verification executed successfully:

```text
Evaluation set OK: 20 cases
Samples passed: 6/6
Package verification OK
sources=9, chunks=21, eval_cases=20
pytest: 2 passed
```
