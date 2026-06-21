# Action-Driven Evaluation Layer for Official Public Healthcare RAG Systems

Paket ini adalah starter project untuk tesis RAG kesehatan publik Indonesia dengan evaluasi deterministik dan routing `ACCEPT / REVIEW / REJECT`.

## Struktur Proyek

- `src/`: logika inti.
  - `parser.py` untuk ekstraksi PDF, parsing tabel, dan chunking hierarkis.
  - `hybrid_retriever.py` untuk BM25, dense retrieval, dan RRF.
  - `generator.py` untuk client Xiaomi Mimo 2.5.
  - `evaluator.py` untuk scoring deterministik dan hard-fail safety rules.
- `scripts/`: pipeline berurutan dari download sampai verifikasi.
- `dataset/`: katalog sumber, raw files, dokumen terproses, dan eval set.
- `reports/`: output evaluasi, summary, dan laporan paket.
- `tests/`: pytest offline untuk komponen deterministik.

## Setup Cepat

Gunakan interpreter virtualenv proyek, bukan asumsi `python` sistem.

```bash
venv/bin/python -m pytest -q
venv/bin/python run_all.py
```

Jika memakai `direnv`, pastikan variabel Mimo sudah termuat. Bila `python` masih menunjuk ke interpreter sistem, jalankan `venv/bin/python ...` secara eksplisit.

## Command Utama

- `venv/bin/python run_all.py` menjalankan validasi offline, evaluator, dan verifikasi paket.
- `venv/bin/python run_all.py --download` mengunduh sumber resmi lalu membangun ulang ekstraksi dan chunk.
- `venv/bin/python scripts/pipeline.py --download --extract --build-index --run-eval --engine mimo` menjalankan pipeline tesis end-to-end.
- `venv/bin/python src/evaluator.py --input dataset/eval/sample_model_outputs.jsonl` mengeksekusi evaluator pada sample output lokal.
- `venv/bin/python scripts/run_tests.py --engine mimo --eval-set dataset/eval/qa_eval.jsonl --output reports/rag_eval_results.jsonl --debug-output reports/rag_eval_debug.jsonl --summary-output reports/rag_eval_summary.json --skip-build-index` menjalankan eksperimen Mimo dengan debug trace.

## Data & Output

- Sumber resmi dicatat di `dataset/source_catalog.csv`.
- Ekstraksi dokumen tersimpan di `dataset/processed/documents_extracted.jsonl`.
- Chunk terstruktur tersimpan di `dataset/processed/chunks.jsonl` dan `dataset/processed/chunks_rebuilt.jsonl`.
- Laporan evaluasi tersimpan di `reports/rag_eval_results.jsonl`, `reports/rag_eval_debug.jsonl`, dan `reports/rag_eval_summary.json`.

## Konfigurasi Mimo

Set variabel berikut di shell atau `.envrc` lokal:

```bash
export MIMO_API_KEY="..."
export MIMO_BASE_URL="https://api.xiaomimimo.com/v1"
export MIMO_MODEL="mimo-v2.5-pro"
```

`MIMO_BASE_URL` harus menunjuk ke base URL OpenAI-compatible, bukan endpoint `/chat/completions`.

## Aturan Kontribusi

- Gunakan 4 spasi dan `snake_case`.
- Pertahankan field JSONL yang sudah ada agar evaluator dan retriever tetap kompatibel.
- Jangan menghapus hard-fail medis individual; itu adalah guardrail inti proyek.
- Tambahkan test offline di `tests/test_*.py` jika mengubah parser, retriever, atau evaluator.

