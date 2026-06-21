# Action-Driven Evaluation Layer for Official Public Healthcare RAG Systems

Starter repo ini membangun RAG kesehatan publik Indonesia dengan tiga hal inti: ekstraksi dokumen resmi yang menjaga struktur tabel, hybrid retrieval BM25 + dense embedding, dan evaluator deterministik `ACCEPT / REVIEW / REJECT` tanpa model call.

## Alur Singkat

```text
Official sources (PDF/HTML)
        |
        v
scripts/01_download_sources.py
        |
        v
scripts/02_extract_documents.py -> dataset/processed/documents_extracted.jsonl
        |
        v
scripts/03_chunk_hierarchy.py -> dataset/processed/chunks.jsonl
        |
        v
src/hybrid_retriever.py + src/generator.py
        |
        v
src/evaluator.py
        |
        v
ACCEPT / REVIEW / REJECT + reports/*
```

## Arsitektur

### 1. Ingestion dan Parsing

- `scripts/01_download_sources.py` mengunduh sumber resmi dari `dataset/source_catalog.csv`.
- `scripts/02_extract_documents.py` mengekstrak PDF/HTML ke `dataset/processed/documents_extracted.jsonl`.
- `src/parser.py` memakai `pdfplumber` sebagai parser utama PDF, lalu memisahkan teks biasa dan tabel agar relasi kolom tetap utuh.
- `scripts/03_chunk_hierarchy.py` membangun chunk hierarkis ke `dataset/processed/chunks.jsonl` dan `dataset/processed/chunks_rebuilt.jsonl`.

### 2. Retrieval

- `src/hybrid_retriever.py` menggabungkan:
  - BM25 (`rank_bm25`) untuk pencocokan kata kunci/istilah eksak,
  - embedding semantik (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`),
  - Reciprocal Rank Fusion untuk gabung peringkat.
- Hasil retrieval dipangkas oleh `prune_context()` agar konteks tidak berisi duplikasi dan noise layout.

### 3. Generation

- `src/generator.py` adalah wrapper Xiaomi Mimo 2.5.
- Client mengirim chat completions OpenAI-compatible dengan `temperature=0.0`.
- `MIMO_BASE_URL` bisa diisi manual; jika kosong, kode mencoba menebak base URL dari prefiks API key.

### 4. Evaluation

- `src/evaluator.py` menghitung:
  - `attribution_score`
  - `specificity_score`
  - `context_quality_score`
- Routing deterministik memakai threshold default `accept_a=0.78`, `accept_s=0.50`, `accept_c=0.70`.
- Ada hard-fail rules untuk diagnosis, dosis obat, resep, dan klaim medis individual.

## Struktur Repo

- `src/`: parser, retriever, generator, evaluator.
- `scripts/`: pipeline download, ekstraksi, chunking, evaluasi, verifikasi.
- `dataset/`: katalog sumber, raw files, dokumen terproses, eval set.
- `reports/`: hasil evaluasi dan ringkasan paket.
- `tests/`: test offline deterministik.
- `run_all.py`: runner cepat untuk path starter/offline.

## Format Data

### `documents_extracted.jsonl`

Setiap record mewakili satu elemen hasil ekstraksi per halaman, termasuk teks atau tabel.

### `chunks.jsonl`

Chunk final menyimpan schema gabungan untuk kompatibilitas lama dan pipeline baru:

```json
{
  "chunk_id": "BPJS_MJKN_ADMIN_MENGUBAH_FASILITAS_KESEHATAN_TINGKAT_PERTAMA_011",
  "doc_id": "BPJS_MJKN_ADMIN",
  "title": "Administrasi Kepesertaan JKN",
  "section": "Mengubah fasilitas kesehatan tingkat pertama",
  "source": "BPJS Kesehatan",
  "metadata": {
    "page": 12,
    "section": "Mengubah fasilitas kesehatan tingkat pertama",
    "subsection": "Langkah penggunaan",
    "chunk_type": "text"
  },
  "content": "..."
}
```

Chunk tabel memakai `metadata.chunk_type = "table"` dan menyimpan isi tabel yang sudah dirapikan menjadi Markdown plus baris key-value.

## Command Utama

Gunakan interpreter virtualenv proyek jika `python` sistem tidak aktif.

```bash
venv/bin/python -m pytest -q
venv/bin/python run_all.py
venv/bin/python run_all.py --download
venv/bin/python scripts/pipeline.py --download --extract --build-index --run-eval --engine mimo
venv/bin/python src/evaluator.py --input dataset/eval/sample_model_outputs.jsonl
venv/bin/python scripts/run_tests.py --engine mimo --eval-set dataset/eval/qa_eval.jsonl --output reports/rag_eval_results.jsonl --debug-output reports/rag_eval_debug.jsonl --summary-output reports/rag_eval_summary.json --skip-build-index
```

Jika memakai `direnv`, variabel Mimo dari `.envrc` tetap perlu terbaca oleh shell. Bila `python` masih menunjuk ke interpreter sistem, pakai `venv/bin/python` secara eksplisit.

## Konfigurasi MiMo

Set variabel ini di shell atau `.envrc`:

```bash
export MIMO_API_KEY="..."
export MIMO_BASE_URL="https://api.xiaomimimo.com/v1"
export MIMO_MODEL="mimo-v2.5-pro"
```

- `MIMO_API_KEY` wajib.
- `MIMO_BASE_URL` opsional jika ingin memakai default yang diinfer dari prefiks key.
- `MIMO_MODEL` default ke `mimo-v2.5-pro`.
- Client mengirim header `api-key` dan endpoint `/chat/completions`.

## Verifikasi

- `venv/bin/python scripts/06_verify_package.py` memeriksa kelengkapan paket dan schema JSONL.
- `venv/bin/python src/evaluator.py --input dataset/eval/sample_model_outputs.jsonl` menjalankan evaluator offline.
- `venv/bin/python scripts/run_tests.py --engine mimo ...` menjalankan retrieval + generation + evaluasi pada eval set.

## Prinsip Kerja

- Dokumen resmi harus menjadi sumber utama.
- Evaluator tetap deterministik dan tidak bergantung pada LLM.
- Jawaban medis individual, dosis, dan diagnosis tetap harus ditolak oleh hard-fail rules.
- Saat mengubah parser atau retriever, pertahankan schema JSONL agar evaluasi dan pipeline tidak pecah.

## Pengembangan

- Gunakan 4 spasi dan `snake_case`.
- Tambahkan test offline di `tests/test_*.py` untuk perubahan parser, retriever, atau evaluator.
- Hindari manual edit pada `dataset/raw/` kecuali sedang kurasi sumber.
- Simpan hasil eksperimen ke `reports/`, bukan ke file sumber.
