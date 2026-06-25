# Action-Driven Evaluation Layer for Official Public Healthcare RAG Systems

Starter repo ini membangun RAG kesehatan publik Indonesia dengan tiga hal inti: ekstraksi dokumen resmi yang menjaga struktur tabel, hybrid retrieval BM25 + dense embedding, dan evaluator deterministik `ACCEPT / REVIEW / REJECT` dengan hybrid semantic-lexical scoring.

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
src/evaluator.py (Hybrid Semantic-Lexical Scoring)
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

### 2. Query Routing dan Retrieval

#### Query Router (`src/query_router.py`)

- Klasifikasi intent query secara deterministik menggunakan regex patterns.
- Dua intent utama:
  - **COMPUTATIONAL**: Query yang memerlukan agregasi, perbandingan, atau komputasi (contoh: "Berapa total vaksin?", "Bandingkan BPJS PBI dan Non-PBI")
  - **FACTUAL**: Query yang mencari informasi prosedural atau deskriptif (contoh: "Apa syarat pendaftaran?", "Bagaimana cara mengubah faskes?")
- Output routing mencakup:
  - Intent classification dengan confidence score
  - Processing path recommendation
  - Suggested chunk types untuk filtering (table-only untuk computational, mixed untuk factual)

#### Hybrid Retriever (`src/hybrid_retriever.py`)

- Menggabungkan:
  - BM25 (`rank_bm25`) untuk pencocokan kata kunci/istilah eksak,
  - embedding semantik (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`),
  - Reciprocal Rank Fusion untuk gabung peringkat.
- Mendukung filtering berdasarkan chunk type (`filter_chunk_types` parameter).
- Hasil retrieval dipangkas oleh `prune_context()` agar konteks tidak berisi duplikasi dan noise layout.

**Contoh Penggunaan Query Router:**

```python
from src.query_router import QueryRouter
from src.hybrid_retriever import HybridRetriever

router = QueryRouter()
retriever = HybridRetriever.from_jsonl("dataset/processed/chunks.jsonl")

# Route query
query = "Berapa total vaksin yang diperlukan?"
routing = router.route_query(query)

# Retrieve dengan filtering
if routing["processing_path"] == "computational":
    contexts = retriever.retrieve(
        query, 
        top_k=4,
        filter_chunk_types=["table"]  # Prioritas tabel untuk komputasi
    )
else:
    contexts = retriever.retrieve(query, top_k=4)
```

### 3. Generation

- `src/generator.py` adalah wrapper Xiaomi Mimo 2.5.
- Client mengirim chat completions OpenAI-compatible dengan `temperature=0.0`.
- `MIMO_BASE_URL` bisa diisi manual; jika kosong, kode mencoba menebak base URL dari prefiks API key.

### 4. Evaluation dengan Hybrid Semantic-Lexical Scoring

- `src/evaluator.py` menghitung:
  - **`attribution_score`** (Hybrid): Kombinasi semantic similarity (60%) dan lexical token overlap (40%)
    - `semantic_score`: Cosine similarity menggunakan SBERT (`paraphrase-multilingual-MiniLM-L12-v2`)
    - `lexical_score`: Token overlap dengan validasi numerik (logika original)
    - Formula: `attribution_score = 0.6 × semantic_score + 0.4 × lexical_score`
  - `specificity_score`: Mengukur detail dan relevansi jawaban
  - `context_quality_score`: Mengukur kualitas konteks yang diambil
- Routing deterministik memakai threshold default `accept_a=0.78`, `accept_s=0.50`, `accept_c=0.70`.
- Ada hard-fail rules untuk diagnosis, dosis obat, resep, dan klaim medis individual.

**Output Evaluasi:**

```json
{
  "route": "ACCEPT",
  "attribution_score": 0.856,
  "semantic_score": 0.912,
  "lexical_score": 0.765,
  "specificity_score": 0.680,
  "context_quality_score": 0.850,
  "supported_claims": 2,
  "total_claims": 2,
  "hard_fails": [],
  "warnings": []
}
```

## Struktur Repo

- `src/`: parser, retriever, generator, evaluator, query router.
  - `parser.py`: Ekstraksi PDF/HTML dengan table-aware parsing
  - `hybrid_retriever.py`: BM25 + dense embeddings + RRF dengan chunk type filtering
  - `query_router.py`: Intent-based query classification (computational vs factual)
  - `generator.py`: Xiaomi Mimo 2.5 wrapper dengan breadcrumb injection dan JSON structured output
  - `evaluator.py`: Hybrid semantic-lexical scoring untuk attribution + deterministic routing
- `scripts/`: pipeline download, ekstraksi, chunking, evaluasi, verifikasi.
- `dataset/`: katalog sumber, raw files, dokumen terproses, eval set.
- `reports/`: hasil evaluasi dan ringkasan paket.
- `tests/`: test offline deterministik.
  - `test_query_router.py`: 22 tests untuk query routing
  - `test_breadcrumb_injection.py`: 5 tests untuk breadcrumb injection
  - `test_json_structured_output.py`: 8 tests untuk JSON structured output
  - `test_hybrid_scoring.py`: 14 tests untuk hybrid semantic-lexical scoring
  - `test_evaluator.py`: 16 tests untuk evaluator (backward compatibility)
- `examples/`: contoh penggunaan query router dan integrasi komponen
- `docs/`: dokumentasi lengkap
  - `QUERY_ROUTER.md`: Dokumentasi query router
  - `HYBRID_SCORING.md`: Dokumentasi hybrid semantic-lexical scoring
  - `IMPLEMENTATION_SUMMARY.md`: Ringkasan implementasi fitur advanced
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
- `venv/bin/python src/evaluator.py --input dataset/eval/sample_model_outputs.jsonl` menjalankan evaluator offline dengan hybrid scoring.
- `venv/bin/python scripts/run_tests.py --engine mimo ...` menjalankan retrieval + generation + evaluasi pada eval set.

## Prinsip Kerja

- Dokumen resmi harus menjadi sumber utama.
- Evaluator tetap deterministik dan tidak bergantung pada LLM untuk routing decisions.
- Hybrid semantic-lexical scoring menggabungkan pemahaman semantik dengan exact token matching.
- Jawaban medis individual, dosis, dan diagnosis tetap harus ditolak oleh hard-fail rules.
- Saat mengubah parser atau retriever, pertahankan schema JSONL agar evaluasi dan pipeline tidak pecah.

## Fitur Lanjutan

### 1. Hierarchy-Preserving Table Parser

Sudah terimplementasi di `src/parser.py`:
- Ekstraksi tabel dengan `pdfplumber` yang menjaga struktur kolom
- Konversi ke format Markdown untuk readability
- Metadata lengkap (page, section, chunk_type)
- Chunk ID hierarkis dengan format `DOC_SECTION_SUBSECTION_###`

### 2. Intent-Based Query Router

Terimplementasi di `src/query_router.py`:
- Klasifikasi deterministik menggunakan regex patterns
- Dua intent: COMPUTATIONAL (agregasi/komputasi) dan FACTUAL (prosedural/deskriptif)
- Confidence scoring untuk setiap pattern match
- Integrasi dengan `HybridRetriever` untuk chunk type filtering
- 22 unit tests dengan coverage lengkap

**Query Classification Examples:**

```python
# Computational queries → filter ke table chunks
"Berapa total vaksin yang diperlukan?"
"Bandingkan BPJS PBI dan Non-PBI"
"Hitung rata-rata biaya persalinan"

# Factual queries → allow text + table chunks
"Apa syarat pendaftaran BPJS?"
"Bagaimana cara mengubah faskes?"
"Kapan jadwal imunisasi BCG?"
```

### 3. Breadcrumb Injection

Terimplementasi di `src/generator.py`:
- Injeksi hierarchy path ke setiap chunk context
- Format: `[Source > Section > Subsection]`
- Membantu LLM memahami struktur dokumen
- 5 unit tests untuk validasi

### 4. JSON Structured Output

Terimplementasi di `src/generator.py`:
- Memaksa LLM ekstrak exact quotes sebelum generate answer
- Format output terstruktur dengan `exact_quotes` dan `answer`
- Mengurangi hallucination dengan grounding eksplisit
- 8 unit tests untuk validasi

### 5. Hybrid Semantic-Lexical Scoring

Terimplementasi di `src/evaluator.py`:
- **Semantic Similarity (60%)**: Menggunakan SBERT (`paraphrase-multilingual-MiniLM-L12-v2`) untuk menghitung cosine similarity antara answer dan contexts
- **Lexical Token Overlap (40%)**: Mempertahankan logika original token matching dengan validasi numerik
- **Formula**: `attribution_score = 0.6 × semantic_score + 0.4 × lexical_score`
- **Transparent Scoring**: Kedua komponen (semantic dan lexical) ditampilkan dalam output JSON
- **Fallback Handling**: Jika model SBERT tidak tersedia, fallback ke lexical-only scoring
- 14 unit tests untuk hybrid scoring + 16 tests backward compatibility

**Keuntungan Hybrid Scoring:**
- Menangkap parafrase dan variasi bahasa (semantic)
- Mempertahankan exact factual grounding (lexical)
- Mengurangi false positives dan false negatives
- Transparent dan debuggable

**Contoh Output:**

```python
from src.evaluator import evaluate

result = evaluate(question, answer, contexts)
print(f"Attribution: {result.attribution_score}")
print(f"  Semantic: {result.semantic_score}")
print(f"  Lexical: {result.lexical_score}")
```

**Download Model SBERT:**

Model akan otomatis didownload saat pertama kali digunakan. Untuk download manual:

```bash
venv/bin/python -c "
import os
os.environ['HF_TOKEN'] = 'your_hf_token_here'
from sentence_transformers import SentenceTransformer
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
"
```

## Pengembangan

- Gunakan 4 spasi dan `snake_case`.
- Tambahkan test offline di `tests/test_*.py` untuk perubahan parser, retriever, atau evaluator.
- Hindari manual edit pada `dataset/raw/` kecuali sedang kurasi sumber.
- Simpan hasil eksperimen ke `reports/`, bukan ke file sumber.
- Jalankan `venv/bin/python -m pytest -q` untuk memastikan semua tests passing sebelum commit.

## Test Coverage

Total: **57 tests** (semua passing)
- Query Router: 22 tests
- Breadcrumb Injection: 5 tests
- JSON Structured Output: 8 tests
- Hybrid Scoring: 14 tests
- Evaluator (backward compatibility): 16 tests
- Retriever Filter: 3 tests
- Integration: 2 tests

## Dokumentasi Lengkap

- **Query Router**: `docs/QUERY_ROUTER.md`
- **Hybrid Scoring**: `docs/HYBRID_SCORING.md`
- **Implementation Summary**: `docs/IMPLEMENTATION_SUMMARY.md`
- **Main README**: `README.md` (file ini)