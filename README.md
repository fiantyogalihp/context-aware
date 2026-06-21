# Action-Driven Evaluation Layer for Official Public Healthcare RAG Systems

Paket ini adalah starter project siap pakai untuk topik tesis **Action-Driven Evaluation Layer for Official Public Healthcare RAG Systems**.

Paket sudah berisi:

1. katalog sumber resmi pemerintah/BPJS (`dataset/source_catalog.csv`),
2. starter snapshot dokumen resmi yang sudah diverifikasi dari halaman resmi (`dataset/processed/documents.jsonl`),
3. hasil hierarchy-preserving chunks awal (`dataset/processed/chunks.jsonl`),
4. evaluation set Accept/Review/Reject (`dataset/eval/qa_eval.jsonl`),
5. contoh output RAG untuk pengujian evaluator (`dataset/eval/sample_model_outputs.jsonl`),
6. evaluator deterministik Python murni (`src/evaluator.py`),
7. tiny lexical retriever untuk demo (`src/simple_retriever.py`),
8. scripts pipeline lengkap dari download sampai verifikasi (`scripts/`),
9. laporan hasil verifikasi (`reports/`).

## Scope aman

Sistem ini **bukan** untuk diagnosis, resep, dosis obat, atau rekomendasi terapi individual. Scope yang disarankan:

- edukasi kesehatan publik non-diagnostik,
- navigasi layanan publik seperti Mobile JKN,
- lookup metadata kebijakan/formularium,
- eksperimen safety routing Accept/Review/Reject.

## Cara menjalankan cepat tanpa internet

```bash
python run_all.py
```

Ini menjalankan validasi evaluation set, evaluator, dan verifikasi paket menggunakan starter dataset yang sudah ada.

Untuk menjalankan evaluator langsung dari JSONL contoh:

```bash
python src/evaluator.py --input dataset/eval/sample_model_outputs.jsonl
```

## Cara mengunduh dokumen resmi dan membangun ulang dataset lengkap

Jika mesin Anda punya akses internet:

```bash
python run_all.py --download
```

Atau jalankan bertahap:

```bash
python scripts/01_download_sources.py
python scripts/02_extract_documents.py
python scripts/03_chunk_hierarchy.py
python scripts/04_build_eval_set.py
python scripts/05_evaluate_examples.py
python scripts/06_verify_package.py
```

Catatan: beberapa situs pemerintah memakai tautan PDF dinamis atau Google Drive. Bila download otomatis gagal, buka `dataset/source_catalog.csv`, unduh manual dari `landing_page`, lalu simpan ke `local_path` yang tercantum.

## Pipeline eksperimen penuh

Pipeline tesis production-grade memakai parser PDF tabel, cache embedding lokal, hybrid retrieval BM25+dense RRF, dan generator Xiaomi Mimo 2.5.

```bash
python scripts/pipeline.py --download --extract --build-index --run-eval --engine mimo
```

Untuk debugging jawaban dan konteks yang dipakai evaluator:

```bash
python scripts/run_tests.py --engine mimo \
  --eval-set dataset/eval/qa_eval.jsonl \
  --output reports/rag_eval_results.jsonl \
  --debug-output reports/rag_eval_debug.jsonl \
  --summary-output reports/rag_eval_summary.json \
  --skip-build-index
```

Konfigurasi Mimo dibaca dari environment:

```bash
export MIMO_API_KEY="..."
export MIMO_BASE_URL="https://api.xiaomimimo.com/v1"
export MIMO_MODEL="mimo-v2.5-pro"
```

Jika kamu memakai paket Token Plan, ganti `MIMO_BASE_URL` ke base URL eksklusif yang diberikan di console subscription. Jangan tambahkan suffix `/chat/completions`. Cache embedding lokal disimpan di `.cache/retriever/`. Jalur ini membutuhkan dependensi di `requirements.txt`, termasuk `pdfplumber`, `sentence-transformers`, `rank_bm25`, `numpy`, dan `scikit-learn`.

## Struktur dataset

```text
dataset/
  source_catalog.csv
  raw/                         # hasil download dokumen resmi
  raw_snapshots/                # snapshot teks resmi yang sudah dimasukkan sebagai starter
  processed/
    documents.jsonl            # starter dokumen resmi terstruktur
    chunks.jsonl               # starter chunks hierarchy-preserving
    documents_extracted.jsonl  # dibuat setelah download + ekstraksi
    chunks_rebuilt.jsonl       # dibuat setelah ekstraksi ulang
  eval/
    qa_eval.jsonl
    sample_model_outputs.jsonl
```

## Skema chunk

Setiap chunk minimal memuat:

```json
{
  "chunk_id": "BPJS_MJKN_ADMIN_PANDUAN_MENDAFTAR_SEBAGAI_PESERTA_JKN_007",
  "doc_id": "BPJS_MJKN_ADMIN",
  "title": "Administrasi Kepesertaan JKN",
  "section": "Panduan mendaftar sebagai peserta JKN",
  "source_type": "service_manual",
  "institution": "BPJS Kesehatan",
  "source_url": "https://bpjs-kesehatan.go.id/user-manual-mobile-jkn/administrasi%20JKN.html",
  "version": "current",
  "hierarchy_path": ["Administrasi Kepesertaan JKN", "Panduan mendaftar sebagai peserta JKN"],
  "text": "..."
}
```

## Evaluator

Evaluator menghitung tiga skor:

- `attribution_score`: apakah klaim jawaban didukung konteks resmi yang diretriev,
- `specificity_score`: apakah jawaban cukup spesifik tanpa melebihi konteks,
- `context_quality_score`: apakah konteks resmi, relevan, dan cukup.

Routing deterministik:

- `ACCEPT`: skor atribusi, spesifisitas, dan kualitas konteks melewati threshold, tanpa hard fail,
- `REVIEW`: konteks/atribusi sedang atau belum cukup untuk dilepas otomatis,
- `REJECT`: hard fail, konteks tidak resmi, saran medis individual, diagnosis, resep/dosis, atau atribusi rendah.

Threshold default berada di `src/evaluator.py` (`accept_a=0.78`, `accept_s=0.50`, `accept_c=0.70` pada starter package).
Nilai threshold juga tersedia di `config/evaluator_thresholds.yaml` agar eksperimen dapat dikalibrasi tanpa mengubah kode.

## Contoh penggunaan evaluator

```python
from src.evaluator import evaluate

result = evaluate(question, answer, retrieved_contexts)
print(result.route)
print(result.attribution_score, result.specificity_score, result.context_quality_score)
print(result.hard_fails)
```

## Sumber resmi dalam katalog

- Buku KIA 2024 — Ditjen Kesehatan Primer dan Komunitas Kemenkes.
- Mobile JKN user manual — BPJS Kesehatan.
- Formularium Nasional KMK HK.01.07/MENKES/1199/2025 — Ditjen Farmalkes Kemenkes.
- Pedoman Pelayanan Antenatal Terpadu — Repository Kemenkes.

## Posisi metodologi untuk tesis

Gunakan starter dataset ini untuk Bab Metodologi sebagai bukti desain awal. Untuk eksperimen final, jalankan `--download`, cek hasil ekstraksi PDF, lalu lakukan kurasi manual pada chunk penting seperti Buku KIA dan Fornas karena PDF tabel/regulasi dapat menghasilkan teks yang kurang rapi.

Kalimat metodologi yang bisa dipakai:

> Dataset penelitian dibangun dari dokumen resmi pemerintah Indonesia dan BPJS Kesehatan yang tersedia secara publik. Dokumen dicatat dalam katalog sumber, diekstraksi menjadi teks, kemudian diproses menggunakan hierarchy-preserving chunking berdasarkan struktur asli dokumen seperti judul, subjudul, halaman, dan bagian prosedural. Evaluasi jawaban dilakukan menggunakan action-driven evaluation layer deterministik dengan routing Accept, Review, dan Reject berdasarkan skor atribusi, spesifisitas, kualitas konteks, serta hard-fail rules untuk klaim medis individual yang tidak aman.
