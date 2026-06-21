# Step 1–8 completion map

1. **Kumpulkan sumber resmi** — selesai. Katalog sumber ada di `dataset/source_catalog.csv` dengan 9 entri resmi.
2. **Simpan raw/starter data** — selesai. Starter snapshot resmi ada di `dataset/raw_snapshots/` dan processed starter data ada di `dataset/processed/documents.jsonl`.
3. **Buat source catalog** — selesai. File: `dataset/source_catalog.csv`.
4. **Ekstraksi dokumen** — selesai untuk starter dataset. Script ekstraksi penuh tersedia di `scripts/02_extract_documents.py` setelah raw PDF/HTML diunduh.
5. **Hierarchy-preserving chunking** — selesai untuk starter dataset. File: `dataset/processed/chunks.jsonl`. Script rebuild: `scripts/03_chunk_hierarchy.py`.
6. **Evaluation set Accept/Review/Reject** — selesai. File: `dataset/eval/qa_eval.jsonl`, 20 kasus.
7. **Evaluator Python murni** — selesai. File: `src/evaluator.py` dengan skor attribution, specificity, context quality, hard fail, dan routing deterministik.
8. **Verifikasi dan packaging** — selesai. `python run_all.py` berhasil, 6/6 sample route benar, `pytest` 2 passed.

## Catatan penting

Lingkungan pembuatan file tidak dapat mengunduh raw PDF/HTML langsung dari domain pemerintah. Karena itu, paket ini menyertakan starter dataset dari snapshot teks resmi yang sudah diverifikasi, plus downloader dan pipeline untuk membangun dataset lengkap pada komputer yang memiliki akses internet.
