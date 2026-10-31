# TA-04: Peramalan Pendapatan Harian E-Commerce (LSTM + Flask)

## Isi folder
- `ta04_lstm_pendapatan_harian.ipynb`: EDA, preprocessing, model LSTM, evaluasi, penyimpanan model
- `ecommerce_orders.csv`: dataset (pemisah `;`, desimal koma)
- `app.py`, `templates/index.html`: aplikasi Flask
- `artifacts/`: dibuat oleh notebook (`lstm_model.keras`, `scaler.joblib`, `meta.json`, `series.csv`, `test_predictions.csv`)

## Langkah
1. Jalankan notebook (Colab atau lokal) dari atas ke bawah. Di Colab, unggah `ecommerce_orders.csv` ke direktori kerja terlebih dulu.
2. Unduh folder `artifacts/` dan letakkan di sebelah `app.py`.
3. Samakan versi TensorFlow di `requirements.txt` dengan yang tercetak di sel terakhir notebook.
4. Jalankan lokal:
   ```
   pip install -r requirements.txt
   python app.py
   ```
   Buka http://localhost:5000
5. Deploy ke Render:
   - Unggah folder ke repositori GitHub (sertakan `artifacts/`).
   - New > Web Service, hubungkan repositori.
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --workers 1 --timeout 120`
   - Environment variable `PYTHON_VERSION` = versi Python yang sama dengan saat pelatihan (mis. `3.11.9`).
   - Railway memakai `Procfile` yang sudah tersedia.

## Catatan
- Paket gratis Render hanya punya RAM 512 MB. TensorFlow bisa melebihi batas itu dan layanan berhenti saat start. Jika terjadi, coba Railway atau naikkan paket.
- Endpoint: `/` (halaman), `/api/data`, `/api/forecast?days=7` (1 sampai 30), `/health`.
