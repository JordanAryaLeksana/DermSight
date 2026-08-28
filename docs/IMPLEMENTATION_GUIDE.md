# Panduan Implementasi DermSight Web

## Cara aplikasi lama bekerja

`web/streamlit_app.py` adalah antarmuka Streamlit. Pengguna memilih/menangkap foto, Streamlit mengirim bytes gambar ke FastAPI, lalu menampilkan label, confidence, rekomendasi, laporan Markdown, dan form aspirasi 3T. FastAPI menggunakan singleton dependency untuk memuat model dan layanan rekomendasi satu kali per proses. Aspirasi lama ditulis ke `web/data/suara_masyarakat_3t.csv`.

## Fitur yang ditemukan

- Klasifikasi 22 kondisi kulit dengan EfficientNet B2 dan bobot `src/outputs/final_model.weights.h5`.
- Input RGB di-resize ke 224×224 lalu memakai `tensorflow.keras.applications.efficientnet.preprocess_input`.
- Output softmax berupa `predicted_label` dan `confidence`.
- Analisis edukatif berbasis RAG dari `llm/data/skin_knowledge_serving.csv` dan Ollama.
- Panduan pengambilan foto, interpretasi confidence, disclaimer medis, dan laporan hasil.
- Aspirasi masyarakat 3T (sebelumnya penyimpanan CSV lokal).

## Framework yang dipilih

Flask dipilih karena aplikasi hanya membutuhkan tiga area server-rendered, integrasi Python langsung ke service yang sudah ada, form, dan SMTP. Django tidak memberi manfaat yang sebanding karena tidak ada kebutuhan ORM, admin, autentikasi, atau sistem konten. Flask menjaga deployment tetap kecil dan mudah dipelihara.

## Struktur baru

```text
web/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── routes.py
│   ├── services/
│   │   ├── prediction_service.py
│   │   ├── llm_service.py
│   │   └── email_service.py
│   ├── templates/
│   │   ├── base.html
│   │   ├── detection.html
│   │   ├── analysis.html
│   │   ├── aspiration.html
│   │   └── errors/
│   └── static/
│       ├── css/main.css
│       └── js/main.js
├── tests/
├── .env.example
├── requirements.txt
└── wsgi.py
```

## File yang diubah

- Menambah aplikasi Flask, template, CSS, JavaScript, service adapter, dan test di `web/`.
- Memperbarui `web/requirements.txt`, `web/.env.example`, `.gitignore`, dan `README.md`.
- `web/streamlit_app.py` dipertahankan sebagai referensi migrasi, tetapi bukan lagi entry point produksi.
- Model, bobot, preprocessing, label mapping, RAG, dan prompt lama tidak diubah.

## Flow deteksi

Browser mengirim gambar multipart dengan CSRF → backend memeriksa keberadaan file, batas ukuran, nama/ekstensi, MIME terdeteksi dari isi, dan dekode aman dengan Pillow → service singleton menjalankan predictor lama → halaman menampilkan kemungkinan kondisi dan confidence. Gambar hanya berada di memori selama request dan tidak disimpan permanen.

## Flow LLM

Hasil prediction ditandatangani server dan dikirim sebagai token pendek → pengguna memilih **Lihat Analisis Lebih Lengkap** → Flask memverifikasi token → service lama mengambil konteks RAG dan meminta penjelasan Ollama → halaman analisis menampilkan jawaban edukatif. Kegagalan Ollama ditangani tanpa mengubah hasil prediksi dan tanpa traceback ke pengguna.

## Flow aspirasi

Form `/aspirasi` + CSRF → validasi jenis, panjang nama/kontak/pesan, dan honeypot → `EmailService` mengirim email melalui SMTP ke `ASPIRATION_EMAIL_TO` → success state. Tujuan dan kredensial tidak di-hardcode. Jika gagal, nilai form dirender ulang agar input pengguna tidak hilang. URL WhatsApp opsional hanya muncul bila dikonfigurasi.

## Design guideline

Mobile-first editorial neo-brutalist: warm cream, pastel blue, soft lime, lavender dan peach; border hampir hitam, kartu persegi, offset shadow ringan, tipografi sistem, target sentuh besar, fokus terlihat, dan satu kolom pada layar kecil. Tidak ada framework front-end, font eksternal, icon pack, atau asset berat. Tutorial tiga langkah memakai vanilla JavaScript dan `localStorage`; fungsi utama tetap berjalan tanpa JavaScript.

## Production consideration

Gunakan secret acak, debug off, Gunicorn di belakang Nginx, HTTPS, batas upload, CSRF, secure cookie, reverse-proxy header yang dipercaya satu hop, logging tanpa data sensitif, SMTP TLS, dan health endpoint. Model di-lazy-load sekali per worker; jumlah worker disesuaikan RAM karena setiap worker memiliki salinan TensorFlow/model. Upload tidak disimpan. LLM dan SMTP diberi timeout. Nginx perlu menyamakan `client_max_body_size` dengan batas Flask.
