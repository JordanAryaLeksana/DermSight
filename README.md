# DermSight

DermSight adalah aplikasi web ringan untuk edukasi awal kondisi kulit dari foto. Antarmuka produksi menggunakan Flask, server-rendered HTML, CSS, dan vanilla JavaScript agar nyaman pada ponsel murah serta koneksi terbatas.

> DermSight bukan alat diagnosis medis. Hasil hanya menunjukkan kemungkinan berdasarkan kemiripan yang dipelajari model. Pemeriksaan langsung oleh tenaga kesehatan tetap diperlukan.

## Fitur

- Upload atau ambil foto langsung dari kamera ponsel.
- Klasifikasi 22 label dengan EfficientNet B2 dan confidence score.
- Analisis edukatif bahasa Indonesia melalui Ollama + RAG knowledge base.
- Tutorial tiga langkah saat kunjungan pertama dan tombol untuk membukanya kembali.
- Aspirasi masyarakat melalui SMTP dengan tujuan yang dapat diganti lewat environment.
- CSRF, pemeriksaan isi gambar, batas upload, security headers, custom error page, dan `/health`.
- Mobile-first tanpa React, font eksternal, icon pack, atau JavaScript bundle besar.

## Arsitektur

```text
Browser
  ↓ server-rendered form + CSRF
Flask (`web/wsgi.py`)
  ├─ PredictionService → predictor lama → EfficientNet B2
  ├─ LLMService → retriever lama → Ollama
  └─ EmailService → SMTP stakeholder
```

Pipeline model lama tetap digunakan:

```text
JPG/PNG tervalidasi
  ↓ RGB + resize 224×224
EfficientNet preprocess_input
  ↓
EfficientNet B2 + classifier head
  ↓
Softmax → label + confidence
```

Model di-lazy-load satu kali per Gunicorn worker. Foto diproses di memori dan tidak disimpan oleh aplikasi Flask. Detail audit dan keputusan implementasi ada di [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md).

## Struktur web

```text
web/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── routes.py
│   ├── services/
│   ├── static/
│   └── templates/
├── tests/
├── .env.example
├── requirements.txt
└── wsgi.py
```

`web/streamlit_app.py` hanya dipertahankan sebagai referensi aplikasi lama dan bukan entry point production.

## Setup lokal

Gunakan Python 3.11 atau versi yang kompatibel dengan TensorFlow yang dipilih.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r web/requirements.txt
cp web/.env.example web/.env
```

Isi minimal `web/.env`:

```env
SECRET_KEY=buat-secret-acak-yang-panjang
SKIN_MODEL_PATH=src/outputs/final_model.weights.h5
CLASS_NAMES_PATH=src/outputs/class_names.json
MODEL_CONFIG_PATH=src/outputs/config.json
DISEASE_LIST_PATH=llm/data/skin_knowledge_serving.csv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:e2b

ASPIRATION_EMAIL_TO=stakeholder@example.go.id
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USERNAME=
SMTP_PASSWORD=
SMTP_FROM=dermsight@example.com
SMTP_USE_TLS=true
```

Muat environment lalu jalankan development server tanpa debug:

```bash
set -a
source web/.env
set +a
python3 web/wsgi.py
```

Buka `http://127.0.0.1:5000`. Ollama hanya diperlukan saat pengguna membuka analisis lengkap; deteksi model tetap terpisah dari ketersediaan LLM.

## Menjalankan test

```bash
pytest -q web/tests
```

Test web memalsukan model, LLM, dan SMTP. Tidak ada email sungguhan atau request Ollama selama automated testing.

## Production dengan Gunicorn dan Nginx

Dari root repository:

```bash
gunicorn --workers 2 --threads 2 --timeout 180 --bind 127.0.0.1:8000 web.wsgi:app
```

Mulai dengan satu atau dua worker dan ukur RAM: setiap worker menyimpan salinan TensorFlow/model. Di Nginx, teruskan HTTPS ke Gunicorn dan set `client_max_body_size 8m`. Aktifkan konfigurasi berikut di production:

```env
SECRET_KEY=secret-random-production
SESSION_COOKIE_SECURE=true
TRUST_PROXY_HEADERS=true
LOG_LEVEL=INFO
```

Jangan expose Gunicorn langsung ke internet. Jalankan debug dalam keadaan off, gunakan HTTPS, simpan `.env` di luar version control, dan batasi akses ke kredensial SMTP.

Health check:

```text
GET /health
→ {"service":"dermsight-web","status":"ok"}
```

## Endpoint web

| Method | Path | Fungsi |
|---|---|---|
| GET | `/` | Form deteksi dan hasil |
| POST | `/deteksi` | Validasi gambar dan prediction |
| GET | `/analisis/<token>` | Analisis LLM dari hasil bertanda tangan |
| GET/POST | `/aspirasi` | Form dan pengiriman aspirasi via SMTP |
| GET | `/health` | Liveness check ringan |

FastAPI lama di `api/` tetap tersedia untuk consumer API yang sudah ada, tetapi website Flask tidak memerlukan proses FastAPI terpisah.

## Label model

Model mendukung 22 label: Acne, Actinic Keratosis, Benign Tumors, Bullous, Candidiasis, Drug Eruption, Eczema, Infestations/Bites, Lichen, Lupus, Moles, Psoriasis, Rosacea, Seborrheic Keratoses, Skin Cancer, Sun/Sunlight Damage, Tinea, Unknown/Normal, Vascular Tumors, Vasculitis, Vitiligo, dan Warts.

## Troubleshooting

- **Model tidak siap:** pastikan ketiga path model/config benar dan bobot tersedia.
- **Analisis lengkap gagal:** pastikan Ollama hidup, model sudah tersedia di Ollama, dan `OLLAMA_BASE_URL` dapat dijangkau.
- **Aspirasi gagal:** periksa host, port, TLS, sender, recipient, serta kredensial SMTP. Form tidak dikosongkan ketika pengiriman gagal.
- **Upload ditolak:** gunakan JPG/JPEG/PNG asli, maksimum sesuai `MAX_UPLOAD_MB`; mengganti ekstensi file saja tidak cukup.
- **502 dari Nginx saat inference:** samakan timeout proxy dengan timeout Gunicorn dan waktu load awal TensorFlow.

## Privasi dan keselamatan

Jangan unggah wajah penuh, bagian intim, atau gambar yang memuat identitas pribadi. Cari bantuan tenaga kesehatan bila keluhan memburuk, nyeri, menyebar, berdarah, bernanah, disertai demam, atau tidak membaik.
