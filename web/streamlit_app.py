

import csv
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Path & Logger setup
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

try:
    from utils.logger import setup_logger, get_logger
    setup_logger(level=logging.INFO, log_dir="logs", log_file="streamlit.log")
    logger = get_logger(__name__)
except Exception:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

import requests
import streamlit as st
from PIL import Image

# Konstanta
DEFAULT_API_BASE_URL    = os.getenv("DERMSIGHT_API_BASE_URL",       "http://localhost:8000")
DEFAULT_ANALYZE_ENDPOINT  = os.getenv("DERMSIGHT_ANALYZE_ENDPOINT",  "/skin/analyze")
DEFAULT_PREDICT_ENDPOINT  = os.getenv("DERMSIGHT_PREDICT_ENDPOINT",  "/skin/predict")
DEFAULT_RECOMMEND_ENDPOINT = os.getenv("DERMSIGHT_RECOMMEND_ENDPOINT", "/recommendation/generate")

ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
FORM_CSV_PATH = Path("data/suara_masyarakat_3t.csv")

DISCLAIMER = (
    "DermSight **bukan pengganti diagnosis dokter**. Hasil ini hanya untuk edukasi awal. "
    "Jika keluhan memburuk, nyeri, menyebar, berdarah, bernanah, disertai demam, "
    "atau tidak membaik, **segera konsultasikan ke tenaga kesehatan terdekat**."
)

# Konfigurasi halaman (harus dipanggil pertama)
st.set_page_config(
    page_title="DermSight",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# CSS minimal — hanya untuk max-width container dan spacing kecil
st.markdown(
    """
    <style>
    .main .block-container { max-width: 720px; padding-top: 1.2rem; padding-bottom: 3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# HELPER: State & URL

def init_session_state() -> None:
    defaults: dict[str, Any] = {
        "uploaded_file_bytes": None,
        "uploaded_file_name": None,
        "uploaded_file_type": None,
        "analysis_result": None,
        "prediction_result": None,
        "recommendation_result": None,
        "last_error": None,
        "analysis_done": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_analysis() -> None:
    """Hapus semua state analisis dan upload."""
    keys = [
        "uploaded_file_bytes", "uploaded_file_name", "uploaded_file_type",
        "analysis_result", "prediction_result", "recommendation_result",
        "last_error", "analysis_done",
    ]
    for k in keys:
        st.session_state[k] = None
    st.session_state["analysis_done"] = False


def build_url(base_url: str, endpoint: str) -> str:
    return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"


# HELPER: Validasi response

def validate_prediction_response(data: dict) -> None:
    label = data.get("predicted_label")
    conf  = data.get("confidence")
    if not isinstance(label, str) or not label:
        raise ValueError("Response tidak memiliki 'predicted_label' yang valid.")
    if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
        raise ValueError("Response tidak memiliki 'confidence' yang valid (harus angka 0–1).")


def validate_analyze_response(data: dict) -> None:
    validate_prediction_response(data)
    if not isinstance(data.get("recommendation"), str):
        raise ValueError("Response tidak memiliki 'recommendation' yang valid.")


def validate_recommendation_response(data: dict) -> None:
    if not isinstance(data.get("recommendation"), str):
        raise ValueError("Response tidak memiliki 'recommendation' yang valid.")


# HELPER: API calls

def _post_image(url: str, file_name: str, file_bytes: bytes,
                content_type: str, timeout: int) -> dict:
    """POST gambar ke endpoint FastAPI."""
    resp = requests.post(
        url,
        files={"image": (file_name, file_bytes, content_type)},
        timeout=timeout,
    )
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise RuntimeError(f"API error {resp.status_code}: {detail}")
    return resp.json()


def _post_json(url: str, payload: dict, timeout: int) -> dict:
    """POST JSON ke endpoint FastAPI."""
    resp = requests.post(url, json=payload, timeout=timeout)
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise RuntimeError(f"API error {resp.status_code}: {detail}")
    return resp.json()


def call_analyze_api(base_url: str, endpoint: str,
                     file_name: str, file_bytes: bytes,
                     content_type: str, timeout: int = 120) -> dict:
    url = build_url(base_url, endpoint)
    logger.info("Calling analyze API | url=%s", url)
    data = _post_image(url, file_name, file_bytes, content_type, timeout)
    validate_analyze_response(data)
    return data


def call_predict_api(base_url: str, endpoint: str,
                     file_name: str, file_bytes: bytes,
                     content_type: str, timeout: int = 120) -> dict:
    url = build_url(base_url, endpoint)
    logger.info("Calling predict API | url=%s", url)
    data = _post_image(url, file_name, file_bytes, content_type, timeout)
    validate_prediction_response(data)
    return data


def call_recommendation_api(base_url: str, endpoint: str,
                             predicted_label: str, confidence: float,
                             timeout: int = 120) -> dict:
    url = build_url(base_url, endpoint)
    logger.info("Calling recommendation API | url=%s | label=%s", url, predicted_label)
    data = _post_json(url, {"predicted_label": predicted_label, "confidence": confidence}, timeout)
    validate_recommendation_response(data)
    return data


# HELPER: Confidence & Laporan

def get_confidence_interpretation(confidence: float) -> tuple[str, str]:
    """Kembalikan (emoji_label, teks_interpretasi)."""
    pct = confidence * 100
    if confidence < 0.50:
        return "🔴", f"{pct:.1f}% — Keyakinan rendah, gunakan hasil ini dengan sangat hati-hati."
    elif confidence < 0.75:
        return "🟡", f"{pct:.1f}% — Keyakinan sedang, sebaiknya diverifikasi ke tenaga kesehatan."
    else:
        return "🟢", f"{pct:.1f}% — Keyakinan cukup tinggi, tetap bukan diagnosis resmi."


def generate_markdown_report(result: dict[str, Any],
                              img_name: str | None = None) -> str:
    label      = result.get("predicted_label", "-")
    conf       = float(result.get("confidence") or 0.0)
    rekom      = result.get("recommendation", "-")
    retrieval  = result.get("retrieval_result") or {}
    now        = datetime.now().strftime("%d %B %Y, %H:%M")

    retrieval_section = ""
    if retrieval:
        retrieval_section = f"""
## Informasi Konteks Penyakit

| Keterangan | Nilai |
|---|---|
| Status pencocokan | {"✅ Ditemukan" if retrieval.get("matched") else "❌ Tidak ditemukan"} |
| Tipe pencocokan   | {retrieval.get("match_type", "-")} |
| Penyakit          | {retrieval.get("disease", "-")} |
| Label yang cocok  | {retrieval.get("matched_label", "-")} |

"""

    return f"""# Laporan DermSight
**Tanggal Analisis:** {now}
{"**File Foto:** " + img_name if img_name else ""}

---

## Perkiraan Kondisi Kulit

| Keterangan | Nilai |
|---|---|
| Kondisi yang terdeteksi | **{label}** |
| Tingkat keyakinan sistem | **{conf:.1%}** |

{retrieval_section}
---

## Rekomendasi Edukatif

{rekom}

---

## ⚠️ Disclaimer Medis

DermSight bukan pengganti diagnosis dokter. Hasil ini hanya untuk edukasi awal.
Jika keluhan memburuk, nyeri, menyebar, berdarah, bernanah, disertai demam, atau tidak membaik,
**segera konsultasikan ke tenaga kesehatan terdekat**.

---
*Laporan dibuat otomatis oleh DermSight — sistem AI edukatif.*
"""


# HELPER: Form 3T

def save_form_3t(data: dict) -> bool:
    FORM_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp", "nama", "daerah", "kendala",
                  "edukasi_dibutuhkan", "pesan_tambahan"]
    file_exists = FORM_CSV_PATH.exists()
    try:
        with open(FORM_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
        return True
    except Exception as e:
        logger.error("Gagal menyimpan form 3T: %s", e)
        return False


# UI COMPONENTS

def render_header() -> None:
    st.title("🩺 DermSight")
    st.caption("Asisten AI untuk edukasi awal penyakit kulit · Gratis, tanpa login, mudah digunakan.")
    st.warning(
        "⚠️ DermSight **bukan alat diagnosis medis**. "
        "Hasil hanya untuk edukasi awal, bukan pengganti pemeriksaan dokter.",
        icon=None,
    )


def render_photo_guide() -> None:
    with st.expander("📸 Panduan Foto yang Baik — baca sebelum mengunggah", expanded=False):
        st.markdown("""
- ☀️ **Pencahayaan cukup** — hindari ruangan gelap atau backlight.
- 🔍 **Gambar tidak buram** — tahan kamera dengan stabil saat memotret.
- 📏 **Jarak sedang** (±15–30 cm) agar detail kulit terlihat jelas.
- 🎯 **Fokus pada area bermasalah** — bukan foto seluruh wajah.
- 🚫 **Jangan sertakan** wajah penuh, bagian intim, atau identitas pribadi.
- ✅ Format yang didukung: **JPG, JPEG, PNG**.

> Foto yang jelas = hasil analisis yang lebih akurat.
        """)


def render_image_input() -> tuple[str | None, bytes | None, str | None]:
    """
    Tampilkan pilihan sumber gambar. Kembalikan (file_name, file_bytes, content_type).
    None jika belum ada gambar.
    """
    st.subheader("📷 Pilih Foto Kulit")

    source = st.radio(
        "Sumber gambar",
        ["Upload Foto dari Perangkat", "Ambil Foto dari Kamera"],
        horizontal=True,
        label_visibility="collapsed",
    )

    file_name: str | None = None
    file_bytes: bytes | None = None
    content_type: str | None = None

    if source == "Upload Foto dari Perangkat":
        uploaded = st.file_uploader(
            "Unggah foto kulit",
            type=ALLOWED_IMAGE_TYPES,
            help="Format: JPG, JPEG, PNG",
            label_visibility="visible",
        )
        if uploaded is not None:
            file_bytes   = uploaded.getvalue()
            file_name    = uploaded.name
            content_type = uploaded.type

    else:  # kamera
        camera_img = st.camera_input("Ambil foto kulit")
        if camera_img is not None:
            file_bytes   = camera_img.getvalue()
            file_name    = f"kamera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            content_type = "image/jpeg"

    # Simpan ke session_state
    if file_bytes is not None:
        st.session_state.uploaded_file_bytes = file_bytes
        st.session_state.uploaded_file_name  = file_name
        st.session_state.uploaded_file_type  = content_type
        logger.info("Gambar dipilih | nama=%s | ukuran=%d bytes", file_name, len(file_bytes))

    # Gunakan dari session_state jika ada (tidak hilang setelah rerun)
    if file_bytes is None and st.session_state.uploaded_file_bytes is not None:
        file_bytes   = st.session_state.uploaded_file_bytes
        file_name    = st.session_state.uploaded_file_name
        content_type = st.session_state.uploaded_file_type

    # Preview
    if file_bytes is not None:
        col_img, col_info = st.columns([1, 1])
        with col_img:
            st.image(file_bytes, caption="Pratinjau foto", use_container_width=True)
        with col_info:
            st.success("✅ Foto berhasil dipilih!")
            try:
                from io import BytesIO
                img = Image.open(BytesIO(file_bytes))
                st.caption(f"📄 {file_name}")
                st.caption(f"📐 {img.width} × {img.height} px")
                st.caption(f"💾 {len(file_bytes) / 1024:.1f} KB")
            except Exception:
                st.caption(f"📄 {file_name}")

    return file_name, file_bytes, content_type


def render_analysis_button(
    api_base_url: str,
    analyze_endpoint: str,
    predict_endpoint: str,
    recommend_endpoint: str,
    mode: str,
    timeout: int,
    file_name: str | None,
    file_bytes: bytes | None,
    content_type: str | None,
) -> None:
    if file_bytes is None:
        st.info("📤 Silakan pilih foto kulit terlebih dahulu.")
        return

    col_btn, col_reset = st.columns([3, 1])
    with col_btn:
        analyze_clicked = st.button(
            "🔍 Analisis Sekarang",
            type="primary",
            use_container_width=True,
        )
    with col_reset:
        if st.button("🔄 Reset", use_container_width=True, help="Hapus foto dan hasil sebelumnya"):
            reset_analysis()
            st.rerun()

    if not analyze_clicked:
        return

    logger.info("Analisis dimulai | mode=%s | file=%s", mode, file_name)

    try:
        if mode == "analyze":
            with st.spinner("🔬 Menganalisis gambar dan menyiapkan rekomendasi edukatif…"):
                result = call_analyze_api(
                    api_base_url, analyze_endpoint,
                    file_name, file_bytes, content_type, timeout,
                )
            st.session_state.analysis_result = result
            st.session_state.prediction_result = {
                "predicted_label": result.get("predicted_label"),
                "confidence": result.get("confidence"),
            }
            st.session_state.recommendation_result = {
                "recommendation": result.get("recommendation"),
            }

        else:  # step-by-step
            with st.spinner("🔬 Langkah 1/2 — Menganalisis foto kulit…"):
                prediction = call_predict_api(
                    api_base_url, predict_endpoint,
                    file_name, file_bytes, content_type, timeout,
                )
            st.session_state.prediction_result = prediction

            with st.spinner("💡 Langkah 2/2 — Menyiapkan rekomendasi edukatif…"):
                recommendation = call_recommendation_api(
                    api_base_url, recommend_endpoint,
                    prediction["predicted_label"], prediction["confidence"], timeout,
                )
            st.session_state.recommendation_result = recommendation
            st.session_state.analysis_result = {
                "predicted_label": prediction["predicted_label"],
                "confidence": prediction["confidence"],
                "recommendation": recommendation.get("recommendation"),
                "retrieval_result": recommendation.get("retrieval_result"),
            }

        st.session_state.analysis_done = True
        logger.info("Analisis selesai | label=%s", st.session_state.analysis_result.get("predicted_label"))
        st.success("✅ Analisis selesai! Gulir ke bawah untuk melihat hasil.")
        st.rerun()

    except requests.exceptions.ConnectionError:
        logger.exception("Connection error")
        st.error("❌ Tidak bisa terhubung ke server analisis. Pastikan FastAPI sudah berjalan di " + api_base_url)
    except requests.exceptions.Timeout:
        logger.exception("Timeout")
        st.error(f"⏱️ Analisis melebihi batas waktu ({timeout} detik). Coba lagi sebentar.")
    except ValueError as e:
        logger.exception("Validation error: %s", e)
        st.error(f"❌ Response dari server tidak sesuai format: {e}")
    except RuntimeError as e:
        logger.exception("Runtime error: %s", e)
        st.error(f"❌ Terjadi kesalahan dari server: {e}")
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        st.error(f"❌ Terjadi kesalahan tak terduga: {e}")


def render_result(result: dict[str, Any]) -> None:
    label      = result.get("predicted_label", "-")
    confidence = float(result.get("confidence") or 0.0)
    retrieval  = result.get("retrieval_result") or {}

    st.divider()
    st.subheader("🔎 Hasil Analisis")

    # Foto + kondisi dalam dua kolom
    col_img, col_pred = st.columns([1, 1])

    with col_img:
        if st.session_state.uploaded_file_bytes:
            st.image(
                st.session_state.uploaded_file_bytes,
                caption=st.session_state.uploaded_file_name or "Foto yang dianalisis",
                use_container_width=True,
            )

    with col_pred:
        st.metric(
            label="Perkiraan Kondisi Kulit",
            value=label,
            help="Sistem memperkirakan kondisi ini mirip dengan label berikut.",
        )
        emoji, interp_text = get_confidence_interpretation(confidence)
        st.metric(
            label="Tingkat Keyakinan Sistem",
            value=f"{confidence:.0%}",
            help="Seberapa yakin model AI dengan prediksi ini.",
        )
        st.progress(min(max(confidence, 0.0), 1.0))
        st.caption(f"{emoji} {interp_text}")

    # Catatan framing — jangan bilang "Anda terkena X"
    st.info(
        f"Sistem memperkirakan kondisi ini **mirip dengan: {label}**. "
        "Ini bukan diagnosis resmi.",
        icon="ℹ️",
    )

    # Konteks retrieval (ringkasan kecil)
    if retrieval:
        with st.expander("🗂️ Detail konteks penyakit yang digunakan", expanded=False):
            st.write({
                "Status":         "✅ Ditemukan" if retrieval.get("matched") else "❌ Tidak ditemukan",
                "Tipe pencocokan": retrieval.get("match_type", "-"),
                "Penyakit":        retrieval.get("disease", "-"),
                "Label cocok":     retrieval.get("matched_label", "-"),
            })


def render_recommendation(result: dict[str, Any]) -> None:
    rekom = result.get("recommendation", "")

    st.subheader("💡 Rekomendasi Edukatif")
    st.caption("Rekomendasi ini dibuat oleh LLM berdasarkan hasil prediksi dan konteks penyakit. Ini bukan diagnosis dokter.")

    if rekom:
        st.markdown(rekom)
    else:
        st.info("Rekomendasi belum tersedia untuk kondisi ini.")

    st.warning(DISCLAIMER)


def render_download_report(result: dict[str, Any]) -> None:
    st.subheader("📄 Unduh Laporan")

    report_md = generate_markdown_report(result, st.session_state.uploaded_file_name)
    st.download_button(
        label="⬇️ Unduh Laporan (.md)",
        data=report_md,
        file_name=f"dermsight_laporan_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
        use_container_width=True,
    )


def render_form_3t() -> None:
    st.divider()
    st.subheader("🗣️ Suara Masyarakat Daerah 3T")
    st.caption(
        "Ceritakan kendala akses kesehatan kulit di daerah Anda. "
        "Aspirasi Anda membantu DermSight berkembang untuk semua."
    )

    with st.form("form_3t", clear_on_submit=True):
        nama   = st.text_input("Nama (opsional)", placeholder="Boleh dikosongkan")
        daerah = st.text_input("Daerah / Kabupaten *", placeholder="Contoh: Kab. Yahukimo, Papua")
        kendala = st.text_area(
            "Kendala akses kesehatan kulit di daerah Anda *",
            placeholder="Contoh: tidak ada dokter kulit, jarak ke puskesmas jauh, biaya mahal…",
            height=100,
        )
        edukasi = st.text_area(
            "Edukasi kesehatan kulit yang Anda butuhkan (opsional)",
            placeholder="Contoh: cara membedakan penyakit biasa dan yang perlu dokter…",
            height=80,
        )
        pesan = st.text_area(
            "Pesan tambahan (opsional)",
            placeholder="Apa saja yang ingin Anda sampaikan…",
            height=70,
        )

        submitted = st.form_submit_button("📨 Kirim Aspirasi", use_container_width=True)

        if submitted:
            if not daerah.strip():
                st.warning("⚠️ Mohon isi kolom Daerah/Kabupaten sebelum mengirim.")
            elif not kendala.strip():
                st.warning("⚠️ Mohon ceritakan kendala akses kesehatan kulit di daerah Anda.")
            else:
                row = {
                    "timestamp":         datetime.now().isoformat(),
                    "nama":              nama.strip() or "(anonim)",
                    "daerah":            daerah.strip(),
                    "kendala":           kendala.strip(),
                    "edukasi_dibutuhkan": edukasi.strip(),
                    "pesan_tambahan":    pesan.strip(),
                }
                if save_form_3t(row):
                    st.success("🙏 Terima kasih, aspirasi Anda berhasil disimpan.")
                    logger.info("Form 3T disimpan | daerah=%s", daerah.strip())
                else:
                    st.error("❌ Gagal menyimpan aspirasi. Mohon coba lagi.")


def render_developer_settings() -> tuple[str, str, str, str, str, int]:
    """
    Expander tersembunyi untuk developer. Kembalikan konfigurasi API.
    Default ditutup agar tidak mengganggu user umum.
    """
    with st.expander("⚙️ Pengaturan Developer", expanded=False):
        st.caption("Pengaturan ini hanya untuk pengembang. Pengguna umum tidak perlu mengubah apapun di sini.")

        api_base_url       = st.text_input("FastAPI Base URL",         value=DEFAULT_API_BASE_URL)
        analyze_endpoint   = st.text_input("Endpoint /analyze",        value=DEFAULT_ANALYZE_ENDPOINT)
        predict_endpoint   = st.text_input("Endpoint /predict",        value=DEFAULT_PREDICT_ENDPOINT)
        recommend_endpoint = st.text_input("Endpoint /recommendation", value=DEFAULT_RECOMMEND_ENDPOINT)

        mode_label = st.radio(
            "Mode Analisis",
            ["Analyze langsung (default untuk user)", "Step-by-step (predict → rekomendasi)"],
            help="Analyze langsung memanggil satu endpoint. Step-by-step memisahkan predict dan rekomendasi.",
        )
        mode = "analyze" if "langsung" in mode_label else "step"

        timeout = st.slider("Timeout (detik)", min_value=30, max_value=300, value=120, step=10)

        st.divider()
        st.caption("**Debug Session State**")
        st.json({
            "uploaded_file_name": st.session_state.uploaded_file_name,
            "has_file_bytes":     st.session_state.uploaded_file_bytes is not None,
            "analysis_done":      st.session_state.analysis_done,
            "predicted_label":    (st.session_state.analysis_result or {}).get("predicted_label"),
            "confidence":         (st.session_state.analysis_result or {}).get("confidence"),
            "last_error":         st.session_state.last_error,
        })

    return api_base_url, analyze_endpoint, predict_endpoint, recommend_endpoint, mode, timeout


# MAIN

def main() -> None:
    init_session_state()
    logger.info("DermSight loaded")

    # --- Developer settings (tersembunyi, default tertutup) ---
    (api_base_url, analyze_endpoint, predict_endpoint,
     recommend_endpoint, mode, timeout) = render_developer_settings()

    # --- Header ---
    render_header()

    st.divider()

    # --- Step 1: Panduan foto ---
    render_photo_guide()

    # --- Step 2: Pilih sumber gambar ---
    file_name, file_bytes, content_type = render_image_input()

    st.divider()

    # --- Step 3: Tombol analisis ---
    render_analysis_button(
        api_base_url, analyze_endpoint, predict_endpoint, recommend_endpoint,
        mode, timeout, file_name, file_bytes, content_type,
    )

    # --- Step 4 & 5: Hasil analisis + Rekomendasi ---
    if st.session_state.analysis_done and st.session_state.analysis_result:
        result = st.session_state.analysis_result
        render_result(result)
        render_recommendation(result)
        render_download_report(result)

    # --- Step 7: Form 3T ---
    render_form_3t()

    # --- Footer ---
    st.divider()
    st.caption(
        "🩺 **DermSight** — Sistem edukasi awal kesehatan kulit berbasis AI. "
        "Bukan alat diagnostik medis. Versi Beta · Untuk masyarakat Indonesia."
    )


if __name__ == "__main__":
    main()