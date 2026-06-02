import logging
import os
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import requests
import streamlit as st
from PIL import Image

from utils.logger import setup_logger, get_logger

setup_logger(
    level=logging.INFO,
    log_dir="logs",
    log_file="streamlit.log",
)

logger = get_logger(__name__)



st.set_page_config(
    page_title="DermSight",
    page_icon="🩺",
    layout="wide",
)


DEFAULT_API_BASE_URL = os.getenv("DERMSIGHT_API_BASE_URL", "http://localhost:8000")

# Ganti ini sesuai prefix FastAPI kamu.
# Kalau router kamu include seperti:
# app.include_router(router, prefix="/api/skin")
# maka endpoint-nya jadi "/api/skin/analyze"
DEFAULT_ANALYZE_ENDPOINT = os.getenv("DERMSIGHT_ANALYZE_ENDPOINT", "/analyze")
DEFAULT_PREDICT_ENDPOINT = os.getenv("DERMSIGHT_PREDICT_ENDPOINT", "/predict")
DEFAULT_RECOMMEND_ENDPOINT = os.getenv("DERMSIGHT_RECOMMEND_ENDPOINT", "/generate")


ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png"]


def init_session_state() -> None:
    if "uploaded_file_bytes" not in st.session_state:
        st.session_state.uploaded_file_bytes = None

    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = None

    if "uploaded_file_type" not in st.session_state:
        st.session_state.uploaded_file_type = None

    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None

    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None

    if "recommendation_result" not in st.session_state:
        st.session_state.recommendation_result = None


def build_url(base_url: str, endpoint: str) -> str:
    base_url = base_url.rstrip("/")
    endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    return f"{base_url}{endpoint}"


def get_uploaded_file_payload(uploaded_file) -> tuple[str, bytes, str]:
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
    content_type = uploaded_file.type

    return file_name, file_bytes, content_type


def call_analyze_api(
    api_base_url: str,
    analyze_endpoint: str,
    file_name: str,
    file_bytes: bytes,
    content_type: str,
    timeout: int = 120,
) -> dict[str, Any]:
    url = build_url(api_base_url, analyze_endpoint)

    logger.info("Calling analyze API | url=%s | filename=%s", url, file_name)

    files = {
        "image": (file_name, file_bytes, content_type)
    }

    response = requests.post(
        url,
        files=files,
        timeout=timeout,
    )

    logger.info(
        "Analyze API response received | status_code=%s",
        response.status_code,
    )

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text

        logger.error(
            "Analyze API failed | status_code=%s | detail=%s",
            response.status_code,
            detail,
        )

        raise RuntimeError(f"API error {response.status_code}: {detail}")

    return response.json()


def call_predict_api(
    api_base_url: str,
    predict_endpoint: str,
    file_name: str,
    file_bytes: bytes,
    content_type: str,
    timeout: int = 120,
) -> dict[str, Any]:
    url = build_url(api_base_url, predict_endpoint)

    logger.info("Calling predict API | url=%s | filename=%s", url, file_name)

    files = {
        "image": (file_name, file_bytes, content_type)
    }

    response = requests.post(
        url,
        files=files,
        timeout=timeout,
    )

    logger.info(
        "Predict API response received | status_code=%s",
        response.status_code,
    )

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text

        logger.error(
            "Predict API failed | status_code=%s | detail=%s",
            response.status_code,
            detail,
        )

        raise RuntimeError(f"API error {response.status_code}: {detail}")

    return response.json()


def call_recommendation_api(
    api_base_url: str,
    recommend_endpoint: str,
    predicted_label: str,
    confidence: float,
    timeout: int = 120,
) -> dict[str, Any]:
    url = build_url(api_base_url, recommend_endpoint)

    logger.info(
        "Calling recommendation API | url=%s | predicted_label=%s | confidence=%s",
        url,
        predicted_label,
        confidence,
    )

    payload = {
        "predicted_label": predicted_label,
        "confidence": confidence,
    }

    response = requests.post(
        url,
        json=payload,
        timeout=timeout,
    )

    logger.info(
        "Recommendation API response received | status_code=%s",
        response.status_code,
    )

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text

        logger.error(
            "Recommendation API failed | status_code=%s | detail=%s",
            response.status_code,
            detail,
        )

        raise RuntimeError(f"API error {response.status_code}: {detail}")

    return response.json()


def render_confidence(confidence: float) -> None:
    confidence_percentage = confidence * 100

    st.metric(
        label="Confidence",
        value=f"{confidence_percentage:.2f}%",
    )

    st.progress(min(max(confidence, 0.0), 1.0))

    if confidence < 0.50:
        st.warning("Confidence rendah. Gunakan hasil ini dengan sangat hati-hati.")
    elif confidence < 0.75:
        st.info("Confidence sedang. Hasil sebaiknya tetap diverifikasi.")
    else:
        st.success("Confidence cukup tinggi.")


def generate_markdown_report(result: dict[str, Any]) -> str:
    predicted_label = result.get("predicted_label", "-")
    confidence = result.get("confidence", 0.0)
    recommendation = result.get("recommendation", "-")

    return f"""# DermSight Analysis Report

## Prediction Result

- Predicted Label: {predicted_label}
- Confidence: {confidence:.2%}

## AI Recommendation

{recommendation}

## Disclaimer

DermSight adalah sistem bantuan berbasis AI untuk edukasi dan pendukung keputusan awal. 
Hasil dari sistem ini bukan diagnosis medis resmi. Untuk keluhan yang serius, memburuk, menyebar, berdarah, nyeri, atau berubah bentuk/warna, segera konsultasikan dengan dokter atau dermatolog.
"""


init_session_state()

logger.info("Streamlit app initialized")


st.title("🩺 DermSight")
st.subheader("AI-powered skin disease assistant")

st.caption(
    "Upload gambar kulit, lalu sistem akan memanggil FastAPI lokal untuk prediksi penyakit kulit dan rekomendasi AI."
)


with st.sidebar:
    st.header("API Settings")

    api_base_url = st.text_input(
        "FastAPI Base URL",
        value=DEFAULT_API_BASE_URL,
    )

    analyze_endpoint = st.text_input(
        "Analyze Endpoint",
        value=DEFAULT_ANALYZE_ENDPOINT,
        help="Endpoint untuk model + rekomendasi. Contoh: /analyze atau /api/skin/analyze",
    )

    predict_endpoint = st.text_input(
        "Predict Endpoint",
        value=DEFAULT_PREDICT_ENDPOINT,
        help="Endpoint untuk prediksi saja. Contoh: /predict atau /api/skin/predict",
    )

    recommend_endpoint = st.text_input(
        "Recommendation Endpoint",
        value=DEFAULT_RECOMMEND_ENDPOINT,
        help="Endpoint untuk rekomendasi saja. Contoh: /generate atau /api/recommendation/generate",
    )

    st.divider()

    mode = st.radio(
        "Analysis Mode",
        options=[
            "Analyze langsung: predict + recommendation",
            "Step-by-step: predict lalu recommendation",
        ],
    )

    st.divider()

    st.info(
        "Pastikan FastAPI sudah jalan, misalnya:\n\n"
        "`uvicorn src.api.main:app --reload`"
    )


tab_upload, tab_result, tab_logs = st.tabs(
    [
        "Upload & Analyze",
        "Result",
        "Debug Info",
    ]
)


with tab_upload:
    st.header("Upload Gambar Kulit")

    uploaded_file = st.file_uploader(
        "Upload gambar kulit",
        type=ALLOWED_IMAGE_TYPES,
    )

    if uploaded_file is not None:
        file_name, file_bytes, content_type = get_uploaded_file_payload(uploaded_file)

        st.session_state.uploaded_file_bytes = file_bytes
        st.session_state.uploaded_file_name = file_name
        st.session_state.uploaded_file_type = content_type

        logger.info(
            "Image uploaded | filename=%s | type=%s | size=%s bytes",
            file_name,
            content_type,
            len(file_bytes),
        )

        col_image, col_info = st.columns([1, 1])

        with col_image:
            st.image(
                uploaded_file,
                caption="Gambar yang diupload",
                use_container_width=True,
            )

        with col_info:
            st.subheader("Informasi File")
            st.write("**Filename:**", file_name)
            st.write("**Content type:**", content_type)
            st.write("**Size:**", f"{len(file_bytes):,} bytes")

            try:
                image = Image.open(uploaded_file)
                st.write("**Image size:**", f"{image.width} x {image.height}")
            except Exception:
                logger.exception("Failed to read image metadata")
                st.warning("Tidak bisa membaca metadata gambar.")

        st.divider()

        analyze_button = st.button(
            "🔍 Analisis Gambar",
            type="primary",
            use_container_width=True,
        )

        if analyze_button:
            logger.info("Analyze button clicked")

            try:
                if mode == "Analyze langsung: predict + recommendation":
                    with st.spinner("Mengirim gambar ke FastAPI /analyze..."):
                        result = call_analyze_api(
                            api_base_url=api_base_url,
                            analyze_endpoint=analyze_endpoint,
                            file_name=file_name,
                            file_bytes=file_bytes,
                            content_type=content_type,
                        )

                    st.session_state.analysis_result = result
                    st.session_state.prediction_result = {
                        "predicted_label": result.get("predicted_label"),
                        "confidence": result.get("confidence"),
                    }
                    st.session_state.recommendation_result = {
                        "predicted_label": result.get("predicted_label"),
                        "confidence": result.get("confidence"),
                        "recommendation": result.get("recommendation"),
                    }

                    logger.info(
                        "Analyze completed | predicted_label=%s | confidence=%s",
                        result.get("predicted_label"),
                        result.get("confidence"),
                    )

                else:
                    with st.spinner("Step 1/2: Mengirim gambar ke FastAPI /predict..."):
                        prediction = call_predict_api(
                            api_base_url=api_base_url,
                            predict_endpoint=predict_endpoint,
                            file_name=file_name,
                            file_bytes=file_bytes,
                            content_type=content_type,
                        )

                    predicted_label = prediction["predicted_label"]
                    confidence = prediction["confidence"]

                    st.session_state.prediction_result = prediction

                    logger.info(
                        "Prediction completed | predicted_label=%s | confidence=%s",
                        predicted_label,
                        confidence,
                    )

                    with st.spinner("Step 2/2: Meminta rekomendasi ke FastAPI /generate..."):
                        recommendation = call_recommendation_api(
                            api_base_url=api_base_url,
                            recommend_endpoint=recommend_endpoint,
                            predicted_label=predicted_label,
                            confidence=confidence,
                        )

                    st.session_state.recommendation_result = recommendation
                    st.session_state.analysis_result = {
                        "predicted_label": prediction.get("predicted_label"),
                        "confidence": prediction.get("confidence"),
                        "recommendation": recommendation.get("recommendation"),
                    }

                    logger.info("Recommendation completed")

                st.success("Analisis berhasil. Buka tab Result untuk melihat hasil.")

            except requests.exceptions.ConnectionError:
                logger.exception("Connection error while calling FastAPI")
                st.error(
                    "Tidak bisa terhubung ke FastAPI. "
                    "Pastikan server FastAPI sudah jalan di localhost."
                )

            except requests.exceptions.Timeout:
                logger.exception("Timeout while calling FastAPI")
                st.error("Request ke FastAPI timeout. Coba lagi atau naikkan timeout.")

            except Exception as exc:
                logger.exception("Analysis failed")
                st.error(f"Terjadi error saat analisis: {exc}")
                st.caption("Detail error juga bisa dicek di `logs/streamlit.log`.")

    else:
        logger.debug("No image uploaded yet")
        st.info("Silakan upload gambar kulit terlebih dahulu.")


with tab_result:
    st.header("Hasil Analisis")

    result = st.session_state.analysis_result

    if result is None:
        st.info("Belum ada hasil. Upload gambar lalu klik tombol Analisis.")
    else:
        predicted_label = result.get("predicted_label", "-")
        confidence = float(result.get("confidence", 0.0) or 0.0)
        recommendation = result.get("recommendation", "-")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Gambar")
            if st.session_state.uploaded_file_bytes is not None:
                st.image(
                    st.session_state.uploaded_file_bytes,
                    caption=st.session_state.uploaded_file_name,
                    use_container_width=True,
                )

        with col2:
            st.subheader("Prediction")

            st.metric(
                label="Predicted Skin Disease",
                value=predicted_label,
            )

            render_confidence(confidence)

        st.divider()

        st.subheader("AI Recommendation")
        st.markdown(recommendation)

        st.divider()

        st.subheader("Download Report")

        report = generate_markdown_report(result)

        st.download_button(
            label="📄 Download Markdown Report",
            data=report,
            file_name="dermsight_analysis_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

        with st.expander("Lihat raw JSON response"):
            st.json(result)


with tab_logs:
    st.header("Debug Info")

    st.subheader("Current API URL")

    st.code(
        f"""API_BASE_URL={api_base_url}
ANALYZE_URL={build_url(api_base_url, analyze_endpoint)}
PREDICT_URL={build_url(api_base_url, predict_endpoint)}
RECOMMEND_URL={build_url(api_base_url, recommend_endpoint)}
""",
        language="bash",
    )

    st.subheader("Session State")

    debug_state = {
        "uploaded_file_name": st.session_state.uploaded_file_name,
        "uploaded_file_type": st.session_state.uploaded_file_type,
        "has_uploaded_file_bytes": st.session_state.uploaded_file_bytes is not None,
        "prediction_result": st.session_state.prediction_result,
        "recommendation_result": st.session_state.recommendation_result,
        "analysis_result": st.session_state.analysis_result,
    }

    st.json(debug_state)

    st.subheader("Expected FastAPI Response")

    st.write("Untuk endpoint `/analyze`, Streamlit mengharapkan response seperti ini:")

    st.code(
        """{
  "predicted_label": "Acne",
  "confidence": 0.92,
  "recommendation": "..."
}""",
        language="json",
    )