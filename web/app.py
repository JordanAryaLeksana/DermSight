import logging
import streamlit as st

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

logger.info("Streamlit app initialized")

st.title("DermSight")
st.subheader("AI-powered skin disease assistant")

uploaded_file = st.file_uploader(
    "Upload gambar kulit",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file:
    logger.info(
        "Image uploaded | filename=%s | type=%s | size=%s bytes",
        uploaded_file.name,
        uploaded_file.type,
        uploaded_file.size,
    )

    st.image(
        uploaded_file,
        caption="Gambar yang diupload",
        use_container_width=True,
    )

    if st.button("Analisis"):
        logger.info("Analyze button clicked")

        try:
            st.info("Model AI nanti dipanggil di sini.")

            # Nanti flow-nya kira-kira:
            # 1. preprocessing image
            # 2. model prediction
            # 3. RAG retrieval
            # 4. LLM recommendation
            # 5. display result

            logger.info("Analysis placeholder executed successfully")

        except Exception:
            logger.exception("Analysis failed")
            st.error("Terjadi error saat analisis. Cek file logs/streamlit.log.")
else:
    logger.debug("No image uploaded yet")