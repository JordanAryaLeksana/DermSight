import streamlit as st

st.set_page_config(
    page_title="DermSight",
    page_icon="🩺",
    layout="wide"
)

st.title("DermSight")
st.subheader("AI-powered skin disease assistant")

uploaded_file = st.file_uploader(
    "Upload gambar kulit",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    st.image(uploaded_file, caption="Gambar yang diupload", use_container_width=True)

    if st.button("Analisis"):
        st.info("Model AI nanti dipanggil di sini.")