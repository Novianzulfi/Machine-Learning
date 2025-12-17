import streamlit as st

st.set_page_config(
    page_title="Deteksi Penyakit Daun Terong",
    page_icon="🌿",
    layout="wide"
)

st.sidebar.title("Menu Navigasi")

menu = st.sidebar.radio(
    "Pilih Halaman:",
    [
        "Home",
        "Upload & Preview Dataset",
        "Run Model",
        "Explainability",
        "Download Hasil"
    ]
)

if menu == "Home":
    from home import app
    app()

elif menu == "Upload & Preview Dataset":
    from Uploud_Preview import app
    app()

elif menu == "Run Model":
    from Run_Model import app
    app()

elif menu == "Explainability":
    from Explainability import app
    app()

elif menu == "Download Hasil":
    from Download import app
    app()
