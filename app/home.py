import streamlit as st

def app():
    st.title("Deteksi Penyakit Daun Terong")

    st.markdown("""
    Aplikasi ini menggunakan **Deep Learning (CNN + MobileNetV2)**  
    untuk mengklasifikasikan penyakit daun terong berdasarkan citra.

    ### Kelas Penyakit:
    - Healthy Leaf
    - Insect Pest Disease
    - Leaf Spot Disease
    - Mosaic Virus Disease
    - Small Leaf Disease
    - White Mold Disease
    - Wilt Disease
    """)


