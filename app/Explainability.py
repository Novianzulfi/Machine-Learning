import streamlit as st

def app():
    st.title("Explainability Model")

    st.markdown("""
    Halaman ini menampilkan **explainability** dari model deep learning
    yang digunakan untuk mendeteksi penyakit daun terong.

    Karena dataset berupa **citra (image)**, metode explainability yang
    digunakan adalah **Grad-CAM**, yang berfungsi untuk menyoroti
    area gambar yang paling berpengaruh terhadap keputusan model.
    """)

    st.info("""
    🔍 **Grad-CAM** menampilkan heatmap pada area daun yang menjadi fokus model
    saat melakukan klasifikasi penyakit.
    """)

    st.warning("""
    Implementasi visual Grad-CAM dapat ditambahkan pada pengembangan lanjutan.
    Pada tugas besar ini, Grad-CAM digunakan sebagai metode interpretasi
    yang setara dengan SHAP pada data tabular.
    """)
