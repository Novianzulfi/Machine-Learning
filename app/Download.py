import streamlit as st
import pandas as pd

def app():
    st.title("Download Hasil Prediksi")

    if "prediction" not in st.session_state:
        st.warning("Belum ada hasil prediksi. Silakan jalankan model terlebih dahulu.")
        return

    data = {
        "filename": [st.session_state.get("filename", "-")],
        "predicted_label": [st.session_state["prediction"]],
        "confidence": [st.session_state["confidence"]]
    }

    df = pd.DataFrame(data)

    st.dataframe(df)

    st.download_button(
        label="Download CSV Hasil Prediksi",
        data=df.to_csv(index=False),
        file_name="hasil_prediksi.csv",
        mime="text/csv"
    )
