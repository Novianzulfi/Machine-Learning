import streamlit as st
import pandas as pd

def app():
    st.title("Download Hasil Prediksi")

    if "pred_result" not in st.session_state:
        st.warning("Belum ada hasil prediksi. Silakan jalankan model terlebih dahulu.")
        return

    result = st.session_state["pred_result"]

    df = pd.DataFrame([result])

    st.dataframe(df)

    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False),
        file_name="hasil_prediksi.csv",
        mime="text/csv"
    )
