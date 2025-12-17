import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image

def app():
    st.title("Upload & Preview Dataset (CSV)")

    img = st.file_uploader(
        "Upload Gambar Daun",
        type=["jpg", "jpeg", "png"])
