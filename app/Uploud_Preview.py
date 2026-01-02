import streamlit as st
import numpy as np
from PIL import Image

def app():
    st.title("Upload & Preview Gambar Daun")

    img = st.file_uploader(
        "Upload gambar daun terong",
        type=["jpg", "jpeg", "png"]
    )

    if img is not None:
        image = Image.open(img).convert("RGB")
        image_resized = image.resize((224, 224))

        img_array = np.array(image_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # SIMPAN KE SESSION STATE
        st.session_state["image"] = image
        st.session_state["img_array"] = img_array
        st.session_state["filename"] = img.name

        st.image(image, caption="Preview Gambar", use_column_width=True)
        st.success("Gambar berhasil disimpan dan siap diproses")

