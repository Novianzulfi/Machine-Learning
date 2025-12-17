import streamlit as st
import numpy as np
import json
from tensorflow.keras.models import load_model
from PIL import Image

def app():
    st.write("DEBUG: Run_Model.py terpanggil")
    st.title("Run Model & Prediksi")

    model = load_model("model/model_daun_terong.h5")

    with open("model/class_indices.json") as f:
        class_indices = json.load(f)

    class_names = list(class_indices.keys())

    img = st.file_uploader(
        "Upload Gambar Daun",
        type=["jpg", "jpeg", "png"]
    )

    st.write("DEBUG img:", img)

    if img is not None:
        image = Image.open(img).convert("RGB")
        image = image.resize((224, 224))
        st.image(image, caption="Gambar Daun Terong", use_column_width=True)

        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)

        predicted_class = class_names[np.argmax(pred)]
        confidence = float(np.max(pred))

        st.success(f"Hasil Prediksi: {predicted_class}")
        st.write(f"Confidence: {confidence:.4f}")

        st.session_state["pred_result"] = {
        "filename": img.name,
        "predicted_label": predicted_class,
        "confidence": confidence
    }

    st.write("DEBUG session_state:", st.session_state["pred_result"])