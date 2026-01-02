import streamlit as st
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ===============================
# CACHE MODEL
# ===============================
@st.cache_resource
def load_model_cached():
    return load_model("model/model_daun_terong.h5")


# ===============================
# HITUNG EVALUASI DARI DATASET
# ===============================
def compute_dataset_evaluation(model, class_names):
    # PATH AMAN & BENAR
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(BASE_DIR, "..", "dataset")

    datagen = ImageDataGenerator(rescale=1./255)

    generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode="categorical",
        shuffle=False
    )

    # 🔎 DEBUG (penting, tapi aman)
    st.write("📂 Dataset path:", dataset_dir)
    st.write("📊 Kelas terdeteksi:", generator.class_indices)

    y_true = generator.classes
    y_pred = model.predict(generator)
    y_pred_classes = np.argmax(y_pred, axis=1)

    report = classification_report(
        y_true,
        y_pred_classes,
        target_names=class_names
    )

    cm = confusion_matrix(y_true, y_pred_classes)

    return report, cm


def app():
    st.title("Run Model, Prediksi & Evaluasi")

    model = load_model_cached()

    with open("model/class_indices.json") as f:
        class_indices = json.load(f)

    class_names = list(class_indices.keys())

    # ===============================
    # 1️⃣ PREDIKSI 1 GAMBAR
    # ===============================
    st.subheader("Prediksi Gambar Daun")

    if "img_array" not in st.session_state:
        st.warning("Silakan upload gambar terlebih dahulu di menu Upload & Preview")
    else:
        img_array = st.session_state["img_array"]

        preds = model.predict(img_array)
        pred_class = class_names[np.argmax(preds)]
        confidence = float(np.max(preds))

        st.session_state["prediction"] = pred_class
        st.session_state["confidence"] = confidence

        st.success(f"Hasil Prediksi: {pred_class}")
        st.write("Confidence:", confidence)

    st.divider()

    # ===============================
    # 2️⃣ EVALUASI MODEL (DATASET)
    # ===============================
    st.subheader("Evaluasi Model (Dataset)")

    if "cm" not in st.session_state:
        with st.spinner("Menghitung evaluasi model (sekali saja)..."):
            report, cm = compute_dataset_evaluation(model, class_names)
            st.session_state["report"] = report
            st.session_state["cm"] = cm
    else:
        report = st.session_state["report"]
        cm = st.session_state["cm"]

    # ===== METRICS =====
    st.subheader("Classification Report")
    st.text(report)

    # ===== CONFUSION MATRIX =====
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm, cmap="Blues")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center", color="black")

    st.pyplot(fig)
