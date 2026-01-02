import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

def app():
    st.title("Explainability Model (Grad-CAM)")

    if "img_array" not in st.session_state:
        st.warning("Silakan upload gambar terlebih dahulu di menu Upload & Preview")
        return

    model = load_model("model/model_daun_terong.h5")
    img_array = st.session_state["img_array"]
    image = np.array(st.session_state["image"])

    # Cari layer convolution terakhir
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)

    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    st.image(overlay, caption="Grad-CAM Heatmap", use_column_width=True)

    st.info("""
    Warna merah dan kuning menunjukkan area daun
    yang paling berpengaruh terhadap keputusan model.
    """)
