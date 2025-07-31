import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.metrics import Precision, Recall
from keras.metrics import F1Score
import cv2
import gdown
import os

# Download sekali saat pertama kali dijalankan
model_path = "vgg16_model.h5"
if not os.path.exists(model_path):
    gdown.download("https://drive.google.com/uc?id=/1DJBGGLGVinkBD5fs7aPn5toXUdPC1bwr", model_path, quiet=False)

model = tf.keras.models.load_model(
    model_path,
    custom_objects={
        "Precision": Precision,
        "Recall": Recall,
        "F1Score": F1Score
    }
)

# Label mapping
class_names = ["healthy", "bean_rust", "angular_leaf_spot"]  # ganti sesuai label kamu

# UI
st.set_page_config(page_title="Klasifikasi Daun Kacang", page_icon="🌿", layout="centered")
st.markdown("<h3 style='text-align: center;'>Klasifikasi Daun Kacang Menggunakan VGG16</h3>", unsafe_allow_html=True)
st.caption("Upload gambar daun untuk memprediksi kesehatannya.")


uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang Diupload", width=300)

    # Preprocessing
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)
    img_array = img_array / 255.0  # Normalisasi

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]

    st.markdown("---")
    st.markdown(f"<p style='font-size: 18px;'>🧠 <b>Prediksi:</b> {class_names[predicted_class]}</p>", unsafe_allow_html=True)

    st.markdown("<p style='margin-bottom: 4px;'>📊 <b>Probabilitas:</b></p>", unsafe_allow_html=True)
    for i, prob in enumerate(prediction[0]):
        st.markdown(f"<small>{class_names[i]}: {prob*100:.2f}%</small>", unsafe_allow_html=True)
        st.progress(float(prob))

