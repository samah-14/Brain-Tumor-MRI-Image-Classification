import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown


os.system('pip install tensorflow==2.12.0 gdown numpy pillow')

model_path = "transfer_model.keras"
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1CrXhoOBd8-ZAoVCfRJi_zc2rOwa3eShy"
    gdown.download(url, model_path, quiet=False)

model = tf.keras.models.load_model(model_path)
class_names = ['glioma', 'meningioma', 'no tumor', 'pituitary']

st.title("🧠 Brain Tumor MRI Classifier")
uploaded_file = st.file_uploader("Upload an MRI scan", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(np.max(prediction) * 100, 2)
    st.markdown(f"### 🎯 Prediction: **{predicted_class}**")
    st.markdown(f"### 📊 Confidence: **{confidence}%**")

