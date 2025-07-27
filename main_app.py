import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown
from tensorflow.keras.preprocessing import image

# 📥 Download model from Google Drive (if not already downloaded)
model_path = "transfer_model.keras"

if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1CrXhoOBd8-ZAoVCfRJi_zc2rOwa3eShy"  
    gdown.download(url, model_path, quiet=False)

# 🔍 Load the model
model = tf.keras.models.load_model(model_path)

# 🧠 Class names
class_names = ["glioma", "meningioma", "pituitary", "no tumor"]

# 🖼️ Streamlit UI
st.title("🧠 Brain Tumor MRI Classifier")

uploaded_file = st.file_uploader("Upload an MRI scan", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.markdown(f"### 🎯 Prediction: **{predicted_class}**")
    st.markdown(f"### 📊 Confidence: **{confidence*100:.2f}%**")
