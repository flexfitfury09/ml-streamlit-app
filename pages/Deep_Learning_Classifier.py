import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image
import numpy as np

st.set_page_config(page_title="Image Classifier", layout="centered")
st.title("🖼️ Deep Learning Image Classifier")
st.write("This module uses a pre-trained CNN to identify objects in your images.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"], key="image_uploader")

@st.cache_resource
def load_model():
    model = MobileNetV2(weights='imagenet')
    return model

model = load_model()

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    with st.spinner("Analyzing image..."):
        img_resized = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img_resized), axis=0)
        img_preprocessed = preprocess_input(img_array)
        predictions = model.predict(img_preprocessed)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
    st.header("✅ Prediction Results")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        st.write(f"{i+1}: **{label.replace('_', ' ').title()}** (Confidence: {score:.2%})")