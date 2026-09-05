# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 09:51:23 2026

@author: debaj
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)


# --------------------------------------------------
# PlantVillage class names
# The order must match the model's training classes.
# --------------------------------------------------

class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# --------------------------------------------------
# Load the trained MobileNetV2 model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_mobilenetv2.keras")


model = load_model()


# --------------------------------------------------
# Application interface
# --------------------------------------------------

st.title("🌿 Plant Disease Detection")

st.write(
    "Upload a plant leaf image and the trained MobileNetV2 "
    "model will predict the disease or healthy class."
)

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Predict Disease"):

        # Resize image to the same size used during training
        image_resized = image.resize((224, 224))

        # Convert image to NumPy array
        image_array = np.array(image_resized)

        # Normalize pixel values to the same [0, 1] range
        # used by the trained model
        image_array = image_array / 255.0

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # Make prediction
        predictions = model.predict(
            image_array,
            verbose=0
        )

        # Find the class with the highest probability
        predicted_index = np.argmax(predictions[0])

        predicted_class = class_names[predicted_index]

        confidence = predictions[0][predicted_index] * 100

        # Display result
        st.success(
            f"Prediction: {predicted_class}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )