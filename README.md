# 🌿 Plant Disease Detection using CNN & Transfer Learning

A deep learning-based image classification project that identifies plant diseases from leaf images using **Convolutional Neural Networks (CNN)** and **MobileNetV2 Transfer Learning**.

The final model classifies images into **38 plant disease and healthy categories** and is deployed as an interactive **Streamlit web application**.

---

## 📌 Project Overview

Plant diseases can significantly affect crop productivity and agricultural output. Manual identification of diseases from leaf images can be time-consuming and requires domain expertise.

This project develops an image classification system that takes a plant leaf image as input and predicts the corresponding disease or healthy condition.

Two approaches were implemented:

1. **Baseline CNN**
2. **MobileNetV2 using Transfer Learning**

The MobileNetV2 model was selected as the final model based on its validation performance.

---

## 🎯 Objectives

- Build an image classification model for plant disease detection.
- Perform image preprocessing and data augmentation.
- Develop a baseline CNN model.
- Apply transfer learning using MobileNetV2.
- Compare model performance.
- Evaluate classification performance across 38 classes.
- Deploy the trained model using Streamlit.

---

## 📊 Dataset

The project uses the **PlantVillage dataset**.

### Dataset Statistics

| Property | Value |
|---|---:|
| Total Images | 54,305 |
| Training Images | 43,444 |
| Validation Images | 10,861 |
| Number of Classes | 38 |
| Image Size | 256 × 256 |
| Model Input Size | 224 × 224 |

The dataset contains images of leaves from multiple crops including:

- Apple
- Blueberry
- Cherry
- Corn
- Grape
- Orange
- Peach
- Pepper
- Potato
- Raspberry
- Soybean
- Squash
- Strawberry
- Tomato

---

## 🧠 Classes

The model predicts one of the following 38 classes:

```text
Apple___Apple_scab
Apple___Black_rot
Apple___Cedar_apple_rust
Apple___healthy
Blueberry___healthy
Cherry_(including_sour)___Powdery_mildew
Cherry_(including_sour)___healthy
Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
Corn_(maize)___Common_rust_
Corn_(maize)___Northern_Leaf_Blight
Corn_(maize)___healthy
Grape___Black_rot
Grape___Esca_(Black_Measles)
Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
Grape___healthy
Orange___Haunglongbing_(Citrus_greening)
Peach___Bacterial_spot
Peach___healthy
Pepper,_bell___Bacterial_spot
Pepper,_bell___healthy
Potato___Early_blight
Potato___Late_blight
Potato___healthy
Raspberry___healthy
Soybean___healthy
Squash___Powdery_mildew
Strawberry___Leaf_scorch
Strawberry___healthy
Tomato___Bacterial_spot
Tomato___Early_blight
Tomato___Late_blight
Tomato___Leaf_Mold
Tomato___Septoria_leaf_spot
Tomato___Spider_mites Two-spotted_spider_mite
Tomato___Target_Spot
Tomato___Tomato_Yellow_Leaf_Curl_Virus
Tomato___Tomato_mosaic_virus
Tomato___healthy