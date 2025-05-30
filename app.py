import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
import pickle
import cv2

# Dummy Credentials (Replace with database authentication if needed)
USER_CREDENTIALS = {"admin": "password123", "user": "foodlover"}

# Load Label Encoder
with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

# Load Models
vgg_model = tf.keras.models.load_model(r"C:\Users\abhip\Music\AI NUTRITION\AI NUTRITION\FrontEnd\vgg16.h5")
resnet_model = tf.keras.models.load_model(r"C:\Users\abhip\Music\AI NUTRITION\AI NUTRITION\FrontEnd\resnet.h5")

# Define Nutritional Information
nutrition_data = {
    "baby_back_ribs": {"Calories": 430, "Protein": "40g", "Carbs": "15g", "Fat": "28g"},
    "chicken_wings": {"Calories": 430, "Protein": "35g", "Carbs": "2g", "Fat": "30g"},
    "chocolate_cake": {"Calories": 350, "Protein": "5g", "Carbs": "50g", "Fat": "15g"},
    "frozen_yogurt": {"Calories": 180, "Protein": "6g", "Carbs": "36g", "Fat": "2g"},
}

# Image Preprocessing Function
def preprocess_uploaded_image(img, model_name):
    img = cv2.imdecode(np.frombuffer(img.read(), np.uint8), 1)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)

    if model_name == "VGG-16":
        img = vgg_preprocess(img)
    elif model_name == "ResNet":
        img = resnet_preprocess(img)

    return img

# Session State for Authentication
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

def login():
    st.title("🔐 Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid username or password. Try again.")

if not st.session_state["authenticated"]:
    login()
else:
    # Main App
    st.title("🍔 Food Classification App - VGG-16 & ResNet")
    st.write("Upload an image and choose a model to classify the food item.")

    # Logout Button
    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.experimental_rerun()

    # File Uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    # Model Selection
    model_choice = st.selectbox("Choose a Model", ["VGG-16", "ResNet-50"])

    # Predict Button
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            with st.spinner("Classifying..."):
                # Preprocess Image
                img_array = preprocess_uploaded_image(uploaded_file, model_choice)

                # Get Model
                model = vgg_model if model_choice == "VGG-16" else resnet_model

                # Predict
                preds = model.predict(img_array)
                predicted_label = label_encoder.inverse_transform([np.argmax(preds)])[0]

                # Display Result
                st.success(f"**Prediction:** {predicted_label}")

                # Display Nutritional Information if food is in the list
                if predicted_label in nutrition_data:
                    st.subheader("🍽️ Nutritional Information:")
                    for key, value in nutrition_data[predicted_label].items():
                        st.write(f"**{key}:** {value}")
