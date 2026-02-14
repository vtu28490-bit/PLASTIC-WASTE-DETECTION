import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Plastic Waste Detection",
    page_icon="♻️",
    layout="centered"
)

# -------------------------------
# Title
# -------------------------------
st.title("♻️ Plastic Waste Detection App")
st.write("Upload an image or use webcam to detect plastic waste.")

# -------------------------------
# Load YOLO Model
# -------------------------------
@st.cache_resource
def load_model():
    model = YOLO("best.pt")   # trained model file
    return model

model = load_model()

# -------------------------------
# Sidebar Options
# -------------------------------
option = st.sidebar.selectbox(
    "Choose Detection Type",
    ["Image Upload", "Webcam Detection"]
)

# =====================================================
# 1️⃣ IMAGE UPLOAD DETECTION
# =====================================================
if option == "Image Upload":

    uploaded_file = st.file_uploader(
        "Upload Plastic Waste Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        img_array = np.array(image)

        st.image(
            image,
            caption="Uploaded Image",
            use_column_width=True
        )

        # Detection
        results = model(img_array)

        # Annotated output
        annotated_frame = results[0].plot()

        st.image(
            annotated_frame,
            caption="Detected Plastic Waste",
            use_column_width=True
        )

# =====================================================
# 2️⃣ WEBCAM DETECTION
# =====================================================
elif option == "Webcam Detection":

    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.write("Failed to access webcam")
            break

        # YOLO detection
        results = model(frame)
        annotated_frame = results[0].plot()

        # Show frame
        FRAME_WINDOW.image(
            annotated_frame,
            channels="BGR"
        )

    camera.release()

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
    "Developed for Plastic Waste Detection Project ♻️"
)
