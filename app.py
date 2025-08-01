 
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

st.title("🧍‍♀️ Viola-Jones Face Detection App")
st.write("Upload an image, choose rectangle color, adjust detection parameters, and save the result.")

rect_color = st.color_picker("Pick a rectangle color", "#FF0000")
scale_factor = st.slider("Scale Factor", 1.05, 1.5, 1.1, 0.01)
min_neighbors = st.slider("Min Neighbors", 1, 10, 5, 1)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

    hex_color = rect_color.lstrip('#')
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    for (x, y, w, h) in faces:
        cv2.rectangle(img_np, (x, y), (x + w, y + h), rgb_color, 2)

    st.image(img_np, caption="Detected Faces", use_column_width=True)

    if st.button("Save Image"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            save_path = tmp_file.name
            cv2.imwrite(save_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            st.success("✅ Image saved!")
            st.download_button("Download Image", open(save_path, "rb"), file_name="detected_faces.jpg")
