import streamlit as st
import cv2
import os
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
from streamlit_drawable_canvas import st_canvas
import uuid

np.set_printoptions(suppress=True)

st.set_page_config(page_title="HealthSense-AI Parkinson's Detection", page_icon=":medical_symbol:")
st.title("HealthSense-AI Parkinson's Detection")
st.header("Unveiling Parkinson's with Precision - 83% Accurate Detection Algorithm Trained on Spiral Sketches.")
st.write("Try drawing a Spiral and watch how an AI Model will detect the Parkinson Disease.")
st.caption("The application will infer the one label out of 2 labels, as follows: Healthy, Parkinson Diseased.")
st.warning("Warning: Do not click Submit Sketch button before drawing spiral on below Canvas.")

with st.sidebar:
    img = Image.open("./Images/parkinson_disease_detection.jpg")
    st.image(img)
    st.subheader("About Parkinson's Disease Detection")
    link_text = "Distinguishing Different Stages of Parkinson's Disease Using Composite Index of Speed and Pen-Pressure of Sketching a Spiral"
    link_url = "https://www.frontiersin.org/articles/10.3389/fneur.2017.00435/full"
    st.write("Parkinson's disease is a neurodegenerative disorder that affects motor functions, leading to tremors, stiffness, and impaired movement. The research presented in the article link mentioned below explores the use of spiral and wave sketch images to develop a robust algorithm for Parkinson's disease detection. Parkonix leverages these sketch images to train an AI model, achieving an impressive accuracy rate of 83%.")
    st.markdown(f"[{link_text}]({link_url})")
    st.header("Dataset")
    img = Image.open("./Images/healthy_diseased_classification.jpeg")
    st.image(img)
    st.header("Drawing Canvas Configurations")

drawing_mode = "freedraw"
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
realtime_update = st.sidebar.checkbox("Update in realtime", True)

col1, col2 = st.columns(2)
canvas_size = 345

with col1:
    st.subheader("Drawable Canvas")
    canvas_image = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        width=canvas_size,
        height=canvas_size,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        drawing_mode=drawing_mode,
        key="canvas",
    )

with col2:
    st.subheader("Preview")
    if canvas_image.image_data is not None:
        input_numpy_array = np.array(canvas_image.image_data)
        input_image = Image.fromarray(input_numpy_array.astype("uint8"), "RGBA")
        st.image(input_image, use_column_width=True)

def generate_user_input_filename():
    unique_id = uuid.uuid4().hex
    filename = f"user_input_{unique_id}.png"
    return filename

def predict_parkinsons(img_path):
    best_model = load_model("./keras_model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    input_numpy_array = np.array(img_path.image_data)
    input_image = Image.fromarray(input_numpy_array.astype("uint8"), "RGBA")
    user_input_filename = generate_user_input_filename()
    input_image.save(user_input_filename)
    image = Image.open(user_input_filename).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    prediction = best_model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    Detection_Result = f"The model has detected {class_name[2:]}, with Confidence Score: {str(np.round(confidence_score * 100))[:-2]}%."
    os.remove(user_input_filename)
    return Detection_Result, prediction

submit = st.button(label="Submit Sketch")
if submit:
    st.subheader("Output")
    classified_label, prediction = predict_parkinsons(canvas_image)
    with st.spinner(text="This may take a moment..."):
        st.write(classified_label)
        class_names = open("labels.txt", "r").readlines()
        data = {
            "Class": class_names,
            "Confidence Score": prediction[0],
        }
        df = pd.DataFrame(data)
        df["Confidence Score"] = df["Confidence Score"].apply(
            lambda x: f"{str(np.round(x*100))[:-2]}%"
        )
        df["Class"] = df["Class"].apply(lambda x: x.split(" ")[1])
        st.subheader("Confidence Scores on other classes:")
        st.write(df)

footer = """
<div style="text-align: center; font-size: medium; margin-top:50px;">
        This is the Sub-Part Of the Project Disease Prediction (Helathsense AI).
    <hr>
    <a href="https://github.com/alwaysashutosh" target="_blank">
    <img src="https://img.shields.io/github/stars/alwaysashutosh/alwaysashutosh.svg?style=social" alt="GitHub stars">
  </a>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)