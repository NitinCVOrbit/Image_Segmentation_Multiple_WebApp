import streamlit as st
import numpy as np
from PIL import Image
import base64
from Models import (
    segmentation_deeplabv3_resnet50,
    segmentation_fcn_resnet50,
    segmentation_unet_resnet34
)

# --- Data Dictionary ---
data = {
    "Blood Cell": {
        "class_names": ['bg', 'Blood Cell'],
        "weights_name": "blood_cell.pth",
        "color_sample": [(0, 0, 0), (255, 0, 0)],
        "model_type": "fcn"
    },
    "Car": {
        "class_names": ["bg", "Car", "Wheel", "Lights", "Window"],
        "weights_name": "car.pth",
        "color_sample": [(0, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0)],
        "model_type": "fcn"
    },
    "Cracks": {
        "class_names": ['bg', 'Cracks'],
        "weights_name": "cracks.pth",
        "color_sample": [(0, 0, 0), (255, 165, 0)],
        "model_type": "deeplab"
    },
    "Leaf Disease": {
        "class_names": ['bg', 'Leaf Disease'],
        "weights_name": "leaf.pth",
        "color_sample": [(0, 0, 0), (255, 252, 0)],
        "model_type": "unet"
    },
    "Person": {
        "class_names": ['bg', 'Person'],
        "weights_name": "people.pth",
        "color_sample": [(0, 0, 0), (255, 105, 180)],
        "model_type": "deeplab"
    },
    "Pothole": {
        "class_names": ['bg', 'Pothole'],
        "weights_name": "pothole.pth",
        "color_sample": [(0, 0, 0), (0, 191, 255)],
        "model_type": "unet"
    }
}

# --- Segmentation Logic ---
def apply_segmentation(image: Image.Image, model_data):
    if model_data["model_type"] == "fcn":
        output = segmentation_fcn_resnet50(image, model_data["class_names"], model_data["color_sample"], model_data["weights_name"])
    elif model_data["model_type"] == "deeplab":
        output = segmentation_deeplabv3_resnet50(image, model_data["class_names"], model_data["color_sample"], model_data["weights_name"])
    elif model_data["model_type"] == "unet":
        output = segmentation_unet_resnet34(image, model_data["class_names"], model_data["color_sample"], model_data["weights_name"])
    else:
        raise ValueError("Unsupported model type.")
    return Image.fromarray(output)

# --- Page Configuration ---
st.set_page_config(page_title="Segmentation App", layout="wide")




def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded

bg_image_path = "img.jpg"  # Adjust path as needed
bg_image_encoded = get_base64_image(bg_image_path)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg_image_encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 50px;
        color: #f0f0f0;
        font-weight: bold;
        margin-top: -60px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown('<div class="title">Image Segmentation using Pretrained Models</div>', unsafe_allow_html=True)

# --- Model Selection ---
model_name = st.selectbox("🔍 Choose a Model", list(data.keys()))
model_data = data[model_name]

# --- Upload Image ---
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    segmented_img = None

    col1, col2, col3 = st.columns([1.2, 1, 1.2], gap="large")

    with col1:
        st.image(img, use_container_width=True, caption="📥 Uploaded Image")

    with col2:
        st.markdown("""<br><br>""", unsafe_allow_html=True)
        st.markdown("""
            <style>
            div.stButton > button {
                margin-left: 30%;
            }            
            div[data-testid="stSpinner"] {
                margin-left: 35%;
            }
            </style>
        """, unsafe_allow_html=True)
    
        if st.button("🚀 Run Segmentation"):
            with st.spinner("Segmenting..."):
                segmented_img = apply_segmentation(img, model_data)

    with col3:
        if segmented_img:
            st.image(np.array(segmented_img), use_container_width=True, caption="🎯 Segmented Output")
