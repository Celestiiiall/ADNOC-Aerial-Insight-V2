import streamlit as st
import torch
import cv2
import numpy as np
import gc
from ultralytics import YOLO

# Function to load YOLO model only when needed (lazy loading)
@st.cache_resource  # Caches model to prevent reloading
def get_model():
    return YOLO("bestt.pt")  # Load the YOLO model only when first called

# Function to process an image
def process_image(image):
    image = cv2.resize(image, (640, 640))  # Resize to reduce memory load
    model = get_model()
    
    with torch.no_grad():  # Disable gradient tracking to save memory
        results = model(image)
    
    del results  # Free memory
    torch.cuda.empty_cache()
    gc.collect()

    return "Processing completed."

# Function to process a video (frame-by-frame)
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    model = get_model()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (640, 640))  # Resize to optimize memory
        
        with torch.no_grad():  # Disable unnecessary computations
            results = model(frame)
        
        del results  # Free memory each frame
        torch.cuda.empty_cache()
        gc.collect()
    
    cap.release()
    return "Video processing completed."

# Streamlit UI
st.set_page_config(page_title="ADNOC Aerial Insight V2", layout="wide")
st.title("🚀 ADNOC Aerial Insight V2")
st.write("Upload an image or video for YOLO-based object detection.")

# File uploader
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    # Handle Image Processing
    if uploaded_file.type.startswith("image"):
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        process_image(image)
        st.success("✅ Image processed successfully.")

    # Handle Video Processing
    elif uploaded_file.type.startswith("video"):
        video_path = f"/tmp/{uploaded_file.name}"  # Temporary file path
        with open(video_path, "wb") as f:
            f.write(file_bytes)
        
        st.video(video_path)
        process_video(video_path)
        st.success("✅ Video processed successfully.")
