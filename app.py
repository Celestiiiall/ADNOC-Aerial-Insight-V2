import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import urllib.request
from ultralytics import YOLO

# Set Streamlit page config
st.set_page_config(page_title="ADNOC Aerial Insight", layout="wide")

# Ensure YOLO model is available
MODEL_PATH = "bestt.pt"
GDRIVE_MODEL_URL = "https://drive.google.com/uc?id=1-GINT3-FjNbBz3INmtudoDgDqKzQRBJt"

if not os.path.exists(MODEL_PATH):
    st.warning("Downloading missing YOLO model (bestt.pt)... This may take a moment.")
    urllib.request.urlretrieve(GDRIVE_MODEL_URL, MODEL_PATH)

# Sidebar with ADNOC branding
LOGO_PATH = "ADNOC-Logo.jpg"
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, use_container_width=True)
else:
    st.sidebar.warning("ADNOC logo missing!")

st.sidebar.markdown(
    "### Developed as part of ADNOC's Analytics & Data Science Initiative."
)

# Main title
st.title("ADNOC Aerial Insight: Collision Detection Platform")
st.markdown(
    "Upload a video, and the system will analyze object movement, detect collisions, and allow downloading the processed output."
)

# File uploader
uploaded_file = st.file_uploader(
    "Upload a Video",
    type=["mp4", "avi", "mov", "mpg", "mpeg4"],
    help="Limit 200MB per file",
)

if uploaded_file:
    # Save the uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())
    
    st.video(video_path)

    # Process video using YOLO
    st.write("Processing video for object detection and collision analysis...")
    model = YOLO(MODEL_PATH)  # Ensure the model path is correct

    output_video_path = os.path.join(temp_dir, "output.mp4")

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.track(source=frame, persist=True, conf=0.3, iou=0.5, tracker="bytetrack.yaml")
        out.write(frame)
    
    cap.release()
    out.release()
    
    st.success("Processing complete!")
    
    with open(output_video_path, "rb") as file:
        btn = st.download_button(
            label="Download Processed Video",
            data=file,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )
